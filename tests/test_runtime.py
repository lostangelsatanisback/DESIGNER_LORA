"""Runtime/device abstraction tests - all accelerators mocked, no GPU
required.  A fake torch module is injected into sys.modules so the same
assertions run identically on Mac, Colab and CI."""
import json
import sys
import types

from lora_studio import runtime
from lora_studio.config import Project, load_project


class _FakeCudaProps:
    name = "NVIDIA A100-SXM4-80GB"
    total_memory = 80 * 1024 ** 3


def _fake_torch(cuda=False, mps=False, bf16=True):
    t = types.ModuleType("torch")
    t.__version__ = "2.fake"
    t.float16, t.float32, t.bfloat16 = "float16", "float32", "bfloat16"
    t.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        is_bf16_supported=lambda: bf16,
        get_device_properties=lambda i: _FakeCudaProps(),
        empty_cache=lambda: None)
    t.mps = types.SimpleNamespace(empty_cache=lambda: None)
    t.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps),
        cuda=types.SimpleNamespace(
            matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False))
    t.device = lambda name: f"device({name})"
    return t


def _with_torch(monkey_torch):
    """Install fake torch; returns a restore function."""
    old = sys.modules.get("torch")
    sys.modules["torch"] = monkey_torch
    def restore():
        if old is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = old
    return restore


def test_auto_prefers_cuda_then_mps_then_cpu():
    for cuda, mps, expect in ((True, True, "cuda"), (False, True, "mps"),
                              (False, False, "cpu")):
        restore = _with_torch(_fake_torch(cuda=cuda, mps=mps))
        try:
            assert runtime.get_device_name("auto") == expect
        finally:
            restore()


def test_explicit_request_falls_back_gracefully():
    restore = _with_torch(_fake_torch(cuda=False, mps=False))
    try:
        assert runtime.get_device_name("cuda") == "cpu"
        assert runtime.get_device_name("mps") == "cpu"
        assert runtime.get_device_name("nonsense") == "cpu"
    finally:
        restore()


def test_dtype_policy():
    restore = _with_torch(_fake_torch(cuda=True, bf16=True))
    try:
        assert runtime.get_recommended_dtype("cuda") == "bfloat16"
        assert runtime.get_recommended_dtype("mps") == "float16"
        assert runtime.get_recommended_dtype("cpu") == "float32"
        # explicit precision always wins
        assert runtime.get_recommended_dtype("cuda", "float32") == "float32"
        assert runtime.get_recommended_dtype("cpu", "fp16") == "float16"
    finally:
        restore()
    # A100 without bf16 -> fp16
    restore = _with_torch(_fake_torch(cuda=True, bf16=False))
    try:
        assert runtime.get_recommended_dtype("cuda") == "float16"
    finally:
        restore()


def test_accelerator_info_serializable_and_complete():
    restore = _with_torch(_fake_torch(cuda=True))
    try:
        info = runtime.get_accelerator_info("auto")
        json.dumps(info)                       # must be JSON-safe
        assert info["selected_device"] == "cuda"
        assert info["cuda_device"] == "NVIDIA A100-SXM4-80GB"
        assert info["cuda_vram_gb"] == 80.0
        for key in ("platform", "python", "torch", "mps_available",
                    "in_colab", "selected_dtype"):
            assert key in info
    finally:
        restore()


def test_no_torch_degrades_to_cpu():
    broken = types.ModuleType("torch")
    # module without expected attrs -> all probes must swallow errors
    restore = _with_torch(broken)
    try:
        assert runtime.get_device_name("auto") == "cpu"
        assert runtime.is_cuda_available() is False
        info = runtime.get_accelerator_info()
        assert info["selected_device"] == "cpu"
    finally:
        restore()


def test_apply_cuda_settings_sets_tf32():
    fake = _fake_torch(cuda=True)
    restore = _with_torch(fake)
    try:
        runtime.apply_cuda_settings(True)
        assert fake.backends.cuda.matmul.allow_tf32 is True
        assert fake.backends.cudnn.allow_tf32 is True
    finally:
        restore()


def test_clear_cache_never_raises():
    restore = _with_torch(_fake_torch(cuda=True))
    try:
        runtime.clear_accelerator_cache("cuda")
        runtime.clear_accelerator_cache("mps")
        runtime.clear_accelerator_cache("cpu")
    finally:
        restore()


def test_config_runtime_section_parsed(tmp_path):
    p = tmp_path / "p.toml"
    p.write_text('name = "X"\ntrigger_token = "tok"\n\n'
                 '[runtime]\ndevice = "cuda"\nprecision = "bfloat16"\n'
                 'allow_tf32 = true\n')
    prj = load_project(str(p))
    assert prj.runtime.get("device") == "cuda"
    assert prj.runtime.get("precision") == "bfloat16"
    assert prj.trigger_token == "tok"      # other fields unaffected


def test_config_runtime_absent_defaults():
    prj = Project()
    assert prj.runtime == {}
    # configure_from_project works with empty runtime + any torch state
    restore = _with_torch(_fake_torch(cuda=False, mps=False))
    try:
        info = runtime.configure_from_project(prj)
        assert info["selected_device"] == "cpu"
    finally:
        restore()


def test_onnx_providers_cpu_fallback(monkeypatch=None):
    # onnxruntime importable in dev env or not - either way the list must
    # end with CPUExecutionProvider and contain no unavailable providers
    provs = runtime.onnx_providers()
    assert provs[-1] == "CPUExecutionProvider"
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
        assert all(p in avail or p == "CPUExecutionProvider" for p in provs)
    except ImportError:
        assert provs == ["CPUExecutionProvider"]
