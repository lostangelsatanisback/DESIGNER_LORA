"""Centralized device / accelerator runtime for GROKKIE Studio.

One resolver for every AI-heavy module so the same repo runs on:

    macOS Apple Silicon  -> mps   (existing behavior, preserved)
    Colab / Linux + A100 -> cuda  (tf32 + fp16/bf16)
    anything else        -> cpu   (float32)

Selection order for "auto": cuda -> mps -> cpu.

Everything here is torch-optional and never raises at import time: if
torch is missing the resolver reports "cpu" and the doctor output says
why.  Configuration comes from the additive [runtime] section of the
project file (see config.RuntimeConfig); absence of the section keeps
all previous defaults.
"""
from __future__ import annotations

import contextlib
import os
import platform
import sys
from typing import Optional

VALID_DEVICES = ("auto", "cuda", "mps", "cpu")

_TF32_APPLIED = False


def _torch():
    """Import torch lazily; returns None when unavailable."""
    try:
        import torch
        return torch
    except Exception:
        return None


def is_cuda_available() -> bool:
    t = _torch()
    try:
        return bool(t and t.cuda.is_available())
    except Exception:
        return False


def is_mps_available() -> bool:
    t = _torch()
    try:
        return bool(t and hasattr(t.backends, "mps")
                    and t.backends.mps.is_available())
    except Exception:
        return False


def in_colab() -> bool:
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def get_device_name(preferred: str = "auto") -> str:
    """Resolve 'auto'/'cuda'/'mps'/'cpu' to the device string to use.
    Explicit requests fall back (with no exception) when unavailable."""
    p = (preferred or "auto").strip().lower()
    if p not in VALID_DEVICES:
        p = "auto"
    if p == "cuda":
        return "cuda" if is_cuda_available() else "cpu"
    if p == "mps":
        return "mps" if is_mps_available() else "cpu"
    if p == "cpu":
        return "cpu"
    # auto: cuda first, then mps, then cpu
    if is_cuda_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"


def get_torch_device(preferred: str = "auto"):
    """torch.device for the resolved accelerator (raises only if torch
    itself is missing - callers that reach this already import torch)."""
    import torch
    return torch.device(get_device_name(preferred))


def get_recommended_dtype(device=None, precision: str = "auto"):
    """Dtype policy:
        cuda: bf16 when supported (A100+), else fp16
        mps : fp16 (matches existing pipeline behavior)
        cpu : fp32
    precision: auto|float32|float16|bfloat16 (explicit always wins)."""
    import torch
    name = str(device or get_device_name("auto"))
    name = name.split(":")[0]
    p = (precision or "auto").strip().lower()
    explicit = {"float32": torch.float32, "fp32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    if p in explicit:
        return explicit[p]
    if name == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    if name == "mps":
        return torch.float16
    return torch.float32


def apply_cuda_settings(allow_tf32: bool = True) -> None:
    """One-time CUDA niceties (TF32 matmul). No-op off-CUDA."""
    global _TF32_APPLIED
    t = _torch()
    if not t or not is_cuda_available():
        return
    try:
        t.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        t.backends.cudnn.allow_tf32 = bool(allow_tf32)
        _TF32_APPLIED = bool(allow_tf32)
    except Exception:
        pass


def clear_accelerator_cache(device=None) -> None:
    """Free cached accelerator memory between heavy jobs. Never raises."""
    import gc
    gc.collect()
    t = _torch()
    if not t:
        return
    name = str(device or get_device_name("auto")).split(":")[0]
    try:
        if name == "cuda" and is_cuda_available():
            t.cuda.empty_cache()
        elif name == "mps" and is_mps_available():
            t.mps.empty_cache()
    except Exception:
        pass


@contextlib.contextmanager
def autocast_context(device=None, dtype=None):
    """torch.autocast for cuda; pass-through (nullcontext) elsewhere -
    MPS autocast support is uneven, existing Mac behavior is preserved."""
    t = _torch()
    name = str(device or get_device_name("auto")).split(":")[0]
    if t and name == "cuda" and is_cuda_available():
        dt = dtype or get_recommended_dtype("cuda")
        with t.autocast(device_type="cuda", dtype=dt):
            yield
    else:
        yield


def onnx_providers() -> list[str]:
    """Ordered onnxruntime execution providers for this machine:
    CUDA (Colab/Linux) -> CoreML (macOS) -> CPU.  Only providers that
    onnxruntime actually reports available are returned."""
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    out = []
    for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider"):
        if p in avail:
            out.append(p)
    out.append("CPUExecutionProvider")
    return out


def get_accelerator_info(preferred: str = "auto",
                         precision: str = "auto") -> dict:
    """Serializable runtime report for doctor / UI."""
    t = _torch()
    info: dict = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": getattr(t, "__version__", None) if t else None,
        "cuda_available": is_cuda_available(),
        "mps_available": is_mps_available(),
        "in_colab": in_colab(),
        "selected_device": get_device_name(preferred),
        "tf32_enabled": _TF32_APPLIED,
        "cuda_device": None,
        "cuda_vram_gb": None,
        "selected_dtype": None,
    }
    if t:
        try:
            info["selected_dtype"] = str(
                get_recommended_dtype(info["selected_device"], precision)
            ).replace("torch.", "")
        except Exception:
            pass
    if info["cuda_available"]:
        try:
            props = t.cuda.get_device_properties(0)
            info["cuda_device"] = props.name
            info["cuda_vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
        except Exception:
            pass
    return info


def configure_from_project(prj) -> dict:
    """Apply a project's [runtime] settings (tf32 etc.) and return the
    resolved info dict.  Safe with any Project object (older ones have
    no runtime field)."""
    rc = getattr(prj, "runtime", None) or {}
    if not isinstance(rc, dict):
        rc = {}
    device = str(rc.get("device", "auto"))
    precision = str(rc.get("precision", "auto"))
    if get_device_name(device) == "cuda":
        apply_cuda_settings(bool(rc.get("allow_tf32", True)))
    if is_mps_available() and rc.get("mps_fallback", True):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return get_accelerator_info(device, precision)
