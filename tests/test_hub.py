"""v2.9 hub tests: app manager, mega-pipeline sequencing, preset writer."""

import json
import sys
from pathlib import Path

import pytest  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_studio import manifest  # noqa: E402
from lora_studio.config import Project  # noqa: E402
from lora_studio.pipeline_dag import write_playground_preset  # noqa: E402
from lora_studio.ui.server import _app_script, _port_open, app_status  # noqa: E402


def test_app_scripts_discoverable():
    st = app_status()
    # both companion apps live next to the package in this repo
    assert st["playground"]["script_found"]
    assert st["factory"]["script_found"]
    assert _app_script("playground").name == "grokkie_playground.py"
    assert st["playground"]["port"] == 7870 and st["factory"]["port"] == 7875


def test_port_open_false_for_unused():
    assert _port_open(54329) is False


def test_write_playground_preset(tmp_path):
    prj = Project(trigger_token="tok", class_word="person")
    target = write_playground_preset(
        prj, "studio_tok", [("tok_char_v001-000004", 0.85)],
        tmp_path / "presets.json")
    data = json.loads(target.read_text())
    p = data["studio_tok"]
    assert p["loras"] == [["tok_char_v001-000004", 0.85]]
    assert "tok person" in p["prompt"]
    # merge-into-existing keeps other presets
    write_playground_preset(prj, "second", [("x", 1.0)], target)
    data = json.loads(target.read_text())
    assert set(data) == {"studio_tok", "second"}


def test_ui_javascript_parses():
    """Guard against Python-string escapes corrupting the hub's JS.

    A single un-escaped \\n inside a quoted JS string kills the ENTIRE
    script (dead sidebar, dead handlers) - exactly the v2.9 regression.
    Uses node --check when available, plus a quote-balance heuristic."""
    import re
    import shutil
    import subprocess
    import tempfile
    from lora_studio.ui.server import UI_HTML

    js = re.search(r"<script>(.*)</script>", UI_HTML, re.S).group(1)

    # heuristic: a literal newline must never split a single-quoted string
    in_template = False
    for i, line in enumerate(js.splitlines(), 1):
        in_template ^= (line.count("`") % 2 == 1)
        if in_template:
            continue
        stripped = re.sub(r"\\.", "", line)
        stripped = re.sub(r"`[^`]*`", "", stripped)
        assert stripped.count("'") % 2 == 0, \
            f"unbalanced quote (likely broken \\n escape) on JS line {i}: {line[:100]}"

    node = shutil.which("node")
    if node:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(js)
        proc = subprocess.run([node, "--check", fh.name],
                              capture_output=True, text=True)
        assert proc.returncode == 0, f"node --check failed:\n{proc.stderr[:500]}"


def test_mega_sequencing_and_stop_on_failure(tmp_path):
    """Mega chain runs stages in order, stops cleanly when training fails."""
    from lora_studio import pipeline_dag as pd

    calls: list[str] = []

    def fake_stages(prj, cfg):
        def mk(name):
            def g():
                calls.append(name)
                yield f"{name} ok"
            return g
        return [(n, mk(n)) for n in
                ("extract", "curate", "caption", "build", "train", "matrix")]

    import lora_studio.wizard as wz

    def fake_wizard(prj, cfg):
        calls.append("wizard")
        yield "wizard ran (no training in sandbox)"

    orig_stages, orig_wiz = pd._stage_generators, wz.wizard_generator
    pd._stage_generators = fake_stages
    wz.wizard_generator = fake_wizard
    try:
        # no completed run in DB -> chain must stop after wizard with guidance
        manifest.connect(tmp_path).close()
        out = list(pd.mega_generator(Project(), pd.MegaConfig(output_base=tmp_path)))
        text = "\n".join(out)
        assert calls == ["extract", "curate", "caption", "wizard"]
        assert "STOPPED: training did not complete" in text

        # with a completed run + checkpoint -> proceeds to sweep + preset
        conn = manifest.connect(tmp_path)
        conn.execute("INSERT INTO runs(name, status) VALUES ('tok_char_v001','completed')")
        conn.commit()
        lora_dir = tmp_path / "LORA_OUTPUT"
        lora_dir.mkdir()
        (lora_dir / "tok_char_v001.safetensors").write_bytes(b"x")
        prj = Project(output_base=str(tmp_path), lora_output_dir=str(lora_dir),
                      trigger_token="tok")
        cfg = pd.MegaConfig(output_base=tmp_path,
                            presets_path=str(tmp_path / "pp.json"))
        out2 = "\n".join(pd.mega_generator(prj, cfg))
        assert "===== sweep" in out2
        assert "FULL PIPELINE COMPLETE" in out2
        data = json.loads((tmp_path / "pp.json").read_text())
        assert "studio_tok" in data
        assert data["studio_tok"]["loras"][0][0].startswith("tok_char_v001")
    finally:
        pd._stage_generators = orig_stages
        wz.wizard_generator = orig_wiz
