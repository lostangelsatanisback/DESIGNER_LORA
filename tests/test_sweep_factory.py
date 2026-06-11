"""Best-Epoch Sweep + stack planner + factory logic tests."""

import io
import sys
from pathlib import Path

import pytest  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_studio import manifest  # noqa: E402
from lora_studio.config import Project  # noqa: E402
from lora_studio.sweep import (  # noqa: E402
    SweepConfig, epoch_number, find_epoch_files, recommend_epoch,
    sweep_generator, sweep_summary,
)
from lora_studio.wizard import suggest_stack  # noqa: E402

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


# ---------- epoch parsing ----------

def test_epoch_number():
    assert epoch_number("run-000004") == 4
    assert epoch_number("run") is None
    assert epoch_number("run-12") is None          # kohya uses 6 digits


def test_find_epoch_files(tmp_path):
    for n in (1, 2, 3):
        (tmp_path / f"tok_char_v001-{n:06d}.safetensors").write_bytes(b"x")
    (tmp_path / "tok_char_v001.safetensors").write_bytes(b"x")       # final
    (tmp_path / "tok_char_v002.safetensors").write_bytes(b"x")       # other run
    epochs = find_epoch_files(tmp_path, "tok_char_v001")
    assert [e for e, _ in epochs] == [1, 2, 3, 4]                    # final = 4
    assert epochs[-1][1].stem == "tok_char_v001"
    # other run untouched
    assert all("v002" not in p.stem for _, p in epochs)


# ---------- recommendation ----------

def test_recommend_epoch_overfit_guard():
    rows = [
        {"epoch": 1, "label": "e1", "likeness": 0.60, "flexibility": 0.58},
        {"epoch": 2, "label": "e2", "likeness": 0.78, "flexibility": 0.70},
        {"epoch": 3, "label": "e3", "likeness": 0.85, "flexibility": 0.60},  # gap .25
    ]
    best = recommend_epoch(rows, max_gap=0.15)
    assert best["epoch"] == 2 and not best["overfit_warning"]   # 3 is overfit
    # all overfit -> falls back to max likeness, flagged
    best2 = recommend_epoch(rows, max_gap=0.01)
    assert best2["epoch"] == 3 and best2["overfit_warning"]
    assert recommend_epoch([{"epoch": 1, "label": "x", "likeness": None}]) is None


def test_recommend_epoch_no_flexibility():
    rows = [{"epoch": 1, "label": "a", "likeness": 0.5, "flexibility": None},
            {"epoch": 2, "label": "b", "likeness": 0.7, "flexibility": None}]
    assert recommend_epoch(rows)["epoch"] == 2


# ---------- sweep run (fake backend, resumable) ----------

@pytest.mark.skipif(not HAVE_PIL, reason="Pillow required")
def test_sweep_runs_and_resumes(tmp_path):
    lora_dir = tmp_path / "LORA_OUTPUT"
    lora_dir.mkdir()
    for n in (1, 2):
        (lora_dir / f"run_x-{n:06d}.safetensors").write_bytes(b"x")
    prj = Project(output_base=str(tmp_path), lora_output_dir=str(lora_dir),
                  trigger_token="tok")
    calls = []

    def fake_gen(prompt, seed):
        calls.append(seed)
        img = Image.new("RGB", (32, 32), (10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    cfg = SweepConfig(output_base=tmp_path, run="run_x",
                      categories=["likeness"], seeds=[1],
                      generate_fn=fake_gen)
    out = list(sweep_generator(prj, cfg))
    text = "\n".join(out)
    assert "EPOCH SCOREBOARD" in text
    assert "No likeness scores" in text            # no anchor in sandbox
    n_first = len(calls)
    assert n_first > 0

    # resumability: second sweep skips all epochs (evals exist)
    out2 = list(sweep_generator(prj, cfg))
    assert len(calls) == n_first                   # no new generations
    assert "already evaluated" in "\n".join(out2)
    # state stored
    conn = manifest.connect(tmp_path)
    assert sweep_summary(conn, "run_x") is not None


# ---------- stack planner ----------

def _ranking(**scores):
    base = {"character": 0.2, "style": 0.2, "outfit": 0.2, "pose": 0.2,
            "detail": 0.2, "explicit": 0.0}
    base.update(scores)
    return sorted([{"type": t, "score": s, "reason": "r"} for t, s in base.items()],
                  key=lambda d: -d["score"])


def test_suggest_stack_character_full_pack():
    analysis = {"framing_mix": {"closeup": 0.35, "portrait": 0.3},
                "clusters": 5, "sharpness_mean": 90}
    plan = suggest_stack(_ranking(character=0.8, style=0.5, outfit=0.6),
                         analysis)
    types = [s["type"] for s in plan["stack"]]
    assert types[0] == "character"
    assert "style" in types and "outfit" in types and "detail" in types
    style = next(s for s in plan["stack"] if s["type"] == "style")
    assert style["blocks"]["te"] == 0.0 and style["blocks"]["up"] == 1.0
    assert plan["merge_name"].startswith("character_pack")


def test_suggest_stack_minimal_when_weak_signals():
    plan = suggest_stack(_ranking(character=0.7, style=0.2, outfit=0.2),
                         {"framing_mix": {}, "clusters": 1, "sharpness_mean": 20})
    assert len(plan["stack"]) == 1                 # primary only


def test_suggest_stack_style_primary_pairs_identity():
    plan = suggest_stack(_ranking(style=0.8, character=0.6),
                         {"framing_mix": {}, "clusters": 4, "sharpness_mean": 50})
    types = [s["type"] for s in plan["stack"]]
    assert types[0] == "style" and "character" in types


# ---------- factory ship logic ----------

def test_factory_ship_writes_playground_preset(tmp_path, monkeypatch=None):
    import grokkie_dataset_factory as gdf
    if not gdf.HAVE_STUDIO:
        return
    sess = gdf.FactorySession(None)
    sess.prj = Project(output_base=str(tmp_path), trigger_token="tok",
                       class_word="person")
    sess.merged_path = str(tmp_path / "tok_pack_v1.safetensors")
    Path(sess.merged_path).write_bytes(b"x")
    target = tmp_path / "presets.json"
    msg = gdf.step_ship(sess, "myset", str(target))
    assert "Shipped preset 'myset'" in msg
    import json
    data = json.loads(target.read_text())
    assert data["myset"]["loras"] == [["tok_pack_v1", 0.85]]
    assert "tok person" in data["myset"]["prompt"]
