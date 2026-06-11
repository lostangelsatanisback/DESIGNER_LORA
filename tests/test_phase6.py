"""Phase 6 tests: lora-spec parsing, prompt composition, forge payloads,
grid math, and the full matrix generator against a fake backend."""

import io
import sys
from pathlib import Path

import pytest  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_studio import manifest  # noqa: E402
from lora_studio.config import Project  # noqa: E402
from lora_studio.eval.forge_api import ForgeClient  # noqa: E402
from lora_studio.eval.matrix import (  # noqa: E402
    CATEGORIES, MatrixConfig, compose_prompt, eval_summary, grid_layout,
    matrix_generator, parse_lora_specs,
)

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


def test_parse_lora_specs():
    specs = parse_lora_specs("/a/b/char_v3.safetensors:0.85, /x/neg.safetensors:-0.4, /p/plain.safetensors")
    assert specs == [
        ("/a/b/char_v3.safetensors", 0.85),
        ("/x/neg.safetensors", -0.4),
        ("/p/plain.safetensors", 1.0),
    ]
    assert parse_lora_specs("") == []


def test_compose_prompt_and_categories():
    prj = Project(trigger_token="spook", class_word="person")
    p = compose_prompt(CATEGORIES["likeness"][0], prj, pony_prefix=True)
    assert p.startswith("score_9, score_8_up, score_7_up, ")
    assert "spook person" in p
    p2 = compose_prompt("{t}, test", prj, pony_prefix=False)
    assert p2 == "spook person, test"
    assert set(CATEGORIES) == {"likeness", "flexibility", "pose", "outfit", "style"}
    assert all(len(v) >= 3 for v in CATEGORIES.values())


def test_forge_payload_lora_tags():
    payload = ForgeClient.build_txt2img_payload(
        prompt="hello", seed=7,
        loras=[("/m/spook_char_v003.safetensors", 0.8), ("/m/det.safetensors", -0.3)],
    )
    assert "<lora:spook_char_v003:0.8>" in payload["prompt"]
    assert "<lora:det:-0.3>" in payload["prompt"]
    assert payload["seed"] == 7 and "enable_hr" not in payload
    hires = ForgeClient.build_txt2img_payload(prompt="x", hires=True)
    assert hires["enable_hr"] is True


def test_grid_layout():
    assert grid_layout(12, 3) == (4, 3)
    assert grid_layout(1, 3) == (1, 3)
    assert grid_layout(7, 3) == (3, 3)


def _fake_png() -> bytes:
    img = Image.new("RGB", (64, 64), (40, 120, 200))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.mark.skipif(not HAVE_PIL, reason="Pillow required")
def test_matrix_with_fake_backend(tmp_path):
    prj = Project(trigger_token="spook", class_word="person")
    calls: list[tuple[str, int]] = []

    def fake_gen(prompt: str, seed: int) -> bytes:
        calls.append((prompt, seed))
        return _fake_png()

    cfg = MatrixConfig(
        output_base=tmp_path, lora="/m/spook_char_v003.safetensors",
        categories=["likeness", "flexibility"], seeds=[1, 2],
        generate_fn=fake_gen,
    )
    updates = list(matrix_generator(prj, cfg))
    assert "MATRIX DONE" in updates[-1]

    n_expected = (len(CATEGORIES["likeness"]) + len(CATEGORIES["flexibility"])) * 2
    assert len(calls) == n_expected
    # deterministic seeds and trigger in every prompt
    assert all("spook person" in p for p, _ in calls)
    assert {s for _, s in calls} == {1, 2}

    # files + grids on disk
    out_dir = tmp_path / "evals" / "spook_char_v003"
    assert len(list(out_dir.glob("*.png"))) == n_expected
    assert (out_dir / "GRID_likeness.jpg").exists()
    assert (out_dir / "GRID_flexibility.jpg").exists()

    # eval rows recorded; summary aggregates
    conn = manifest.connect(tmp_path)
    n_rows = conn.execute("SELECT COUNT(*) FROM evals").fetchone()[0]
    assert n_rows == n_expected
    summary = eval_summary(conn)
    cats = {s["category"] for s in summary}
    assert cats == {"likeness", "flexibility"}


@pytest.mark.skipif(not HAVE_PIL, reason="Pillow required")
def test_matrix_handles_backend_errors(tmp_path):
    prj = Project()
    flake = {"n": 0}

    def flaky(prompt, seed):
        flake["n"] += 1
        if flake["n"] % 2 == 0:
            raise RuntimeError("backend hiccup")
        return _fake_png()

    cfg = MatrixConfig(output_base=tmp_path, lora="x.safetensors",
                       categories=["likeness"], seeds=[1], generate_fn=flaky)
    updates = list(matrix_generator(prj, cfg))
    assert "MATRIX DONE" in updates[-1]          # survives failures
    assert any("ERR" in u for u in updates)      # and reports them
