"""Phase 7 tests: merge math, DAG gating/resume, GC, registry, watch detect."""

import sys
from pathlib import Path

import pytest  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_studio import manifest  # noqa: E402
from lora_studio.config import Project  # noqa: E402
from lora_studio.maintenance import GcConfig, gc_generator, space_report  # noqa: E402
from lora_studio.merge import (  # noqa: E402
    block_group, merge_available, module_bases, parse_block_string,
)
from lora_studio.pipeline_dag import (  # noqa: E402
    PipelineConfig, clear_state, load_state, save_state,
)
from lora_studio.registry import build_registry, model_card  # noqa: E402

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


# ---------- merge: pure helpers ----------

def test_block_group_mapping():
    assert block_group("lora_te1_text_model_encoder_layers_0_mlp_fc1") == "te"
    assert block_group("lora_unet_down_blocks_1_attentions_0_proj_in") == "down"
    assert block_group("lora_unet_mid_block_attentions_0_proj_out") == "mid"
    assert block_group("lora_unet_up_blocks_2_attentions_1_ff_net") == "up"
    assert block_group("lora_unet_input_blocks_4_1") == "down"


def test_parse_block_string():
    b = parse_block_string("te=0, up=0.5")
    assert b == {"te": 0.0, "down": 1.0, "mid": 1.0, "up": 0.5}
    assert parse_block_string("") == {"te": 1.0, "down": 1.0, "mid": 1.0, "up": 1.0}


def test_module_bases():
    keys = ["m1.lora_down.weight", "m1.lora_up.weight", "m1.alpha",
            "m2.lora_down.weight", "m2.lora_up.weight"]
    assert module_bases(keys) == {"m1", "m2"}


@pytest.mark.skipif(not HAVE_NUMPY, reason="numpy required")
def test_merge_state_dicts_concat_and_negative():
    from lora_studio.merge import merge_state_dicts
    rng = np.random.default_rng(0)

    def make_lora(rank, base="lora_unet_up_blocks_0_x"):
        return {
            f"{base}.lora_down.weight": rng.normal(size=(rank, 16)).astype(np.float32),
            f"{base}.lora_up.weight": rng.normal(size=(32, rank)).astype(np.float32),
            f"{base}.alpha": np.asarray(float(rank), dtype=np.float32),
        }

    a, b = make_lora(4), make_lora(8)   # mixed ranks
    merged, stats = merge_state_dicts([a, b], [1.0, -0.5],
                                      [{"up": 1.0}, {"up": 1.0}])
    base = "lora_unet_up_blocks_0_x"
    assert stats["modules"] == 1
    d = merged[f"{base}.lora_down.weight"]
    u = merged[f"{base}.lora_up.weight"]
    assert d.shape == (12, 16) and u.shape == (32, 12)   # rank 4+8
    assert float(merged[f"{base}.alpha"]) == 12.0
    # effective delta check: up@down == w1*BA_1 + w2*BA_2 (alpha/rank = 1 here)
    expect = (a[f"{base}.lora_up.weight"] @ a[f"{base}.lora_down.weight"]
              - 0.5 * b[f"{base}.lora_up.weight"] @ b[f"{base}.lora_down.weight"])
    got = u.astype(np.float32) @ d.astype(np.float32)
    assert np.allclose(got, expect, atol=0.05)
    # zero block multiplier drops the module entirely
    merged2, stats2 = merge_state_dicts([a, b], [1.0, 1.0],
                                        [{"up": 0.0}, {"up": 0.0}])
    assert stats2["modules"] == 0 and stats2["zeroed"] == 2


# ---------- DAG state ----------

def test_dag_state_roundtrip(tmp_path):
    conn = manifest.connect(tmp_path)
    cfg = PipelineConfig(output_base=tmp_path, recipe="r1", preset="style",
                         gates=["curate"])
    save_state(conn, cfg, "caption")
    st = load_state(conn)
    assert st["next_stage"] == "caption" and st["preset"] == "style"
    clear_state(conn)
    assert load_state(conn) is None


def test_dag_gate_pauses(tmp_path):
    """Gated pipeline stops after the gated stage and records the next one."""
    from lora_studio import pipeline_dag

    calls = []

    def fake_stages(prj, cfg):
        def mk(name):
            def g():
                calls.append(name)
                yield f"{name} ran"
            return g
        return [(n, mk(n)) for n in ("extract", "curate", "caption")]

    orig = pipeline_dag._stage_generators
    pipeline_dag._stage_generators = fake_stages
    try:
        cfg = PipelineConfig(output_base=tmp_path, gates=["curate"])
        out = list(pipeline_dag.pipeline_generator(Project(), cfg))
        assert calls == ["extract", "curate"]          # caption NOT run
        assert any("GATED" in u for u in out)
        conn = manifest.connect(tmp_path)
        assert load_state(conn)["next_stage"] == "caption"
        # resume runs the rest and clears state
        out2 = list(pipeline_dag.resume_generator(Project(), tmp_path))
        assert calls == ["extract", "curate", "caption"]
        assert any("PIPELINE COMPLETE" in u for u in out2)
        assert load_state(conn) is None
    finally:
        pipeline_dag._stage_generators = orig


# ---------- GC / space / registry ----------

def test_gc_orphans_and_apply(tmp_path):
    conn = manifest.connect(tmp_path)
    thumbs = tmp_path / ".thumbs"
    thumbs.mkdir()
    conn.execute("INSERT INTO frames(frame_id, path) VALUES ('keepme', 'x.jpg')")
    conn.commit()
    (thumbs / "keepme.jpg").write_bytes(b"k" * 10)
    (thumbs / "orphan1.jpg").write_bytes(b"o" * 100)
    prj = Project()
    out = list(gc_generator(prj, GcConfig(output_base=tmp_path, apply=False)))
    assert any("Orphan thumbnails: 1" in u for u in out)
    assert (thumbs / "orphan1.jpg").exists()           # dry run keeps it
    list(gc_generator(prj, GcConfig(output_base=tmp_path, apply=True)))
    assert not (thumbs / "orphan1.jpg").exists()
    assert (thumbs / "keepme.jpg").exists()


def test_space_report(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "a.jpg").write_bytes(b"x" * 1000)
    rows = space_report(Project(), tmp_path)
    frames = next(r for r in rows if r["area"] == "frames")
    assert frames["files"] == 1 and frames["bytes"] == 1000


def test_registry_and_card(tmp_path):
    conn = manifest.connect(tmp_path)
    lora_dir = tmp_path / "LORA_OUTPUT"
    lora_dir.mkdir()
    (lora_dir / "tok_character_v001.safetensors").write_bytes(b"f" * 10)
    (lora_dir / "random_external.safetensors").write_bytes(b"f" * 10)
    conn.execute(
        "INSERT INTO runs(name, dataset_version, preset, status, started_at, last_loss) "
        "VALUES ('tok_character_v001', 1, 'character', 'completed', 't', 0.05)")
    conn.execute(
        "INSERT INTO datasets(version, recipe_name, content_hash, built_at, "
        "image_count, val_count, dir) VALUES (1,'char_v1','abc','t',100,5,'d')")
    conn.execute(
        "INSERT INTO evals(lora, label, category, likeness, created_at) "
        "VALUES ('x','tok_character_v001','likeness',0.82,'t')")
    conn.commit()
    prj = Project(output_base=str(tmp_path), lora_output_dir=str(lora_dir))
    entries = build_registry(prj, conn)
    names = {e["name"]: e for e in entries}
    assert names["tok_character_v001"]["kind"] == "trained"
    assert names["random_external"]["kind"] == "external"
    assert names["tok_character_v001"]["evals"]["likeness"]["avg"] == 0.82
    card = model_card(prj, conn, "tok_character_v001")
    text = card.read_text()
    assert "v001 (char_v1" in text and "0.82" in text and "trained" in text
