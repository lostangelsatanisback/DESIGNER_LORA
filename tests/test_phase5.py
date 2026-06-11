"""Phase 4-finish + Phase 5 tests: crop math, presets, kohya adapter
(command builder, dry-run, stub-trainer progress parsing into runs/metrics)."""

import json
import sys
import textwrap
from pathlib import Path

import pytest  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_studio import manifest  # noqa: E402
from lora_studio.config import Project  # noqa: E402
from lora_studio.crop import (  # noqa: E402
    choose_bucket, crop_box_for_poi, poi_from_bbox, scaled_buckets,
)
from lora_studio.train.kohya import (  # noqa: E402
    TrainConfig, list_runs, run_metrics, train_generator,
)
from lora_studio.train.presets import PRESETS  # noqa: E402


# ---------- crop math ----------

def test_choose_bucket():
    assert choose_bucket(1000, 1000) == (1024, 1024)
    assert choose_bucket(1920, 1080) in ((1344, 768), (1216, 832))
    assert choose_bucket(1080, 1920) in ((768, 1344), (832, 1216))
    w, h = choose_bucket(640, 640, base=768)
    assert w == h and w % 64 == 0 and w <= 768


def test_scaled_buckets_aligned():
    for w, h in scaled_buckets(768):
        assert w % 64 == 0 and h % 64 == 0


def test_crop_box_centered_and_clamped():
    # face on far left -> window clamps to left edge
    rw, rh, l, t, r, b = crop_box_for_poi(2000, 1000, 1344, 768, poi_x=0.02, poi_y=0.5)
    assert l == 0 and (r - l, b - t) == (1344, 768)
    assert r <= rw and b <= rh
    # centered POI -> centered window
    rw, rh, l, t, r, b = crop_box_for_poi(2000, 1000, 1344, 768, 0.5, 0.5)
    assert abs((l + r) / 2 - rw / 2) <= 1


def test_poi_from_bbox():
    assert poi_from_bbox(None) == (0.5, 0.5)
    assert poi_from_bbox("notjson") == (0.5, 0.5)
    px, py = poi_from_bbox(json.dumps([0.4, 0.2, 0.6, 0.4]))
    assert abs(px - 0.5) < 1e-9
    assert py > 0.3  # nudged below face center for headroom


# ---------- presets ----------

def test_presets_cover_roadmap_table():
    assert set(PRESETS) == {"character", "style", "outfit", "pose", "detail"}
    c = PRESETS["character"]
    assert (c["network_dim"], c["network_alpha"]) == (32, 16)
    assert c["unet_lr"] == 1e-4 and c["te_lr"] == 5e-5
    assert PRESETS["style"]["te_lr"] == 0.0
    assert PRESETS["detail"]["network_dim"] == 64


# ---------- adapter ----------

def _seed_dataset(out: Path) -> None:
    conn = manifest.connect(out)
    (out / "DATASET" / "v001__t" / "img").mkdir(parents=True, exist_ok=True)
    conn.execute(
        "INSERT INTO datasets(version, recipe_name, content_hash, built_at, "
        "image_count, val_count, dir) VALUES (1,'t','h','now',4,0,?)",
        (str(out / "DATASET" / "v001__t"),),
    )
    conn.commit()
    conn.close()


def test_dry_run_command(tmp_path):
    _seed_dataset(tmp_path)
    prj = Project(trigger_token="tok", sd_scripts_dir=str(tmp_path),
                  base_model="/models/pony.safetensors")
    (tmp_path / "sdxl_train_network.py").write_text("# stub")
    cfg = TrainConfig(output_base=tmp_path, dataset_version=1,
                      preset="character", dry_run=True)
    out = "\n".join(train_generator(prj, cfg))
    assert "--network_dim 32" in out and "--network_alpha 16" in out
    assert "--text_encoder_lr 5e-05" in out
    assert "--min_snr_gamma 5" in out and "--gradient_checkpointing" in out
    assert "tok_character_v001" in out
    # style preset freezes TE
    out2 = "\n".join(train_generator(
        prj, TrainConfig(output_base=tmp_path, dataset_version=1,
                         preset="style", dry_run=True)))
    assert "--network_train_unet_only" in out2
    assert "--text_encoder_lr" not in out2


def test_stub_trainer_run_tracking(tmp_path):
    """Full adapter loop against a fake kohya process emitting progress lines."""
    _seed_dataset(tmp_path)
    stub = tmp_path / "stub_trainer.py"
    stub.write_text(textwrap.dedent("""
        import sys, time
        for e in (1, 2):
            print(f"epoch {e}/2", flush=True)
            for s in range(1, 6):
                step = (e - 1) * 5 + s
                loss = 0.5 / step
                print(f"steps: {step}/10 [00:01, avr_loss={loss:.4f}]", flush=True)
                time.sleep(0.01)
        print("training complete", flush=True)
    """))
    prj = Project(trigger_token="tok")
    cfg = TrainConfig(
        output_base=tmp_path, dataset_version=1, preset="pose",
        trainer_cmd=[sys.executable, str(stub)],
    )
    updates = list(train_generator(prj, cfg))
    assert "COMPLETED" in updates[-1]

    conn = manifest.connect(tmp_path)
    runs = list_runs(conn)
    assert len(runs) == 1
    r = runs[0]
    assert r["status"] == "completed"
    assert r["step"] == 10 and r["total_steps"] == 10
    assert r["epoch"] == 2
    assert abs(r["last_loss"] - 0.05) < 1e-6
    pts = run_metrics(conn, r["run_id"])
    assert len(pts) == 10
    assert pts[0][1] > pts[-1][1]  # loss decreased
    # run metadata written
    meta = json.loads((tmp_path / "runs" / r["name"] / "run.json").read_text())
    assert meta["status"] == "completed" and meta["preset"] == "pose"


def test_failed_trainer_marks_error(tmp_path):
    _seed_dataset(tmp_path)
    stub = tmp_path / "bad.py"
    stub.write_text("import sys; print('boom'); sys.exit(3)")
    prj = Project()
    cfg = TrainConfig(output_base=tmp_path, dataset_version=1,
                      trainer_cmd=[sys.executable, str(stub)])
    updates = list(train_generator(prj, cfg))
    assert "FAILED" in updates[-1].upper()
    conn = manifest.connect(tmp_path)
    assert list_runs(conn)[0]["status"] == "failed"
