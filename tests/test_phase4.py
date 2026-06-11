"""Phase 4 tests: recipe parsing, deterministic builds, diff accuracy."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_studio import manifest  # noqa: E402
from lora_studio.builder import (  # noqa: E402
    BuildConfig, build_generator, diff_datasets, list_datasets, stable_split,
)
from lora_studio.config import Project, _parse_flat_toml  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def test_flat_toml_sections():
    text = """
name = "X"
ui_port = 7861
[recipes.char_v1]
repeats = 10
quota = "closeup=0.3"
[recipes.combo]
concepts = "char_v1"
val_fraction = 0.1
"""
    d = _parse_flat_toml(text)
    assert d["name"] == "X"
    assert d["recipes"]["char_v1"]["repeats"] == 10
    assert d["recipes"]["combo"]["val_fraction"] == 0.1


def test_stable_split_deterministic_and_proportional():
    ids = [f"frame{i:05d}" for i in range(2000)]
    s1 = [stable_split(i, 0.10) for i in ids]
    s2 = [stable_split(i, 0.10) for i in ids]
    assert s1 == s2
    frac = s1.count("val") / len(s1)
    assert 0.06 < frac < 0.14
    assert all(stable_split(i, 0.0) == "train" for i in ids[:50])


def _seed_manifest(out: Path, n_frames: int = 12) -> None:
    """Create a manifest with real (tiny) image files marked selected."""
    img_dir = out / "frames" / "src"
    img_dir.mkdir(parents=True, exist_ok=True)
    conn = manifest.connect(out)
    for i in range(n_frames):
        p = img_dir / f"f{i:03d}.jpg"
        subprocess.run(
            ["ffmpeg", "-v", "quiet", "-y", "-f", "lavfi",
             "-i", f"color=c=gray:size=64x64:duration=0.1:rate=10",
             "-frames:v", "1", str(p)],
            check=True, capture_output=True,
        )
        fid = f"fid{i:03d}"
        conn.execute(
            "INSERT INTO frames(frame_id, source_id, path, sharpness, status, cluster_id) "
            "VALUES (?,?,?,?, 'selected', ?)",
            (fid, f"src{i % 3}", str(p), 100 - i, i % 4),
        )
        conn.execute(
            "INSERT INTO captions(frame_id, caption_text, edited) VALUES (?,?,0)",
            (fid, f"tok person, tag{i}"),
        )
    conn.commit()
    conn.close()


def _project() -> Project:
    return Project(
        trigger_token="tok", class_word="person",
        recipes={
            "char_v1": {"repeats": 10, "max_total": 8, "val_fraction": 0.25},
            "mini": {"repeats": 5, "max_total": 4, "token": "mini", "class_word": "thing"},
            "combo": {"concepts": "char_v1,mini"},
        },
    )


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg required")
def test_build_reproducible_hash(tmp_path):
    _seed_manifest(tmp_path)
    prj = _project()
    out1 = list(build_generator(prj, BuildConfig(output_base=tmp_path, recipe="char_v1")))
    conn = manifest.connect(tmp_path)
    builds = list_datasets(conn)
    assert len(builds) == 1
    h1 = builds[0]["hash"]
    # second build from identical state -> recognized as identical, no v2
    out2 = list(build_generator(prj, BuildConfig(output_base=tmp_path, recipe="char_v1")))
    assert "Identical dataset already exists" in out2[-1]
    assert len(list_datasets(conn)) == 1
    # dataset.json exists and matches
    ds_dir = Path(builds[0]["dir"])
    snap = json.loads((ds_dir / "dataset.json").read_text())
    assert snap["content_hash"] == h1
    assert snap["counts"]["train"] + snap["counts"]["val"] == 8
    # images + captions on disk
    imgs = list(ds_dir.rglob("*.jpg"))
    txts = list(ds_dir.rglob("*.txt"))
    assert len(imgs) == 8 and len(txts) == 8


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg required")
def test_multiconcept_and_diff(tmp_path):
    _seed_manifest(tmp_path)
    prj = _project()
    list(build_generator(prj, BuildConfig(output_base=tmp_path, recipe="char_v1")))
    conn = manifest.connect(tmp_path)
    # change state: edit one caption -> rebuild same recipe = new version
    conn.execute(
        "UPDATE captions SET caption_text='tok person, EDITED', edited=1 "
        "WHERE frame_id='fid000'"
    )
    conn.commit()
    list(build_generator(prj, BuildConfig(output_base=tmp_path, recipe="char_v1")))
    builds = list_datasets(conn)
    assert len(builds) == 2 and builds[0]["hash"] != builds[1]["hash"]
    d = diff_datasets(conn, 1, 2)
    assert len(d["caption_changed"]) == 1
    assert d["caption_changed"][0][0] == "fid000"
    assert not d["added"] and not d["removed"]
    # multi-concept build creates both folders
    list(build_generator(prj, BuildConfig(output_base=tmp_path, recipe="combo")))
    builds = list_datasets(conn)
    combo_dir = Path(builds[-1]["dir"])
    folders = {p.name for p in (combo_dir / "img").iterdir()}
    assert "10_tok person" in folders and "5_mini thing" in folders
