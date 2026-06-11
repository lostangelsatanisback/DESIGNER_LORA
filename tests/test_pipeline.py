"""End-to-end and unit tests. Requires ffmpeg; run:  pytest tests/ -v"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_studio import manifest  # noqa: E402
from lora_studio.config import CurateConfig, ExtractConfig, PackageConfig  # noqa: E402
from lora_studio.curate.basic import dhash_bits, hamming, laplacian_variance  # noqa: E402
from lora_studio.extract import pipeline_generator  # noqa: E402
from lora_studio.packager import package_generator  # noqa: E402
from lora_studio.curate import curate_generator  # noqa: E402
from lora_studio.util import safe_slug, stable_id  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def test_safe_slug():
    assert safe_slug("IMG 5662 (1).mov") == "IMG_5662_1_.mov"
    assert safe_slug("") == "unnamed"
    assert "/" not in safe_slug("a/b\\c")


def test_dhash_hamming():
    flat = [128] * 72          # left > right never true -> all 0 bits
    desc = [(8 - c) * 30 for _ in range(8) for c in range(9)]  # left > right always -> all 1 bits
    assert hamming(dhash_bits(flat), dhash_bits(flat)) == 0
    assert hamming(dhash_bits(flat), dhash_bits(desc)) == 64


def test_laplacian_flat_is_zero():
    assert laplacian_variance([100] * (16 * 16), 16, 16) == 0.0


def test_stable_id_distinct(tmp_path):
    a = tmp_path / "x.mp4"
    b = tmp_path / "x.mov"
    a.write_bytes(b"a" * 10)
    b.write_bytes(b"b" * 20)
    assert stable_id(a) != stable_id(b)


def test_manifest_migration(tmp_path):
    conn = manifest.connect(tmp_path)
    version = conn.execute("PRAGMA user_version;").fetchone()[0]
    assert version == manifest.SCHEMA_VERSION
    # tables exist
    for table in ("sources", "frames", "events", "detections", "captions", "meta"):
        conn.execute(f"SELECT COUNT(*) FROM {table}")
    manifest.meta_set(conn, "k", {"a": 1})
    assert manifest.meta_get(conn, "k") == {"a": 1}


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg required")
def test_end_to_end(tmp_path):
    videos = tmp_path / "videos"
    out = tmp_path / "out"
    videos.mkdir()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=duration=40:size=320x240:rate=24",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(videos / "clip.mp4")],
        check=True,
    )
    # dry run must not mark completed
    cfg = ExtractConfig(output_base=out, fps=1, segment_seconds=30,
                        import_photos=False, dry_run=True)
    list(pipeline_generator([str(videos)], "", cfg))
    conn = manifest.connect(out)
    assert conn.execute("SELECT status FROM sources").fetchone()[0] == "dry_run"

    # real run extracts
    cfg.dry_run = False
    list(pipeline_generator([str(videos)], "", cfg))
    n_frames = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    assert n_frames >= 35

    # curate selects fewer than extracted (dedup active)
    list(curate_generator(CurateConfig(output_base=out, min_sharpness=0)))
    sel = conn.execute(
        "SELECT COUNT(*) FROM frames WHERE status='selected'"
    ).fetchone()[0]
    assert 0 < sel <= n_frames

    # package produces images + captions
    list(package_generator(PackageConfig(output_base=out, token="tok", class_word="person")))
    ds = out / "DATASET" / "img" / "10_tok person"
    jpgs = list(ds.glob("*.jpg"))
    txts = list(ds.glob("*.txt"))
    assert jpgs and len(jpgs) == len(txts)
    assert txts[0].read_text() == "tok person"
