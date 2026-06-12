"""Creator Wizard tests: analysis, type detection heuristics, recipe
injection, and the build-only wizard chain on a seeded manifest."""

import json
import subprocess
import sys
from pathlib import Path

import pytest  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_studio import manifest  # noqa: E402
from lora_studio.config import Project  # noqa: E402
from lora_studio.wizard import (  # noqa: E402
    TYPE_TEMPLATES, WizardConfig, analyze, detect_type, make_recipe,
    wizard_generator,
)


def _seed(conn, n=20, face_rate=1.0, id_sims=0.7, framing="portrait",
          tags='[["smile",0.9]]', cluster_spread=True):
    for i in range(n):
        fid = f"f{i:03d}"
        conn.execute(
            "INSERT INTO frames(frame_id, source_id, path, sharpness, status, cluster_id) "
            "VALUES (?,?,?,?, 'selected', ?)",
            (fid, f"s{i % 3}", f"/x/{fid}.jpg", 100.0,
             (i % 5) if cluster_spread else 0),
        )
        has_face = i < int(n * face_rate)
        conn.execute(
            "INSERT INTO detections(frame_id, face_count, face_area, identity_sim, framing) "
            "VALUES (?,?,?,?,?)",
            (fid, 1 if has_face else 0, 0.1 if has_face else 0,
             id_sims if has_face else None, framing if has_face else "none"),
        )
        conn.execute(
            "INSERT INTO captions(frame_id, tags_json, caption_text) VALUES (?,?,?)",
            (fid, tags, "tok person, smile"),
        )
    conn.commit()


def test_analyze_and_character_detection(tmp_path):
    conn = manifest.connect(tmp_path)
    _seed(conn, face_rate=0.95, id_sims=0.72, framing="portrait")
    a = analyze(conn)
    assert a["selected"] == 20 and a["face_rate"] >= 0.9
    assert a["explicit_rate"] == 0.0
    ranking = detect_type(a)
    assert ranking[0]["type"] == "character"
    assert all(0 <= r["score"] <= 1 for r in ranking)


def test_style_detection_low_faces(tmp_path):
    conn = manifest.connect(tmp_path)
    _seed(conn, face_rate=0.1, framing="none")
    ranking = detect_type(analyze(conn))
    assert ranking[0]["type"] == "style"


def test_explicit_detection(tmp_path):
    conn = manifest.connect(tmp_path)
    _seed(conn, face_rate=0.9, tags='[["nude",0.95],["smile",0.8]]')
    a = analyze(conn)
    assert a["explicit_rate"] == 1.0
    ranking = detect_type(a)
    assert ranking[0]["type"] == "explicit"


def test_detail_detection(tmp_path):
    conn = manifest.connect(tmp_path)
    _seed(conn, face_rate=1.0, framing="closeup")
    conn.execute("UPDATE frames SET sharpness=200")
    conn.commit()
    types = {r["type"]: r["score"] for r in detect_type(analyze(conn))}
    assert types["detail"] > types["style"]


def test_make_recipe_and_templates():
    assert set(TYPE_TEMPLATES) == {"character", "style", "outfit", "pose",
                                   "detail", "explicit"}
    r = make_recipe("detail", "tok", "person")
    assert r["token"] == "tok" and r["smart_crop"] is True
    assert "preset" not in r          # preset is training-side, not recipe-side
    assert TYPE_TEMPLATES["explicit"]["preset"] == "character"


def test_wizard_build_only_chain(tmp_path):
    """Wizard with --no-train: analysis -> auto type -> recipe injection ->
    real versioned build from seeded frames with real files."""
    conn = manifest.connect(tmp_path)
    img_dir = tmp_path / "frames"
    img_dir.mkdir()
    for i in range(8):
        p = img_dir / f"f{i}.jpg"
        subprocess.run(
            ["ffmpeg", "-v", "quiet", "-y", "-f", "lavfi",
             "-i", "color=c=gray:size=64x64:duration=0.1:rate=10",
             "-frames:v", "1", str(p)], check=True, capture_output=True)
        fid = f"f{i:03d}"
        conn.execute(
            "INSERT INTO frames(frame_id, source_id, path, sharpness, status) "
            "VALUES (?,?,?,?, 'selected')", (fid, "s0", str(p), 100.0))
        conn.execute(
            "INSERT INTO detections(frame_id, face_count, face_area, identity_sim, framing) "
            "VALUES (?,1,0.1,0.7,'portrait')", (fid,))
        conn.execute(
            "INSERT INTO captions(frame_id, tags_json, caption_text) "
            "VALUES (?,'[[\"smile\",0.9]]','tok person')", (fid,))
    conn.commit()

    prj = Project(trigger_token="tok", class_word="person",
                  output_base=str(tmp_path))
    out = list(wizard_generator(prj, WizardConfig(
        output_base=tmp_path, lora_type="auto", train=False)))
    text = "\n".join(out)
    assert "auto-selected: character" in text
    assert "WIZARD DONE (build-only)" in text
    assert "wizard_character" in prj.recipes
    row = conn.execute(
        "SELECT version, recipe_name FROM datasets ORDER BY version DESC LIMIT 1"
    ).fetchone()
    assert row and row[1] == "wizard_character"
    snap = json.loads(
        (Path(tmp_path / "DATASET").glob("v*__wizard_character").__next__()
         / "dataset.json").read_text())
    assert snap["counts"]["train"] + snap["counts"]["val"] == 8


def test_character_detection_with_exact_zero_id_std():
    """Regression (py3.12/Colab): id_std == 0.0 is PERFECT consistency,
    not missing data - must classify character, never style."""
    from lora_studio.wizard import detect_type
    a = {"face_rate": 0.95, "id_std": 0.0, "cluster_entropy": 1.0,
         "framing_mix": {"portrait": 0.95, "none": 0.05},
         "explicit_rate": 0.0, "sharpness_mean": 100}
    ranking = detect_type(a)
    assert ranking[0]["type"] == "character"
    # low-face datasets still classify as style (guard must not leak)
    b = {"face_rate": 0.1, "id_std": None, "cluster_entropy": 1.0,
         "framing_mix": {"none": 0.9, "portrait": 0.1},
         "explicit_rate": 0.0, "sharpness_mean": 100}
    assert detect_type(b)[0]["type"] == "style"
