"""Preset Recommendation System tests - fixture manifests, no real images."""
import json
from pathlib import Path

import pytest

from lora_studio import manifest
from lora_studio.train.preset_recommender import (available_presets,
                                                  collect_signals,
                                                  format_recommendation,
                                                  suggest_preset)
from lora_studio.train.presets import PRESETS


def _seed(tmp_path: Path, n: int, framing: str = "full_body",
          study: str = "figure_study_candidate", id_sim: float = 0.85,
          sharp: float = 110.0, captions: bool = True,
          study_rows: bool = True, recipe_json: str = "") -> object:
    """Build a manifest with one dataset version (v1) of n frames."""
    conn = manifest.connect(tmp_path)
    conn.execute(
        "INSERT INTO datasets (version, recipe_name, recipe_json, "
        "image_count, val_count, dir, built_at) VALUES (1,?,?,?,0,'d','t')",
        ("fixture_recipe", recipe_json, n))
    for i in range(n):
        fid = f"f{i:05d}"
        conn.execute(
            "INSERT INTO frames (frame_id, path, status, sharpness, "
            "cluster_id) VALUES (?,?,?,?,?)",
            (fid, f"/x/{fid}.jpg", "selected", sharp, i % 12))
        conn.execute(
            "INSERT INTO dataset_frames (version, frame_id, concept, split) "
            "VALUES (1,?,?,'train')", (fid, "10_tok person"))
        conn.execute(
            "INSERT INTO detections (frame_id, face_count, identity_sim, "
            "framing) VALUES (?,?,?,?)",
            (fid, 1, id_sim + (0.01 if i % 2 else -0.01),
             framing if i % 3 else "closeup"))
        if captions:
            conn.execute(
                "INSERT INTO captions (frame_id, caption_text) VALUES (?,?)",
                (fid, "tok person, studio lighting"))
        if study_rows:
            conn.execute(
                "INSERT INTO study_labels (frame_id, study_primary, "
                "study_confidence, identity_lock_score) VALUES (?,?,?,?)",
                (fid, study, 0.7, 0.8))
    conn.commit()
    return conn


def test_strong_figure_dataset_recommends_figure_preset(tmp_path):
    conn = _seed(tmp_path, 400, framing="full_body",
                 study="figure_study_candidate")
    rec = suggest_preset(conn, 1)
    assert rec["recommended_preset"] in (
        "fine_art_figure_study", "figure_study", "intimate_anatomy_study")
    assert rec["confidence"] >= 0.6
    assert "figure_study_rate_high" in rec["reason_codes"]
    assert "strong_identity_signal" in rec["reason_codes"]
    assert rec["signals"]["selected_frames"] == 400
    assert rec["alternative_presets"]


def test_fashion_dataset_recommends_garment_preset(tmp_path):
    conn = _seed(tmp_path, 350, framing="upper_body",
                 study="lingerie_fashion_candidate")
    rec = suggest_preset(conn, 1)
    assert rec["recommended_preset"] in ("lingerie_form_study",
                                         "fashion_editorial")
    assert "fashion_study_rate_high" in rec["reason_codes"]


def test_small_dataset_gated_to_safe_preset(tmp_path):
    conn = _seed(tmp_path, 60, framing="full_body",
                 study="figure_study_candidate")
    rec = suggest_preset(conn, 1)
    assert rec["recommended_preset"] in ("balanced_study", "intimate_figure")
    assert "dataset_size_small" in rec["reason_codes"]
    assert "specialized_presets_gated" in rec["reason_codes"]
    assert any("specialized presets are gated" in w.lower() or
               "balanced preset" in w.lower() for w in rec["warnings"])


def test_weak_identity_gated(tmp_path):
    conn = _seed(tmp_path, 400, id_sim=0.30)
    rec = suggest_preset(conn, 1)
    assert rec["recommended_preset"] in ("balanced_study", "intimate_figure")
    assert "identity_consistency_low" in rec["reason_codes"]


def test_missing_study_and_caption_data_falls_back(tmp_path):
    conn = _seed(tmp_path, 320, study_rows=False, captions=False)
    rec = suggest_preset(conn, 1)
    assert rec["recommended_preset"] in PRESETS
    assert "study_coverage_incomplete" in rec["reason_codes"]
    assert "caption_coverage_low" in rec["reason_codes"]
    assert rec["signals"]["study_coverage"] == 0.0


def test_unknown_dataset_version_raises(tmp_path):
    conn = manifest.connect(tmp_path)
    with pytest.raises(ValueError):
        suggest_preset(conn, 99)


def test_dynamic_preset_availability_fallback(tmp_path):
    """Only presets that exist may be recommended; targets fall back
    along the preference chain."""
    conn = _seed(tmp_path, 400, framing="full_body",
                 study="figure_study_candidate")
    limited = {"balanced_study": PRESETS["balanced_study"],
               "character": PRESETS["character"]}
    rec = suggest_preset(conn, 1, presets=limited)
    assert rec["recommended_preset"] in limited
    for a in rec["alternative_presets"]:
        assert a["preset"] in limited
    empty = suggest_preset(conn, 1, presets={})
    assert empty["recommended_preset"] is None
    assert "no_presets" in empty["reason_codes"]


def test_recipe_study_context_nudges(tmp_path):
    rj = json.dumps({"type": "character",
                     "study_primary": "figure_study_candidate"})
    conn = _seed(tmp_path, 400, framing="full_body",
                 study="figure_study_candidate", recipe_json=rj)
    rec = suggest_preset(conn, 1)
    assert "recipe_study_context_match" in rec["reason_codes"]
    assert rec["signals"]["recipe_study_primary"] == "figure_study_candidate"
    # explicit CLI recipe override is recorded
    rec2 = suggest_preset(conn, 1, recipe_name="mdma_intimate_v1")
    assert rec2["signals"]["recipe_override"] == "mdma_intimate_v1"


def test_collect_signals_shapes(tmp_path):
    conn = _seed(tmp_path, 100)
    s = collect_signals(conn, 1)
    for key in ("selected_frames", "identity_strength", "framing_mix",
                "figure_study_rate", "fashion_study_rate", "quality_score",
                "cluster_count", "caption_coverage", "study_coverage",
                "recipe_name"):
        assert key in s
    assert 0.0 <= s["quality_score"] <= 1.0
    assert s["cluster_count"] == 12


def test_format_recommendation_professional(tmp_path):
    conn = _seed(tmp_path, 400)
    text = format_recommendation(suggest_preset(conn, 1))
    assert "Recommended preset:" in text
    assert "Confidence:" in text
    assert "Signals:" in text


def test_available_presets_lists_real_presets():
    names = available_presets()
    assert "balanced_study" in names and "character" in names
