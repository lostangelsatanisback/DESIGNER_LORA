"""Study Intelligence Layer - classifier, schema, recipes, planner, presets."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora_studio import manifest  # noqa: E402
from lora_studio.config import Project  # noqa: E402
from lora_studio.study import (STUDY_RECIPES, StudyConfig,  # noqa: E402
                               classify_frame, classify_generator,
                               register_study_recipes, set_manual_label,
                               study_report, suggest_study_stack,
                               write_study_presets)


def _seed_manifest(conn, n=6):
    """Selected frames with varied framing/identity/caption signals."""
    rows = [
        # fid, framing, ident, faces, sharp, bright, caption
        ("f1", "full_body", 0.62, 1, 120, 110,
         "1girl, standing, dress, looking at viewer"),
        ("f2", "upper_body", 0.55, 1, 90, 120,
         "1girl, shirt, jeans, smile"),
        ("f3", "full_body", 0.58, 1, 100, 100,
         "1girl, lingerie, lace, standing"),
        ("f4", "closeup", 0.50, 1, 140, 90, "1girl, face, freckles"),
        ("f5", "full_body", None, 0, 30, 20, "silhouette, dark"),
        ("f6", "portrait", 0.60, 1, 80, 130,
         "1girl, nude, standing"),     # -> review gate
    ][:n]
    for fid, framing, ident, faces, sharp, bright, cap in rows:
        conn.execute("INSERT INTO frames(frame_id, source_id, path, sharpness, "
                     "brightness, status) VALUES (?,?,?,?,?,'selected')",
                     (fid, "s1", f"/x/{fid}.jpg", sharp, bright))
        conn.execute("INSERT INTO detections(frame_id, face_count, identity_sim, "
                     "framing) VALUES (?,?,?,?)", (fid, faces, ident, framing))
        conn.execute("INSERT INTO captions(frame_id, caption_text) VALUES (?,?)",
                     (fid, cap))
    conn.commit()


def test_schema_v7_migrates_and_old_manifests_load(tmp_path):
    conn = manifest.connect(tmp_path)
    v = conn.execute("PRAGMA user_version").fetchone()[0]
    assert v >= 7
    # study table exists and is empty - report on a fresh manifest never crashes
    rep = study_report(conn)
    assert rep["categories"] == {} and rep["needs_review"] == 0
    conn.close()


def test_classify_frame_scores_and_reasons():
    rec = classify_frame({"sharpness": 120, "brightness": 110,
                          "framing": "full_body", "identity_sim": 0.62,
                          "face_count": 1,
                          "caption": "1girl, standing, dress", "tags_json": None})
    reasons = json.loads(rec["study_reason_codes"])
    assert "clear_full_body_framing" in reasons
    assert "face_identity_visible" in reasons
    assert "garment_structure_visible" in reasons
    assert rec["identity_lock_score"] > 0.8
    assert rec["pose_clarity_score"] > 0.8
    assert rec["study_export_eligible"] == 1
    tags = json.loads(rec["study_tags"])
    assert "full_body_frame" in tags and "wardrobe_focus_frame" in tags


def test_review_gate_routes_to_manual_review():
    rec = classify_frame({"sharpness": 100, "brightness": 110,
                          "framing": "portrait", "identity_sim": 0.6,
                          "face_count": 1, "caption": "1girl, nude, standing",
                          "tags_json": None})
    assert rec["study_review_status"] == "needs_review"
    assert rec["study_export_eligible"] == 0
    assert "requires_manual_review" in json.loads(rec["study_reason_codes"])


def test_classify_generator_resumable_and_report(tmp_path):
    conn = manifest.connect(tmp_path)
    _seed_manifest(conn)
    conn.close()
    prj = Project()
    out = list(classify_generator(prj, StudyConfig(output_base=tmp_path)))
    assert any("Classification complete" in u for u in out)
    # resumable: second run with nothing new
    out2 = list(classify_generator(prj, StudyConfig(output_base=tmp_path)))
    assert any("Nothing to classify" in u for u in out2)
    conn = manifest.connect(tmp_path)
    rep = study_report(conn)
    assert sum(rep["categories"].values()) == 6
    assert rep["needs_review"] >= 1          # the review-gated frame
    assert rep["export_eligible"] >= 1
    # lingerie/fashion candidate detected from apparel caption signal
    cat = conn.execute("SELECT study_primary FROM study_labels "
                       "WHERE frame_id='f3'").fetchone()[0]
    assert cat == "lingerie_fashion_candidate"
    conn.close()


def test_manual_override_survives_reclassification(tmp_path):
    conn = manifest.connect(tmp_path)
    _seed_manifest(conn, n=2)
    conn.close()
    prj = Project()
    list(classify_generator(prj, StudyConfig(output_base=tmp_path)))
    conn = manifest.connect(tmp_path)
    set_manual_label(conn, "f1", "fashion_study_candidate")
    conn.close()
    list(classify_generator(prj, StudyConfig(output_base=tmp_path, rescan=True)))
    conn = manifest.connect(tmp_path)
    row = conn.execute("SELECT study_primary, manual_override FROM study_labels "
                       "WHERE frame_id='f1'").fetchone()
    assert row == ("fashion_study_candidate", 1)
    conn.close()


def test_study_recipes_select_expected_records(tmp_path):
    from lora_studio.builder import RECIPE_DEFAULTS, select_for_recipe
    conn = manifest.connect(tmp_path)
    _seed_manifest(conn)
    prj = Project()
    list(classify_generator(prj, StudyConfig(output_base=tmp_path)))
    conn = manifest.connect(tmp_path)
    rcp = {**RECIPE_DEFAULTS, **STUDY_RECIPES["lingerie_fashion_study_v1"],
           "quota": "", "min_identity": 0.0, "min_sharpness": 0.0}
    rows = select_for_recipe(conn, rcp)
    ids = {r[0] for r in rows}
    assert ids == {"f3"}            # only the apparel-signal frame
    # a recipe with no study filter still selects normally (backward compat)
    base = {**RECIPE_DEFAULTS, "quota": ""}
    assert len(select_for_recipe(conn, base)) == 6
    conn.close()


def test_register_study_recipes_additive():
    prj = Project()
    prj.recipes = {"figure_study_v1": {"type": "character", "repeats": 99}}
    added = register_study_recipes(prj)
    assert "figure_study_v1" not in added            # never overwritten
    assert prj.recipes["figure_study_v1"]["repeats"] == 99
    assert "fashion_editorial_v1" in added


def test_suggest_study_stack_modes():
    rec = suggest_study_stack("character_figure_fashion_editorial",
                              "/x/cyberrealisticPony_v18.safetensors")
    assert rec["profile"].startswith("CyberRealistic")
    assert rec["stack"][0]["type"] == "character"
    assert rec["stack"][0]["weight"] == 0.85          # identity dominant
    assert all(s["weight"] < 0.85 for s in rec["stack"][1:])
    assert rec["merge_order"][-1] == "character"      # identity merged last
    assert any("Conflict risk" in w for w in rec["warnings"])
    try:
        suggest_study_stack("nope")
        assert False
    except KeyError:
        pass


def test_training_presets_present():
    from lora_studio.train.presets import PRESETS, STUDY_VALIDATION_PROMPTS
    for name in ("figure_study", "fashion_editorial", "balanced_study"):
        p = PRESETS[name]
        assert 8 <= p["network_dim"] <= 64 and p["unet_lr"] > 0
        assert "notes" in p
    assert len(STUDY_VALIDATION_PROMPTS) == 5


def test_write_study_presets_render(tmp_path):
    prj = Project()
    prj.trigger_token, prj.class_word = "gk_person", ""
    prj.base_model = "/x/cyberrealisticPony_v18CoreShift.safetensors"
    target = write_study_presets(prj, tmp_path / "pp.json")
    data = json.loads(target.read_text())
    assert len(data) == 8
    p = data["Identity-Locked Figure Study"]
    assert p["sampler"] == "DPM++ SDE Karras" and p["cfg"] == 5.0
    assert p["clip_skip"] == 2
    assert "artistic figure study" in p["prompt"]
    assert "consistent character identity" in p["prompt"]
    assert "gk_person" in p["prompt"]
    assert p["_study"]["qa_notes"]
    qa = data["Merge QA - Identity Preservation"]
    assert "neutral portrait" in qa["prompt"]
    # merging is non-destructive: presets file merges, never truncates
    write_study_presets(prj, tmp_path / "pp.json")
    assert len(json.loads(target.read_text())) == 8
