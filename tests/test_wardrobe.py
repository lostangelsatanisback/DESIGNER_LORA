"""Wardrobe Variation & Selective Region Editing."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora_studio import manifest  # noqa: E402
from lora_studio.config import Project  # noqa: E402
from lora_studio.wardrobe import (EDIT_MODES, REGION_PRESETS,  # noqa: E402
                                  WardrobeEditRequest, analyze_edit_readiness,
                                  build_wardrobe_edit_payload,
                                  get_region_model_requirements,
                                  list_region_presets, record_wardrobe_edit)

CR = "/x/cyberrealisticPony_v18CoreShift.safetensors"


def _prj(tmp_path=None):
    prj = Project()
    prj.trigger_token, prj.class_word = "spookums", "person"
    prj.base_model = CR
    if tmp_path is not None:
        prj.forge_root = str(tmp_path)
    return prj


def test_region_presets_complete():
    presets = list_region_presets()
    ids = {p["region_id"] for p in presets}
    assert ids == {"upper_body_torso", "lower_body_bottomwear",
                   "full_body_wardrobe", "arms_hands",
                   "background_environment"}
    for p in presets:
        assert p["label"] and p["description"] and p["notes"]
        lo, hi = p["recommended_denoise"]
        assert 0 < lo < hi <= 0.9
        assert p["identity_priority"] in ("high", "medium")
        assert p["recommended_controlnets"]
    assert REGION_PRESETS["full_body_wardrobe"].recommended_controlnets[0] \
        == "openpose"
    assert len(EDIT_MODES) == 5


def test_model_requirements_detection_and_degradation(tmp_path):
    # nothing configured -> never raises, clear statuses
    reqs = get_region_model_requirements(Project(), "upper_body_torso")
    assert all(r["detected"] in ("not configured", "optional", "missing")
               for r in reqs)
    # forge layout with some components present
    (tmp_path / "models/Stable-diffusion").mkdir(parents=True)
    (tmp_path / "models/Stable-diffusion/pony_inpaint_v1.safetensors"
     ).write_bytes(b"x")
    (tmp_path / "models/ControlNet").mkdir(parents=True)
    (tmp_path / "models/ControlNet/control_openpose_sdxl.safetensors"
     ).write_bytes(b"x")
    reqs = get_region_model_requirements(_prj(tmp_path),
                                         "full_body_wardrobe")
    by_cat = {r["category"]: r for r in reqs}
    assert by_cat["inpainting_checkpoint"]["detected"] == "found"
    assert by_cat["controlnet_openpose"]["detected"] == "found"
    assert by_cat["controlnet_depth"]["detected"] == "missing"
    assert "identity preservation" in by_cat["identity_guidance"]["guidance"]
    assert by_cat["identity_postprocess"]["detected"] == "optional"


def test_payload_builder_identity_preserving(tmp_path):
    img = tmp_path / "char.png"
    img.write_bytes(b"png-bytes")
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"mask-bytes")
    req = WardrobeEditRequest(
        image_path=str(img), mask_path=str(mask),
        region_id="upper_body_torso", edit_mode="garment_replacement",
        garment_direction_prompt="tailored charcoal blazer, satin blouse",
        selected_loras=[["spook_character_v1", 0.75], ["fashion_pack", 0.4]],
        seed=99)
    p = build_wardrobe_edit_payload(_prj(), req)
    assert "<lora:spook_character_v1:0.75>" in p["prompt"]
    assert "consistent character identity" in p["prompt"]
    assert "tailored charcoal blazer" in p["prompt"]
    assert "identity drift" in p["negative_prompt"]
    assert p["denoising_strength"] == 0.55          # region midpoint
    assert p["mask"] and p["init_images"][0]
    assert p["mask_blur"] == 8 and p["inpaint_full_res_padding"] == 24
    assert p["override_settings"]["CLIP_stop_at_last_layers"] == 2
    assert p["sampler_name"] == "DPM++ SDE Karras"
    cns = p["alwayson_scripts"]["controlnet"]["args"]
    assert {u["module"] for u in cns} == {"softedge", "depth"}
    # pose unit included for full-body, dropped when pose is unlocked
    req2 = WardrobeEditRequest(image_path=str(img),
                               region_id="full_body_wardrobe",
                               preserve_pose=False)
    p2 = build_wardrobe_edit_payload(_prj(), req2)
    mods = {u["module"] for u in
            p2["alwayson_scripts"]["controlnet"]["args"]}
    assert "openpose_full" not in mods
    # unknown edit mode rejected
    try:
        build_wardrobe_edit_payload(_prj(), WardrobeEditRequest(
            image_path=str(img), edit_mode="nope"))
        assert False
    except ValueError:
        pass


def test_readiness_identity_policy(tmp_path):
    from tests.test_concept import _card
    cards = [_card("idl_character", "identity"),
             _card("sty_a", "style", "high"), _card("sty_b", "style", "high")]
    req = WardrobeEditRequest(
        image_path="/x/c.png", region_id="full_body_wardrobe",
        selected_loras=[["idl_character", 0.75], ["sty_a", 0.35],
                        ["sty_b", 0.35]],
        denoise=0.8, preserve_pose=False)
    r = analyze_edit_readiness(_prj(), req, cards)
    assert 0 < r["identity_preservation_score"] < 1
    sugg = " ".join(r["auto_adjustment_suggestions"])
    assert "denoise" in sugg.lower()
    assert "pose guidance" in sugg
    # no identity anchor -> high risk + clear suggestion
    r2 = analyze_edit_readiness(_prj(), WardrobeEditRequest(
        image_path="/x/c.png", selected_loras=[["sty_a", 0.35]]),
        [c for c in cards if c.lora_id != "idl_character"])
    assert r2["identity_risk_level"] == "high"
    assert any("identity anchor" in s
               for s in r2["auto_adjustment_suggestions"])
    # background note for environment region with preservation on
    r3 = analyze_edit_readiness(_prj(), WardrobeEditRequest(
        image_path="/x/c.png", region_id="background_environment"))
    assert any("Background consistency" in n
               for n in r3["consistency_notes"])


def test_manifest_v10_tracking(tmp_path):
    conn = manifest.connect(tmp_path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 10
    req = WardrobeEditRequest(image_path="/x/c.png",
                              garment_direction_prompt="soft knit cardigan")
    readiness = analyze_edit_readiness(_prj(), req)
    eid = record_wardrobe_edit(conn, _prj(), req, readiness, "")
    row = conn.execute(
        "SELECT region_id, edit_mode, denoise, risk_level, readiness "
        "FROM wardrobe_edits WHERE edit_id=?", (eid,)).fetchone()
    assert row[0] == "upper_body_torso"
    assert row[1] == "garment_replacement" and row[2] == 0.55
    assert row[3] in ("low", "medium", "high")
    snapshot = json.loads(row[4])
    assert "inpainting_checkpoint" in snapshot
    conn.close()
