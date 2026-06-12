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
    # guidance models present so policy units can activate
    (tmp_path / "models/ControlNet").mkdir(parents=True)
    for f in ("control_softedge.safetensors", "control_depth.safetensors",
              "control_openpose.safetensors"):
        (tmp_path / "models/ControlNet" / f).write_bytes(b"x")
    req = WardrobeEditRequest(
        image_path=str(img), mask_path=str(mask),
        region_id="upper_body_torso", edit_mode="garment_replacement",
        garment_direction_prompt="tailored charcoal blazer, satin blouse",
        selected_loras=[["spook_character_v1", 0.75], ["fashion_pack", 0.4]],
        seed=99, faceid_preset="off", preserve_pose=False,
        body_structure_lock=True, silhouette_guidance=True)
    p = build_wardrobe_edit_payload(_prj(tmp_path), req)
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
                               preserve_pose=False, faceid_preset="off")
    p2 = build_wardrobe_edit_payload(_prj(tmp_path), req2)
    mods = {u["module"] for u in
            p2["alwayson_scripts"]["controlnet"]["args"]}
    assert "openpose_full" not in mods and "depth" in mods
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


# ---------------------------------------------------------------------------
# v3.10: Identity Integration Layer
# ---------------------------------------------------------------------------

def test_identity_config_defaults_and_presets():
    from lora_studio.identity_integration import (FACEID_PRESETS,
                                                  IdentityIntegrationConfig)
    cfg = IdentityIntegrationConfig()
    assert cfg.enabled and cfg.faceid_enabled
    assert not cfg.strong_face_lock and cfg.faceid_strength == 0.75
    cfg2 = IdentityIntegrationConfig.from_payload(
        {"faceid_preset": "maximum", "strong_face_lock": True,
         "unknown_field": 1})
    assert cfg2.faceid_strength == FACEID_PRESETS["maximum"]
    assert cfg2.strong_face_lock
    off = IdentityIntegrationConfig.from_payload({"faceid_preset": "off"})
    assert not off.faceid_enabled


def test_controlnet_policy_region_aware():
    from lora_studio.identity_integration import (IdentityIntegrationConfig,
                                                  controlnet_policy)
    full = controlnet_policy(IdentityIntegrationConfig(
        region_preset="full_body_wardrobe", pose_consistency=True))
    assert "openpose" in full and "depth" in full
    torso = controlnet_policy(IdentityIntegrationConfig(
        region_preset="upper_body_torso"))
    assert torso == []                       # conservative: nothing by default
    torso2 = controlnet_policy(IdentityIntegrationConfig(
        region_preset="upper_body_torso", silhouette_guidance=True))
    assert torso2 == ["softedge"]
    bg = controlnet_policy(IdentityIntegrationConfig(
        region_preset="background_environment",
        background_consistency=True, pose_consistency=True))
    assert bg == ["depth"]                   # identity units never on bg


def test_augment_payload_with_tools_and_degradation(tmp_path):
    from lora_studio.identity_integration import (IdentityIntegrationConfig,
                                                  augment_payload)
    # full toolset present
    for d, f in (("models/ipadapter", "ip-adapter-faceid-plusv2.bin"),
                 ("models/insightface", "inswapper_128.onnx"),
                 ("models/ControlNet", "control_openpose.safetensors"),
                 ("models/ControlNet", "control_depth.safetensors")):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
        (tmp_path / d / f).write_bytes(b"x")
    prj = _prj(tmp_path)
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"png")
    cfg = IdentityIntegrationConfig(
        strong_face_lock=True, pose_consistency=True,
        body_structure_lock=True, identity_lora_weight=0.85,
        reference_image_path=str(ref),
        region_preset="full_body_wardrobe")
    payload = {"denoising_strength": 0.6, "init_images": ["abc"]}
    res = augment_payload(prj, payload, cfg)
    assert "identity_lora" in res.active_tools
    assert "ip_adapter_faceid" in res.active_tools
    assert "controlnet_openpose" in res.active_tools
    units = payload["alwayson_scripts"]["controlnet"]["args"]
    fid = [u for u in units if "face_id" in u["module"]][0]
    assert fid["weight"] == 0.75 and fid["image"]
    frag = res.manifest_fragment["identity_integration"]
    assert frag["identity_preservation_score"] == res.identity_score
    assert frag["active_tools"] == res.active_tools
    assert res.identity_score >= 0.9
    # nothing configured -> graceful degradation, generation not blocked
    res2 = augment_payload(Project(), {"denoising_strength": 0.8,
                                       "init_images": ["abc"]},
                           IdentityIntegrationConfig(
                               strong_face_lock=True,
                               region_preset="upper_body_torso"))
    assert "ip_adapter_faceid" in res2.degraded_features
    assert "inswapper_postprocess" in res2.degraded_features
    assert any("denoise" in r.lower() for r in res2.recommendations)
    assert res2.payload is not None          # payload survives untouched
    # disabled layer is a no-op
    res3 = augment_payload(Project(), {}, IdentityIntegrationConfig(
        enabled=False))
    assert res3.manifest_fragment["identity_integration"]["enabled"] is False


def test_wardrobe_payload_carries_identity_integration(tmp_path):
    img = tmp_path / "char.png"
    img.write_bytes(b"png")
    (tmp_path / "models/ipadapter").mkdir(parents=True)
    (tmp_path / "models/ipadapter/ip-adapter-faceid-plusv2.bin"
     ).write_bytes(b"x")
    prj = _prj(tmp_path)
    req = WardrobeEditRequest(
        image_path=str(img), region_id="upper_body_torso",
        strong_face_lock=True, faceid_preset="strong",
        selected_loras=[["spook_character_v1", 0.8]])
    p = build_wardrobe_edit_payload(prj, req)
    frag = p["_identity_integration"]
    assert frag["enabled"] and frag["strong_face_lock"]
    assert frag["faceid_strength"] == 0.75
    assert frag["identity_lora_weight"] == 0.8     # auto-detected anchor
    assert "ip_adapter_faceid" in frag["active_tools"]
    # post-process degraded (no inswapper) but generation proceeds
    assert "inswapper_postprocess" in frag["degraded_features"]
    # readiness surfaces identity tools
    r = analyze_edit_readiness(prj, req)
    assert "identity_tools" in r
    assert any("Identity tools active" in n for n in r["consistency_notes"])


# ---------------------------------------------------------------------------
# v3.11: system-gap closures
# ---------------------------------------------------------------------------

def test_controlnet_name_resolution(tmp_path):
    from lora_studio.wardrobe import resolve_controlnet_model
    assert resolve_controlnet_model(Project(), "depth") == "auto:depth"
    (tmp_path / "models/ControlNet").mkdir(parents=True)
    (tmp_path / "models/ControlNet/controlnet-sdxl-depth-xinsir.safetensors"
     ).write_bytes(b"x")
    assert resolve_controlnet_model(_prj(tmp_path), "depth") == \
        "controlnet-sdxl-depth-xinsir"


def test_automatic_region_mask(tmp_path):
    from lora_studio.wardrobe import generate_region_mask
    try:
        from PIL import Image
    except ImportError:
        return
    img = tmp_path / "char.png"
    Image.new("RGB", (200, 400), "gray").save(img)
    for region in ("upper_body_torso", "background_environment"):
        mp = generate_region_mask(img, region, tmp_path / f"{region}.png")
        assert mp and Path(mp).exists()
        m = Image.open(mp).convert("L")
        px = list(m.getdata())
        assert max(px) > 200 and min(px) < 50      # editable + protected zones
    # torso mask keeps the face band (top) dark, background mask inverts it
    torso = Image.open(tmp_path / "upper_body_torso.png").convert("L")
    bg = Image.open(tmp_path / "background_environment.png").convert("L")
    assert torso.getpixel((100, 8)) < 60           # face area protected
    assert bg.getpixel((3, 200)) > 200             # edges editable
    assert bg.getpixel((100, 200)) < 60            # subject protected
    # tolerant: bogus input -> None, never raises
    assert generate_region_mask(tmp_path / "missing.png",
                                "upper_body_torso") is None


def test_preset_diff():
    from lora_studio.stack_workflow import diff_stack_presets
    a = {"sel": {"idl_character": 0.75, "fash": 0.4}, "name": "a",
         "preservation_score": 0.9}
    b = {"sel": {"idl_character": 0.8, "lit": 0.3}, "name": "b"}
    d = diff_stack_presets(a, b)
    row = next(r for r in d["rows"] if r["lora_id"] == "idl_character")
    assert row["delta"] == 0.05
    assert d["only_in_a"] == ["fash"] and d["only_in_b"] == ["lit"]
    assert d["score_a"] == 0.9


def test_insights_health_lint_clusters(tmp_path):
    from lora_studio import manifest
    from lora_studio.insights import (caption_lint, dataset_health,
                                      name_clusters)
    conn = manifest.connect(tmp_path)
    assert dataset_health(conn)["score"] == 0          # empty manifest
    for i, (framing, faces, cap) in enumerate([
            ("portrait", 1, "spookums person, smile, solo"),
            ("portrait", 2, "spookums person, solo, jewelry"),   # lint hit
            ("upper_body", 1, "spookums person, dress")]):
        conn.execute("INSERT INTO frames(frame_id, source_id, path, status, "
                     "cluster_id) VALUES (?,?,?,'selected',?)",
                     (f"f{i}", "s1", f"/x/f{i}.jpg", i % 2))
        conn.execute("INSERT INTO detections(frame_id, face_count, "
                     "identity_sim, framing) VALUES (?,?,0.6,?)",
                     (f"f{i}", faces, framing))
        conn.execute("INSERT INTO captions(frame_id, caption_text) "
                     "VALUES (?,?)", (f"f{i}", cap))
    conn.commit()
    h = dataset_health(conn)
    assert 0 < h["score"] <= 100 and h["selected"] == 3
    assert any("full-body" in r for r in h["reasons"])
    lint = caption_lint(conn)
    assert len(lint) == 1 and lint[0]["frame_id"] == "f1"
    names = name_clusters(conn)
    assert set(names) == {"0", "1"} and all(names.values())
    import json as _j
    stored = _j.loads(conn.execute(
        "SELECT value FROM meta WHERE key='cluster_names'").fetchone()[0])
    assert stored == names
    conn.close()


def test_v11_measured_sim_and_backup(tmp_path):
    from lora_studio import manifest
    from lora_studio.batch_variations import load_batch
    from lora_studio.maintenance import backup_project, restore_project
    conn = manifest.connect(tmp_path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 11
    conn.execute("INSERT INTO variation_batches(batch_id, spec, job_count, "
                 "created_at) VALUES ('b1','{}',1,'t')")
    conn.execute("INSERT INTO variation_jobs(batch_id, variation_id, seed, "
                 "loras, slider_state, warnings, output_path, "
                 "measured_face_sim, status) VALUES "
                 "('b1','v000',42,'[]','{}','[]','/x/v.png',0.61,"
                 "'generated')")
    conn.commit()
    job = load_batch(conn, "b1")["jobs"][0]
    assert job["measured_face_sim"] == 0.61
    conn.close()
    prj = _prj()
    prj.output_base = str(tmp_path)
    arc = backup_project(prj, tmp_path / "bk.tar.gz")
    assert arc.exists() and arc.stat().st_size > 0
    dest = restore_project(arc, tmp_path / "restored")
    assert (dest / "manifest").exists()


def test_qa_render_offline_and_golden_prompts(tmp_path, monkeypatch=None):
    from lora_studio.qa_render import (ab_compare, golden_prompts,
                                       render_lora_previews, side_by_side)
    prj = _prj()
    prj.trigger_token, prj.class_word = "spookums", "person"
    # offline: every generator explains instead of crashing
    for gen in (render_lora_previews(prj, "/x/l.safetensors"),
                ab_compare(prj, ("a", 0.7), ("b", 0.7))):
        msgs = list(gen)
        assert any("not reachable" in m for m in msgs)
    # golden prompt set: created once, 10 prompts, trigger present
    gp = tmp_path / "golden_prompts.json"
    prompts = golden_prompts(prj, path=gp)
    assert len(prompts) == 10 and gp.exists()
    assert all("spookums person" in p for p in prompts)
    assert all(p.startswith("score_9") for p in prompts)
    # second call loads the saved set verbatim
    assert golden_prompts(prj, path=gp) == prompts
    # side-by-side composite math
    try:
        import io
        from PIL import Image
    except ImportError:
        return
    buf_a, buf_b = io.BytesIO(), io.BytesIO()
    Image.new("RGB", (64, 80), "red").save(buf_a, "PNG")
    Image.new("RGB", (32, 100), "blue").save(buf_b, "PNG")
    out = side_by_side(buf_a.getvalue(), buf_b.getvalue(),
                       tmp_path / "ab.png")
    img = Image.open(out)
    assert img.size == (96, 100)


def test_global_search_and_stack_history(tmp_path):
    from lora_studio import manifest
    from lora_studio.concept_control import load_presets
    from lora_studio.insights import global_search, log_stack_history
    conn = manifest.connect(tmp_path)
    conn.execute("INSERT INTO frames(frame_id, source_id, path, status) "
                 "VALUES ('fx','s','/x/a.jpg','selected')")
    conn.execute("INSERT INTO captions(frame_id, caption_text) VALUES "
                 "('fx','spookums person, neon lighting')")
    conn.execute("INSERT INTO variation_batches(batch_id, spec, job_count, "
                 "created_at) VALUES ('batch_neon1','{}',1,'t')")
    conn.commit()
    res = global_search(conn, Project(), "neon")
    assert res["frames"] and res["frames"][0]["frame_id"] == "fx"
    assert res["batches"] == ["batch_neon1"]
    # history logs + prunes
    for i in range(7):
        log_stack_history(conn, {"loras": [["idl", 0.75]], "score": 0.9,
                                 "risk": "stable"}, keep=5)
    hist = load_presets(conn, "stack_history")
    assert len(hist) == 5
    assert hist[0]["payload"]["score"] == 0.9
    conn.close()
