"""Concept Control Layer - explorer, sliders, stack intelligence, batches."""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora_studio import manifest  # noqa: E402
from lora_studio.batch_variations import (VariationGrid,  # noqa: E402
                                          expand_grid, save_batch)
from lora_studio.concept_control import (ConceptSliderState,  # noqa: E402
                                         load_presets, resolve_concept_weights,
                                         resolve_controlled_stack, save_preset,
                                         slider_specs)
from lora_studio.config import Project  # noqa: E402
from lora_studio.lora_explorer import (LoraCard,  # noqa: E402
                                       LoraInfluenceProfile, filter_cards,
                                       infer_profile, load_sidecar,
                                       read_safetensors_metadata, save_sidecar,
                                       scan_loras, sync_profiles_to_manifest)
from lora_studio.stack_intelligence import resolve_stack  # noqa: E402

CR = "/x/cyberrealisticPony_v18CoreShift.safetensors"


def _fake_safetensors(path: Path, meta: dict) -> None:
    header = json.dumps({"__metadata__": meta}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def _card(lora_id, family, risk="medium", default=None, conflicts=()):
    lo_hi = {"identity": (0.65, 0.85), "style": (0.15, 0.35)}
    lo, hi = lo_hi.get(family, (0.25, 0.55))
    return LoraCard(lora_id=lora_id, path=f"/x/{lora_id}.safetensors",
                    profile=LoraInfluenceProfile(
                        family=family, weight_min=lo, weight_max=hi,
                        weight_default=default or round((lo + hi) / 2, 2),
                        identity_risk=risk,
                        known_conflicts=list(conflicts)))


def test_safetensors_metadata_reader(tmp_path):
    f = tmp_path / "spook_character_v001.safetensors"
    _fake_safetensors(f, {"ss_network_dim": "32", "ss_output_name": "spook"})
    meta = read_safetensors_metadata(f)
    assert meta["ss_network_dim"] == "32"
    # corrupt file -> {} (never raises)
    (tmp_path / "bad.safetensors").write_bytes(b"\x00" * 4)
    assert read_safetensors_metadata(tmp_path / "bad.safetensors") == {}


def test_scan_sidecars_and_filtering(tmp_path):
    a = tmp_path / "spook_character_v001.safetensors"
    b = tmp_path / "neon_style_pack.safetensors"
    _fake_safetensors(a, {"ss_network_dim": "32"})
    _fake_safetensors(b, {})
    (tmp_path / "neon_style_pack.preview.png").write_bytes(b"png")
    prj = Project()
    cards = scan_loras(prj, dirs=[tmp_path])
    by_id = {c.lora_id: c for c in cards}
    assert by_id["spook_character_v001"].profile.family == "identity"
    assert by_id["spook_character_v001"].network_dim == 32
    assert by_id["neon_style_pack"].profile.family == "style"
    assert by_id["neon_style_pack"].preview        # thumbnail associated
    # sidecar overrides inference
    prof = infer_profile("neon_style_pack", {})
    prof.family = "lighting"
    prof.identity_risk = "low"
    save_sidecar(b, prof)
    assert load_sidecar(b).family == "lighting"
    cards = scan_loras(prj, dirs=[tmp_path])
    by_id = {c.lora_id: c for c in cards}
    assert by_id["neon_style_pack"].profile.family == "lighting"
    # filters + sorting
    assert [c.lora_id for c in filter_cards(cards, family="identity")] == \
        ["spook_character_v001"]
    assert filter_cards(cards, search="neon")[0].lora_id == "neon_style_pack"
    risk_sorted = filter_cards(cards, sort="identity_risk")
    assert risk_sorted[0].profile.identity_risk in ("none", "low")


def test_profiles_sync_to_manifest_v8(tmp_path):
    conn = manifest.connect(tmp_path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 8
    n = sync_profiles_to_manifest(conn, [_card("a_character", "identity")])
    assert n == 1
    fam = conn.execute("SELECT family FROM lora_influence_profiles "
                       "WHERE lora_id='a_character'").fetchone()[0]
    assert fam == "identity"
    conn.close()


def test_slider_specs_and_weight_resolution():
    specs = slider_specs()
    assert len(specs) == 9
    assert all(s["explanation"] for s in specs)
    cards = [_card("idl_character", "identity"), _card("sty", "style", "high")]
    # identity floor: even at slider 0 the anchor keeps >= half its range
    w0 = resolve_concept_weights(cards, ConceptSliderState(
        values={"identity_anchor_strength": 0.0, "style_intensity": 0.0}))
    assert w0["idl_character"] >= 0.75       # 0.65 + half of 0.20
    assert w0["sty"] == 0.15                 # style at its minimum
    w1 = resolve_concept_weights(cards, ConceptSliderState(
        values={"identity_anchor_strength": 1.0, "style_intensity": 1.0}))
    assert w1["idl_character"] == 0.85 and w1["sty"] == 0.35


def test_stack_intelligence_scores_and_warnings():
    cards = [_card("idl_character", "identity"),
             _card("fashion_a", "fashion", "low"),
             _card("style_a", "style", "high"),
             _card("style_b", "style", "high")]
    st = resolve_stack(cards, {"idl_character": 0.75, "fashion_a": 0.35,
                               "style_a": 0.3, "style_b": 0.3}, CR)
    assert st.base_model.startswith("CyberRealistic")
    assert st.identity_anchor.lora_id == "idl_character"
    assert st.identity_anchor.reason == "Primary identity preservation anchor"
    codes = set(st.reason_codes)
    assert "identity_anchor_present" in codes
    assert "multiple_high_risk_loras" in {w.code for w in st.warnings} or \
        any(w.code == "identity_risk_stack" for w in st.warnings)
    assert 0.0 < st.identity_preservation_score < 1.0
    # explicit known conflicts surface
    c2 = [_card("idl_character", "identity"),
          _card("a_style", "style", "high", conflicts=("b_fashion",)),
          _card("b_fashion", "fashion")]
    st2 = resolve_stack(c2, base_model=CR)
    assert any(w.code == "known_conflict" for w in st2.warnings)
    # missing anchor flagged
    st3 = resolve_stack([_card("s", "style", "high")], base_model=CR)
    assert any(w.code == "identity_anchor_missing" for w in st3.warnings)


def test_excessive_strength_normalized():
    cards = [_card("idl_character", "identity"),
             _card("f1", "fashion"), _card("f2", "wardrobe"),
             _card("p1", "pose"), _card("e1", "environment")]
    st = resolve_stack(cards, {"idl_character": 0.8, "f1": 0.55, "f2": 0.55,
                               "p1": 0.5, "e1": 0.5}, CR)
    assert st.total_concept_strength <= 1.60 + 1e-6
    assert "concept_strength_normalized" in st.reason_codes


def test_controlled_stack_and_presets(tmp_path):
    cards = [_card("idl_character", "identity"), _card("fash", "fashion")]
    st = resolve_controlled_stack(cards, ConceptSliderState(
        values={"garment_style_intensity": 1.0}), CR)
    assert st.identity_anchor and st.concept_loras[0].weight == 0.55
    conn = manifest.connect(tmp_path)
    save_preset(conn, "editorial_a", "lora_stack",
                {"sel": {"idl_character": 0.75}})
    got = load_presets(conn, "lora_stack")
    assert got[0]["name"] == "editorial_a"
    assert got[0]["payload"]["sel"]["idl_character"] == 0.75
    conn.close()


def test_batch_expansion_and_manifest(tmp_path):
    prj = Project()
    prj.trigger_token, prj.class_word = "gk_person", ""
    prj.base_model = CR
    cards = [_card("idl_character", "identity"), _card("sty", "style", "high")]
    grid = VariationGrid(
        prompt_tail="full body framing, soft studio lighting",
        seeds=[42, 1042],
        slider_axes=[{"slider": "style_intensity",
                      "values": [0.2, 0.4, 0.6]}])
    jobs = expand_grid(prj, cards, grid)
    assert len(jobs) == 6                      # 3 values x 2 seeds
    assert all(j.batch_id == jobs[0].batch_id for j in jobs)
    j = jobs[0]
    assert j.loras[0][0] == "idl_character"    # identity anchor held fixed
    assert "consistent character identity" in j.prompt
    assert "identity drift" in j.negative
    assert j.payload["sampler_name"] == "DPM++ SDE Karras"
    assert j.payload["override_settings"]["CLIP_stop_at_last_layers"] == 2
    # style weight actually sweeps
    ws = sorted({jj.loras[1][1] for jj in jobs})
    assert len(ws) == 3
    # manifest persistence
    conn = manifest.connect(tmp_path)
    bid = save_batch(conn, prj, grid, jobs)
    assert bid == j.batch_id
    n = conn.execute("SELECT COUNT(*) FROM variation_jobs "
                     "WHERE batch_id=?", (bid,)).fetchone()[0]
    assert n == 6
    conn.close()
    # cap respected
    grid2 = VariationGrid(seeds=list(range(100)), max_jobs=10)
    assert len(expand_grid(prj, cards, grid2)) == 10


# ---------------------------------------------------------------------------
# Feature A: Visual LoRA Explorer (rich sidecars, previews, degradation)
# ---------------------------------------------------------------------------

def test_rich_sidecar_schema_and_levels(tmp_path):
    from lora_studio.lora_explorer import (build_explorer_payload,
                                           find_preview_images)
    f = tmp_path / "example_style.safetensors"
    _fake_safetensors(f, {})
    (tmp_path / "example_style.json").write_text(json.dumps({
        "display_name": "Example Style",
        "concept_tags": ["lighting_mood", "composition_flow"],
        "description": "Studio-facing visual style concept.",
        "recommended_weight": 0.65,
        "category": "lighting",
        "preview_images": {"default": "example_style.preview.png",
                           "low": "example_style.low.png"},
    }))
    for name in ("example_style.preview.png", "example_style.low.png",
                 "example_style.medium.png"):
        (tmp_path / name).write_bytes(b"png")
    cards = scan_loras(Project(), dirs=[tmp_path])
    c = cards[0]
    assert c.display_name == "Example Style"
    assert c.description.startswith("Studio-facing")
    assert c.profile.family == "lighting"          # category mapped
    assert c.profile.weight_default == 0.65        # recommended_weight mapped
    assert c.profile.influence_tags == ["lighting_mood", "composition_flow"]
    assert c.metadata_source.endswith("example_style.json")
    # declared previews + beside-the-model discovery both work
    assert set(c.preview_levels) == {"default", "low", "medium"}
    assert c.has_preview
    payload = build_explorer_payload(cards)
    item = payload["items"][0]
    assert payload["count"] == 1
    assert item["preview_levels_available"] == ["default", "low", "medium"]
    assert item["preview_urls"]["low"].endswith("/example_style/low")
    # direct API parity
    levels = find_preview_images(f, None)
    assert "default" in levels


def test_previews_root_and_meta_json(tmp_path):
    lora_dir = tmp_path / "models"
    lora_dir.mkdir()
    f = lora_dir / "soft_glow.safetensors"
    _fake_safetensors(f, {})
    proot = tmp_path / "previews" / "lora"
    pdir = proot / "soft_glow"
    pdir.mkdir(parents=True)
    (pdir / "default.png").write_bytes(b"png")
    (pdir / "high.png").write_bytes(b"png")
    (pdir / "meta.json").write_text(json.dumps(
        {"display_name": "Soft Glow", "category": "lighting"}))
    cards = scan_loras(Project(), dirs=[lora_dir], previews_root=proot)
    c = cards[0]
    assert c.display_name == "Soft Glow"
    assert c.profile.family == "lighting"
    assert set(c.preview_levels) == {"default", "high"}


def test_pt_ckpt_and_graceful_degradation(tmp_path):
    (tmp_path / "legacy_pose.pt").write_bytes(b"x")
    (tmp_path / "old_detail.ckpt").write_bytes(b"x")
    (tmp_path / "broken.safetensors").write_bytes(b"\x00\x01")
    (tmp_path / "broken.json").write_text("{not json")     # tolerated
    cards = scan_loras(Project(), dirs=[tmp_path])
    ids = {c.lora_id for c in cards}
    assert ids == {"legacy_pose", "old_detail", "broken"}
    by_id = {c.lora_id: c for c in cards}
    assert by_id["legacy_pose"].profile.family == "pose"   # name inference
    assert by_id["legacy_pose"].metadata_source == "inferred"
    assert not by_id["broken"].has_preview                 # placeholder state
    assert by_id["broken"].display_name == "broken"


def test_discover_loras_explicit_roots(tmp_path):
    from lora_studio.lora_explorer import discover_loras
    _fake_safetensors(tmp_path / "a_character.safetensors", {})
    items = discover_loras([str(tmp_path), str(tmp_path / "missing")])
    assert len(items) == 1 and items[0].profile.family == "identity"


# ---------------------------------------------------------------------------
# Feature B: slider intelligence + stack intelligence v2
# ---------------------------------------------------------------------------

def test_family_aliases_and_unknown_fallback():
    from lora_studio.concept_control import (canonical_family, family_range,
                                             map_slider_to_weight)
    assert canonical_family("identity_anchor") == "identity"
    assert canonical_family("garment_style") == "wardrobe"
    assert canonical_family("lighting_mood") == "lighting"
    assert canonical_family("rendering_style") == "style"
    assert canonical_family("mystery_concept") == "unknown"
    assert family_range("mystery_concept") == (0.10, 0.30)
    # unknown family at full slider stays conservative (studio-safe)
    assert map_slider_to_weight(1.0, "mystery_concept").resolved_weight <= 0.30


def test_slider_mapping_curves_and_caps():
    from lora_studio.concept_control import map_slider_to_weight
    # monotonic smoothstep within the fashion range
    ws = [map_slider_to_weight(t / 10, "fashion").resolved_weight
          for t in range(11)]
    assert ws == sorted(ws) and ws[0] == 0.25 and ws[-1] == 0.55
    # slow start: first quarter of travel moves less than the linear share
    assert ws[2] - ws[0] < (0.55 - 0.25) * 0.2
    # identity protected band: slider 0 -> half-range, never lower
    m0 = map_slider_to_weight(0.0, "identity")
    assert m0.curve == "linear_protected" and m0.resolved_weight == 0.75
    # identity-aware damping when the stack is already heavy
    heavy = map_slider_to_weight(1.0, "style",
                                 {"total_concept_strength": 1.5,
                                  "anchor_weight": 0.8})
    assert heavy.identity_scaled and heavy.resolved_weight < 0.35


def test_numeric_slider_sync_roundtrip():
    from lora_studio.concept_control import (suggest_safer_slider_value,
                                             sync_numeric_to_slider,
                                             sync_slider_to_numeric)
    for fam in ("fashion", "lighting", "identity"):
        for t in (0.1, 0.5, 0.9):
            w = sync_slider_to_numeric(t, fam)
            t2 = sync_numeric_to_slider(w, fam)
            assert abs(sync_slider_to_numeric(t2, fam) - w) < 0.02
    # safer value pulls aggressive style back, never raises identity demand
    assert suggest_safer_slider_value(1.0, "style") < 1.0
    assert suggest_safer_slider_value(0.2, "identity") == 0.5


def test_priority_aware_rebalance_protects_anchor():
    cards = [_card("idl_character", "identity"),
             _card("f1", "fashion"), _card("w1", "wardrobe"),
             _card("p1", "pose"), _card("e1", "environment")]
    st = resolve_stack(cards, {"idl_character": 0.8, "f1": 0.55, "w1": 0.55,
                               "p1": 0.5, "e1": 0.5}, CR)
    assert st.total_concept_strength <= 1.60 + 1e-6
    # anchor untouched; lowest-priority concept (environment) trimmed first
    assert st.identity_anchor.weight == 0.8
    assert not st.identity_anchor.adjusted
    e1 = [i for i in st.concept_loras if i.lora_id == "e1"][0]
    p1 = [i for i in st.concept_loras if i.lora_id == "p1"][0]
    assert e1.adjusted and e1.weight < 0.5
    assert p1.weight == 0.5                      # higher priority untouched
    assert "concept_strength_normalized" in st.reason_codes


def test_pinned_overrides_respected():
    cards = [_card("idl_character", "identity"),
             _card("f1", "fashion"), _card("e1", "environment")]
    st = resolve_stack(cards, {"idl_character": 0.8, "f1": 1.0, "e1": 1.0},
                       CR, pinned={"f1", "e1"})
    f1 = [i for i in st.concept_loras if i.lora_id == "f1"][0]
    assert f1.pinned and f1.weight == 1.0 and not f1.adjusted
    assert "pinned_prevent_rebalance" in st.reason_codes
    assert any(w.code == "concept_strength_excessive" for w in st.warnings)


def test_risk_levels_and_severity_tiers():
    from lora_studio.stack_intelligence import SEVERITY_ORDER, risk_level_for
    assert risk_level_for(0.9) == "stable"
    assert risk_level_for(0.8) == "watch"
    assert risk_level_for(0.7) == "elevated"
    assert risk_level_for(0.5) == "high"
    # safe stack classifies stable, no caution+ warnings
    st = resolve_stack([_card("idl_character", "identity"),
                        _card("f1", "fashion", "low")],
                       {"idl_character": 0.8, "f1": 0.35}, CR)
    assert st.risk_level == "stable"
    assert all(SEVERITY_ORDER[w.severity] < 2 for w in st.warnings)
    # aggressive stack escalates and sorts critical warnings first
    st2 = resolve_stack([_card("s1", "style", "high"),
                         _card("s2", "style", "high")],
                        {"s1": 0.35, "s2": 0.35}, CR)
    assert st2.risk_level in ("elevated", "high")
    assert st2.warnings[0].severity == "critical"
    assert all(SEVERITY_ORDER[a.severity] >= SEVERITY_ORDER[b.severity]
               for a, b in zip(st2.warnings, st2.warnings[1:]))


def test_recommendations_and_apply_balance_schema():
    cards = [_card("idl_character", "identity", default=0.5),
             _card("sty_a", "style", "high"), _card("sty_b", "style", "high"),
             _card("tex", "texture"), _card("env", "environment")]
    st = resolve_stack(cards, {"idl_character": 0.5, "sty_a": 0.35,
                               "sty_b": 0.35, "tex": 0.3, "env": 0.3}, CR)
    codes = {r.code for r in st.recommendations}
    assert "raise_identity_anchor" in codes      # weak anchor -> raise it
    assert "use_batch_variation" in codes        # many modifiers active
    # recommended_weights is apply-ready: anchor raised, never reduced
    assert st.recommended_weights["idl_character"] >= 0.75
    assert set(st.recommended_weights) == {c.lora_id for c in cards}
    # conflict pair style+texture produces a deprioritize recommendation
    assert any(c.startswith("concept_conflict") for c in st.reason_codes)
    assert "deprioritize_conflict" in codes
    # UI-stable schema fields present
    j = st.to_json()
    for k in ("risk_level", "summary", "influence_pressure", "conflicts",
              "recommended_weights", "recommendations"):
        assert k in j
    assert st.summary.startswith("Identity ")


def test_influence_pressure_ranking():
    st = resolve_stack([_card("idl_character", "identity"),
                        _card("sty", "style", "high"),
                        _card("det", "detail", "low")],
                       {"idl_character": 0.8, "sty": 0.3, "det": 0.3}, CR)
    assert st.influence_pressure[0]["lora_id"] == "sty"   # risk + low priority
    assert st.influence_pressure[0]["pressure"] > \
        st.influence_pressure[-1]["pressure"]


def test_controlled_stack_with_overrides_and_pins():
    from lora_studio.concept_control import resolve_controlled_stack
    cards = [_card("idl_character", "identity"), _card("fash", "fashion")]
    st = resolve_controlled_stack(
        cards, ConceptSliderState(values={}), CR,
        overrides={"fash": 0.9})
    f = st.concept_loras[0]
    assert f.pinned and f.requested_weight == 0.9
    assert any(w.code == "weight_above_recommended" for w in st.warnings)


# ---------------------------------------------------------------------------
# Feature C: Batch Variation Controller (modes, axes, scoring, resumability)
# ---------------------------------------------------------------------------

def test_variation_axis_specs_and_modes():
    from lora_studio.batch_variations import (VARIATION_MODES, VariationAxis,
                                              estimate_job_count)
    # min/max/step expansion, deterministic + deduplicated
    ax = VariationAxis("style_intensity", minimum=0.2, maximum=0.6, step=0.2)
    assert ax.resolve_values("balanced") == [0.2, 0.4, 0.6]
    # low-risk mode clips aggressive values to its ceiling
    hot = VariationAxis("style_intensity", values=[0.2, 0.9, 1.0])
    assert hot.resolve_values("low_risk") == [0.2, 0.6]    # clipped + deduped
    assert hot.resolve_values("creative") == [0.2, 0.9, 1.0]
    # identity axis is protected: never below the safe band
    ida = VariationAxis("identity_anchor_strength", values=[0.0, 0.3, 0.9])
    assert min(ida.resolve_values("creative")) >= 0.5
    # unknown slider yields nothing
    assert VariationAxis("not_a_slider", values=[0.5]).resolve_values() == []
    # estimates + caps
    est = estimate_job_count([ax, hot], seeds=[1, 2], mode="low_risk")
    assert est["estimated"] == 3 * 2 * 2 and est["cap"] == 24
    assert est["within_cap"]
    big = estimate_job_count(
        [VariationAxis("style_intensity",
                       values=[i / 20 for i in range(13)])],
        seeds=[1, 2, 3], mode="low_risk")
    assert not big["within_cap"] and "exceeds" in big["guidance"]
    assert VARIATION_MODES["creative"]["cap"] == 128


def test_multi_axis_grid_scoring_and_anchor_fixed(tmp_path):
    from lora_studio.batch_variations import load_batch
    prj = Project()
    prj.trigger_token, prj.base_model = "gk_person", CR
    cards = [_card("idl_character", "identity"),
             _card("sty", "style", "high"), _card("lit", "lighting", "low")]
    grid = VariationGrid(
        mode="balanced", seeds=[42],
        slider_axes=[{"slider": "style_intensity", "values": [0.2, 0.8]},
                     {"slider": "lighting_mood",
                      "minimum": 0.2, "maximum": 0.6, "step": 0.4}])
    jobs = expand_grid(prj, cards, grid)
    assert len(jobs) == 4                          # 2 x 2 x 1 seed
    anchors = {j.loras[0][0] for j in jobs}
    assert anchors == {"idl_character"}            # anchor fixed every job
    assert all(0.0 < j.preservation_score <= 1.0 for j in jobs)
    assert all(j.risk_level in ("info", "caution", "high_risk",
                                "blocked_or_needs_review") for j in jobs)
    assert all(j.status == "planned" for j in jobs)
    # manifest roundtrip with v9 fields
    conn = manifest.connect(tmp_path)
    bid = save_batch(conn, prj, grid, jobs)
    data = load_batch(conn, bid)
    assert data["mode"] == "balanced" and data["hard_cap"] == 64
    assert data["identity_anchor"].startswith("idl_character:")
    assert len(data["jobs"]) == 4
    j0 = data["jobs"][0]
    assert j0["risk_level"] in ("info", "caution", "high_risk",
                                "blocked_or_needs_review")
    assert j0["status"] == "planned"
    assert load_batch(conn, "nope") == {}
    conn.close()


def test_mode_cap_enforced_and_blocked_without_anchor():
    prj = Project()
    prj.base_model = CR
    cards = [_card("idl_character", "identity"), _card("sty", "style")]
    grid = VariationGrid(
        mode="low_risk", seeds=list(range(30)),
        slider_axes=[{"slider": "style_intensity", "values": [0.2, 0.4]}])
    assert len(expand_grid(prj, cards, grid)) == 24    # low-risk hard cap
    # no identity anchor -> every job flagged for review
    jobs = expand_grid(prj, [_card("sty", "style", "high")], VariationGrid(
        mode="balanced",
        slider_axes=[{"slider": "style_intensity", "values": [0.3]}]))
    assert jobs[0].risk_level == "blocked_or_needs_review"


def test_variation_resume_and_status(tmp_path):
    prj = Project()
    prj.base_model = CR
    cards = [_card("idl_character", "identity"), _card("sty", "style")]
    grid = VariationGrid(slider_axes=[
        {"slider": "style_intensity", "values": [0.2, 0.4]}])
    jobs = expand_grid(prj, cards, grid)
    conn = manifest.connect(tmp_path)
    bid = save_batch(conn, prj, grid, jobs)
    # simulate one generated job; resume skips it
    conn.execute("UPDATE variation_jobs SET output_path='/x/v000.png', "
                 "status='generated' WHERE batch_id=? AND "
                 "variation_id='v000'", (bid,))
    conn.commit()
    from lora_studio.batch_variations import load_batch
    data = load_batch(conn, bid)
    statuses = {j["variation_id"]: j["status"] for j in data["jobs"]}
    assert statuses["v000"] == "generated" and statuses["v001"] == "planned"
    conn.close()


def test_variation_axes_metadata():
    from lora_studio.concept_control import variation_axes
    axes = variation_axes()
    by_id = {a["slider"]: a for a in axes}
    assert by_id["identity_anchor_strength"]["identity_impact"] == "protected"
    assert by_id["style_intensity"]["identity_impact"] == "high"
    assert by_id["garment_style_intensity"]["identity_impact"] == "low"
    assert all(a["explanation"] and a["step"] > 0 for a in axes)


# ---------------------------------------------------------------------------
# Feature D: Concept Metadata & Sidecar System Hardening
# ---------------------------------------------------------------------------

V2_SIDECAR = {
    "schema_version": 2,
    "display_name": "Example Concept LoRA",
    "concept_family": "fashion",
    "concept_tags": ["studio", "fabric_detail", "soft_lighting"],
    "control_axes": [
        {"axis_id": "style_intensity", "label": "Style Intensity",
         "description": "How strongly this LoRA influences visual style.",
         "recommended_range": [0.2, 0.75], "safe_default": 0.45,
         "response_curve": "damped", "identity_sensitivity": "medium"},
        {"label": "missing id - skipped"},
    ],
    "priority_hint": "supporting",
    "identity_risk_level": "high",
    "recommended_weight_range": [0.25, 0.65],
    "known_conflicts": [
        {"concept_family": "lighting",
         "reason": "May compete with lighting treatment.",
         "severity": "high"},
        "rival_lora",
    ],
    "preview_images": {"default": "ex.preview.png"},
    "signal_hooks": {"clip_probe": "reserved"},
    "notes": "Studio-safe concept metadata for controlled stack planning.",
}


def test_v2_sidecar_normalization():
    from lora_studio.concept_metadata import normalize_concept_metadata
    cm = normalize_concept_metadata(V2_SIDECAR, "example")
    assert cm.schema_version == 2
    assert cm.display_name == "Example Concept LoRA"
    assert cm.priority_hint == "supporting"
    assert cm.identity_risk_level == "high"
    assert cm.recommended_weight_range == (0.25, 0.65)
    assert len(cm.control_axes) == 1            # axis without id skipped
    ax = cm.control_axes[0]
    assert ax.recommended_range == (0.2, 0.75) and ax.safe_default == 0.45
    assert ax.response_curve == "damped"
    assert len(cm.known_conflicts) == 2
    fam_c = cm.known_conflicts[0]
    assert fam_c.concept_family == "lighting" and fam_c.severity == "high"
    assert cm.known_conflicts[1].lora_id == "rival_lora"
    assert cm.signal_hooks == {"clip_probe": "reserved"}   # reserved field
    assert any("skipped" in w for w in cm.normalization_warnings)


def test_legacy_and_invalid_metadata_normalization():
    from lora_studio.concept_metadata import normalize_concept_metadata
    # legacy simple sidecar: tags + preview map into the new structure
    cm = normalize_concept_metadata(
        {"tags": ["style", "lighting"], "preview": "preview.jpg"}, "old")
    assert cm.concept_tags == ["style", "lighting"]
    assert cm.preview_images == {"default": "preview.jpg"}
    assert cm.concept_family == "general_concept"
    assert cm.identity_risk_level == "medium"
    assert cm.recommended_weight_range == (0.2, 0.7)
    assert any("legacy" in w for w in cm.normalization_warnings)
    # unknown enums -> safe defaults + warnings; None -> full defaults
    cm2 = normalize_concept_metadata(
        {"priority_hint": "ultra", "identity_risk_level": "extreme",
         "recommended_weight_range": "broken"}, "x")
    assert cm2.priority_hint == "normal"
    assert cm2.identity_risk_level == "medium"
    assert cm2.recommended_weight_range == (0.2, 0.7)
    assert len(cm2.normalization_warnings) == 3
    assert normalize_concept_metadata(None, "y").display_name == "y"


def test_response_curves_registry():
    from lora_studio.concept_metadata import curve_fn
    for name in ("linear", "slow_start", "fast_start", "damped", "stepped"):
        f = curve_fn(name)
        assert f(0.0) == 0.0 and f(1.0) == 1.0
    assert curve_fn("slow_start")(0.5) < 0.5 < curve_fn("fast_start")(0.5)
    assert curve_fn("stepped")(0.6) == 0.5
    assert curve_fn("unknown")(0.5) == curve_fn("damped")(0.5)


def test_v2_sidecar_flows_into_scan_and_payload(tmp_path):
    from lora_studio.lora_explorer import build_explorer_payload
    f = tmp_path / "example_concept.safetensors"
    _fake_safetensors(f, {})
    (tmp_path / "example_concept.concept.json").write_text(
        json.dumps(V2_SIDECAR))
    (tmp_path / "ex.preview.png").write_bytes(b"png")
    cards = scan_loras(Project(), dirs=[tmp_path])
    c = cards[0]
    assert c.profile.priority_hint == "supporting"
    assert c.profile.identity_risk == "high"
    assert (c.profile.weight_min, c.profile.weight_max) == (0.25, 0.65)
    assert c.profile.conflict_families == ["lighting"]
    assert c.preview_levels["default"].endswith("ex.preview.png")
    meta = c.concept_meta
    assert meta["schema_version"] == 2
    assert len(meta["control_axes"]) == 1
    payload = build_explorer_payload(cards)
    assert payload["items"][0]["concept_meta"]["priority_hint"] == "supporting"
    # inferred cards also carry normalized metadata
    _fake_safetensors(tmp_path / "plain_style.safetensors", {})
    cards2 = scan_loras(Project(), dirs=[tmp_path])
    plain = [c2 for c2 in cards2 if c2.lora_id == "plain_style"][0]
    assert plain.concept_meta["metadata_source"] == "inferred"
    assert plain.concept_meta["concept_family"] == "style"


def test_priority_hints_in_resolver():
    from lora_studio.concept_metadata import priority_for_hint
    assert priority_for_hint(60, "supporting") == 45
    assert priority_for_hint(30, "experimental") == 5
    # supporting LoRA outweighing the anchor -> caution
    sup = _card("wardrobe_support", "wardrobe")
    sup.profile.priority_hint = "supporting"
    st = resolve_stack([_card("idl_character", "identity"), sup],
                       {"idl_character": 0.65, "wardrobe_support": 0.7}, CR)
    assert "supporting_dominates_stack" in st.reason_codes
    # anchor-hinted concept survives rebalance; others are trimmed
    keep = _card("pose_anchor", "pose")
    keep.profile.priority_hint = "anchor"
    cards = [_card("idl_character", "identity"), keep,
             _card("e1", "environment"), _card("f1", "fashion"),
             _card("w1", "wardrobe")]
    st2 = resolve_stack(cards, {"idl_character": 0.8, "pose_anchor": 0.35,
                                "e1": 0.5, "f1": 0.55, "w1": 0.55}, CR)
    pa = [i for i in st2.concept_loras if i.lora_id == "pose_anchor"][0]
    assert not pa.adjusted and pa.weight == 0.35
    assert st2.total_concept_strength <= 1.60 + 1e-6
    # family-level known conflict against an active family
    fc = _card("soft_glow_set", "style", "high")
    fc.profile.conflict_families = ["lighting"]
    st3 = resolve_stack([_card("idl_character", "identity"), fc,
                         _card("lit", "lighting", "low")],
                        {"idl_character": 0.8, "soft_glow_set": 0.3,
                         "lit": 0.2}, CR)
    assert "known_family_conflict" in st3.reason_codes
    assert any("family:lighting" in c for c in st3.conflicts)


# ---------------------------------------------------------------------------
# v3.8: One-Click Character + Concept Stack workflow
# ---------------------------------------------------------------------------

def test_starter_stack_recommendations():
    from lora_studio.starter_stacks import recommend_starter_stacks
    lit = _card("soft_studio_light", "lighting", "low")
    fash = _card("editorial_wardrobe", "fashion", "low")
    sty = _card("film_style", "style", "high")
    cards = [_card("spook_character_v1", "identity"), lit, fash, sty]
    recs = recommend_starter_stacks(cards, CR)
    by_name = {r["name"]: r for r in recs}
    assert by_name["Identity + Studio Lighting"]["available"]
    s = by_name["Identity + Studio Lighting"]
    assert s["stack"][0][0] == "spook_character_v1"      # anchor first
    assert s["identity"] == "spook_character_v1"
    assert 0 < s["preservation_score"] <= 1.0
    assert any("identity anchor" in r for r in s["reasons"])
    assert "conservative_concept_weights" in s["reasons"]
    # combo template picks two non-conflicting families
    combo = by_name.get("Identity + Fashion + Lighting")
    assert combo and len(combo["stack"]) == 3
    # family conflict respected: lighting-conflicting fashion is skipped
    fash2 = _card("clashing_wardrobe", "fashion", "low")
    fash2.profile.conflict_families = ["lighting"]
    recs2 = recommend_starter_stacks(
        [_card("idl_character", "identity"), fash2, lit], CR)
    combo2 = [r for r in recs2 if r["name"] == "Identity + Fashion + Lighting"]
    assert not combo2 or "clashing_wardrobe" not in \
        [l[0] for r in combo2 for l in r["stack"]]
    # no identity -> explainable empty state
    none = recommend_starter_stacks([lit, fash], CR)
    assert not none[0]["available"]
    assert "no_identity_lora_found" in none[0]["reasons"]


def test_workflow_overview_and_cache(tmp_path):
    from lora_studio.lora_explorer import scan_loras_cached, _SCAN_CACHE
    from lora_studio.stack_workflow import workflow_overview
    _fake_safetensors(tmp_path / "a_character.safetensors", {})
    _fake_safetensors(tmp_path / "b_style.safetensors", {})
    prj = Project()
    prj.lora_output_dir = str(tmp_path)
    _SCAN_CACHE["key"] = None
    cards = scan_loras_cached(prj)
    assert {c.lora_id for c in cards} == {"a_character", "b_style"}
    assert scan_loras_cached(prj) is cards            # cache hit
    # adding a file invalidates (dir mtime + entry count key)
    _fake_safetensors(tmp_path / "c_light.safetensors", {})
    cards2 = scan_loras_cached(prj)
    assert len(cards2) == 3 and cards2 is not cards
    ov = workflow_overview(cards2)
    assert ov["ready"] and ov["identity_candidates"] == ["a_character"]
    assert "style" in ov["concept_families"]
    assert not workflow_overview([])["ready"]


def test_stack_preset_v2_roundtrip_and_legacy(tmp_path):
    from lora_studio.stack_workflow import (make_stack_preset,
                                            validate_stack_preset)
    cards = [_card("idl_character", "identity"), _card("fash", "fashion")]
    st = resolve_stack(cards, {"idl_character": 0.75, "fash": 0.4}, CR)
    rec = make_stack_preset("editorial_v1", st,
                            requested_weights={"idl_character": 0.75,
                                               "fash": 0.4},
                            slider_state={"garment_style_intensity": 0.6},
                            notes="studio baseline")
    assert rec["preset_version"] == 2 and rec["preset_id"].startswith("stack_")
    assert rec["identity"]["lora_id"] == "idl_character"
    assert rec["handoff"]["loras"][0] == ["idl_character", 0.75]
    assert rec["preservation_score"] == st.identity_preservation_score
    # v2 validates as-is
    same, w = validate_stack_preset(rec)
    assert same["preset_id"] == rec["preset_id"] and not w
    # legacy normalizes with warning, handoff built from sel
    legacy, w2 = validate_stack_preset(
        {"sel": {"idl_character": 0.8, "sty": 0.3}})
    assert legacy["preset_version"] == 2
    assert ["idl_character", 0.8] in legacy["handoff"]["loras"]
    assert any("legacy" in x for x in w2)
    # garbage never raises
    empty, _ = validate_stack_preset(None)
    assert empty["handoff"]["loras"] == []
    # manifest persistence via existing preset store
    conn = manifest.connect(tmp_path)
    save_preset(conn, rec["name"], "lora_stack", rec)
    got = load_presets(conn, "lora_stack")[0]["payload"]
    assert got["preset_id"] == rec["preset_id"]
    conn.close()


def test_send_stack_to_playground_handoff(tmp_path):
    from lora_studio.stack_workflow import send_stack_to_playground
    prj = Project()
    prj.trigger_token, prj.class_word = "spookums", "person"
    prj.base_model = CR
    cards = [_card("idl_character", "identity"), _card("fash", "fashion")]
    st = resolve_stack(cards, {"idl_character": 0.75, "fash": 0.4}, CR)
    import lora_studio.pipeline_dag as pd
    orig = pd.write_playground_preset
    captured = {}
    def fake(prj2, name, loras, path=None):
        captured.update(name=name, loras=loras)
        return tmp_path / "pp.json"
    pd.write_playground_preset = fake
    try:
        send_stack_to_playground(prj, "wf_test", st)
    finally:
        pd.write_playground_preset = orig
    assert captured["name"] == "wf_test"
    assert captured["loras"][0] == ("idl_character", 0.75)
    # empty stack raises a clear error
    empty = resolve_stack([], base_model=CR)
    try:
        send_stack_to_playground(prj, "x", empty)
        assert False
    except ValueError as exc:
        assert "identity anchor" in str(exc)
