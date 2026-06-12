"""MergeForge tests - analyzer, compatibility, weights, recipes, engine."""
import json
from pathlib import Path

import numpy as np

from lora_studio.lora_explorer import LoraCard, LoraInfluenceProfile
from lora_studio.mergeforge import (CompatibilityResult, LoraHealth,
                                    _unique_path, analyze_library,
                                    assess_health, build_recommendations,
                                    classify_for_merge, execute_recipe,
                                    load_recipes, make_recipe,
                                    merge_weighted_sum,
                                    recommend_merge_weights, save_recipe,
                                    score_compatibility, stack_compatibility,
                                    validate_recipe, wizard_plan)


def _card(lora_id, family, risk="medium", conflicts=(), conflict_fams=(),
          hint="normal", source="/x/sidecar.concept.json", preview=True):
    return LoraCard(
        lora_id=lora_id, path=f"/x/{lora_id}.safetensors",
        preview=("/x/p.png" if preview else None),
        metadata_source=source, network_dim=32,
        sd_metadata={"ss_output_name": lora_id},
        display_name=lora_id, has_preview=preview,
        preview_levels=({"default": f"/x/{lora_id}.png"} if preview else {}),
        profile=LoraInfluenceProfile(
            family=family, identity_risk=risk,
            known_conflicts=list(conflicts),
            conflict_families=list(conflict_fams),
            priority_hint=hint))


def _lora_sd(rng, base, rank, out=16, inn=12, alpha=None):
    d = rng.normal(size=(rank, inn)).astype(np.float32)
    u = rng.normal(size=(out, rank)).astype(np.float32)
    return {f"{base}.lora_down.weight": d, f"{base}.lora_up.weight": u,
            f"{base}.alpha": np.float32(float(alpha or rank))}


def _delta(sd, base, w):
    down = sd[f"{base}.lora_down.weight"]
    up = sd[f"{base}.lora_up.weight"]
    alpha = float(sd[f"{base}.alpha"])
    return w * (alpha / down.shape[0]) * (up @ down)


# ---------------------------------------------------------------------------
# weighted_sum engine
# ---------------------------------------------------------------------------

def test_weighted_sum_exact_at_full_rank():
    rng = np.random.default_rng(7)
    a = _lora_sd(rng, "lora_unet_x", 4)
    b = _lora_sd(rng, "lora_unet_x", 8, alpha=4.0)
    blocks = [{"te": 1, "down": 1, "mid": 1, "up": 1}] * 2
    merged, stats = merge_weighted_sum([a, b], [0.7, 0.3], blocks,
                                       target_rank=12)
    assert stats["modules"] == 1 and stats["rank_truncated"] == 0
    md = merged["lora_unet_x.lora_down.weight"].astype(np.float32)
    mu = merged["lora_unet_x.lora_up.weight"].astype(np.float32)
    got = (float(merged["lora_unet_x.alpha"]) / md.shape[0]) * (mu @ md)
    ref = _delta(a, "lora_unet_x", 0.7) + _delta(b, "lora_unet_x", 0.3)
    rel = float(np.abs(got - ref).max() / np.abs(ref).max())
    assert rel < 0.01, f"weighted_sum not exact at full rank: {rel}"


def test_weighted_sum_truncation_flagged_and_conv_shapes():
    rng = np.random.default_rng(8)
    a = _lora_sd(rng, "lora_unet_x", 4)
    b = _lora_sd(rng, "lora_unet_x", 8)
    blocks = [{"te": 1, "down": 1, "mid": 1, "up": 1}] * 2
    _, stats = merge_weighted_sum([a, b], [0.5, 0.5], blocks)  # auto rank=8
    assert stats["rank_truncated"] == 1  # true rank 12 > kept 8
    # conv (4D) module keeps kohya shapes
    c = {"lora_unet_c.lora_down.weight":
         rng.normal(size=(4, 8, 3, 3)).astype(np.float32),
         "lora_unet_c.lora_up.weight":
         rng.normal(size=(16, 4, 1, 1)).astype(np.float32),
         "lora_unet_c.alpha": np.float32(4.0)}
    d = {k: v.copy() for k, v in c.items()}
    m, st = merge_weighted_sum([c, d], [0.5, 0.5], blocks)
    assert st["modules"] == 1
    assert m["lora_unet_c.lora_down.weight"].shape == (4, 8, 3, 3)
    assert m["lora_unet_c.lora_up.weight"].shape == (16, 4, 1, 1)


def test_weighted_sum_zero_block_skips():
    rng = np.random.default_rng(9)
    a = _lora_sd(rng, "lora_te_x", 4)
    blocks = [{"te": 0.0, "down": 1, "mid": 1, "up": 1}] * 2
    merged, stats = merge_weighted_sum([a, a], [1.0, 1.0], blocks)
    assert stats["zeroed"] == 2 and not merged


# ---------------------------------------------------------------------------
# compatibility + weights
# ---------------------------------------------------------------------------

def test_dual_identity_penalized():
    r = score_compatibility(_card("id_a", "identity", risk="none"),
                            _card("id_b", "identity", risk="none"))
    assert isinstance(r, CompatibilityResult)
    assert "dual_identity" in r.reason_codes
    assert r.score <= 60 and r.verdict in ("workable", "risky", "avoid")


def test_declared_conflict_bidirectional():
    a = _card("id_a", "identity", risk="none", conflicts=["style_b"])
    b = _card("style_b", "style", risk="high")
    r1, r2 = score_compatibility(a, b), score_compatibility(b, a)
    assert "declared_conflict_pair" in r1.reason_codes
    assert r1.score == r2.score  # symmetric
    assert r1.verdict in ("risky", "avoid")


def test_anchor_plus_concept_scores_well():
    r = score_compatibility(_card("id_a", "identity", risk="none"),
                            _card("ward", "wardrobe", risk="low"))
    assert r.score >= 85 and "anchor_plus_concept" in r.reason_codes


def test_stack_compatibility_weakest_pair_rules():
    cards = [_card("id_a", "identity", risk="none"),
             _card("id_b", "identity", risk="none"),
             _card("ward", "wardrobe", risk="low")]
    s = stack_compatibility(cards)
    assert s["overall_score"] == min(p["score"] for p in s["pairs"])


def test_recommend_weights_identity_first_with_damping():
    cards = [_card("id_a", "identity", risk="none"),
             _card("w1", "wardrobe"), _card("w2", "fashion"),
             _card("d1", "detail"), _card("l1", "lighting")]
    r = recommend_merge_weights(cards)
    assert r["weights"]["id_a"] == 0.70
    assert all(v < 0.35 for k, v in r["weights"].items() if k != "id_a")
    assert any("damped" in n for n in r["notes"])


# ---------------------------------------------------------------------------
# health + classification + analyzer
# ---------------------------------------------------------------------------

def test_health_penalties_and_grades():
    good = assess_health(_card("a", "identity"))
    assert isinstance(good, LoraHealth) and good.grade in ("A", "B")
    bare = assess_health(_card("b", "style", source="inferred",
                               preview=False))
    assert bare.score < good.score and bare.reasons


def test_classification_reason_codes():
    c = classify_for_merge(_card("a", "wardrobe"))
    assert c["role"] == "wardrobe_layer"
    assert "family_from_sidecar" in c["reason_codes"]
    anchor = classify_for_merge(_card("b", "style", hint="anchor"))
    assert anchor["role"] == "identity_anchor"
    assert "priority_hint_anchor" in anchor["reason_codes"]


def test_analyze_library_and_recommendation_groups():
    cards = [_card("id_a", "identity", risk="none"),
             _card("ward", "wardrobe"), _card("det", "detail"),
             _card("sty", "style", risk="high"), _card("lit", "lighting")]
    rep = analyze_library(object(), cards=cards)
    assert rep["library_size"] == 5 and rep["merge_ready"]
    assert rep["role_counts"]["identity_anchor"] == 1
    groups = build_recommendations(object(), cards=cards)
    ids = {g["group_id"] for g in groups}
    assert "character_complete" in ids and "character_style_fusion" in ids
    cc = next(g for g in groups if g["group_id"] == "character_complete")
    assert len(cc["members"]) == 3
    assert cc["members"][0]["weight"] == 0.70


def test_wizard_plan_smoke():
    cards = [_card("id_a", "identity", risk="none"),
             _card("ward", "wardrobe")]

    class _P:
        base_model = "cyberrealisticPony_v180Coreshift.safetensors"
    plan = wizard_plan(_P(), ["id_a", "ward", "ghost"], cards=cards)
    assert plan["ready"] and plan["missing"] == ["ghost"]
    assert plan["recipe_draft"]["recipe_version"] == 1
    assert len(plan["recipe_draft"]["inputs"]) == 2


# ---------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------

def test_recipe_roundtrip_and_sha256(tmp_path):
    f1, f2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    f1.write_bytes(b"\x01" * 64)
    f2.write_bytes(b"\x02" * 64)
    rec = make_recipe("combo", [(str(f1), 0.7), (str(f2), 0.3)])
    assert rec["recipe_version"] == 1
    assert all(len(i["sha256"]) == 64 for i in rec["inputs"])
    out = tmp_path / "out"
    out.mkdir()
    save_recipe(out, rec)
    loaded = load_recipes(out)
    assert loaded and loaded[0]["recipe_id"] == rec["recipe_id"]


def test_validate_recipe_tolerant():
    rec, probs = validate_recipe({"inputs": [
        {"path": "/x/a.safetensors", "weight": "0.5"},
        {"path": "/x/b.safetensors", "weight": 0.5},
        {"weight": 1.0},                      # malformed -> dropped
        {"path": "/x/c.safetensors", "weight": "zebra"}]})
    assert len(rec["inputs"]) == 2
    assert any("malformed" in p for p in probs)
    assert any("bad weight" in p for p in probs)
    garbage, gp = validate_recipe(None)
    assert garbage["method"] == "weighted_sum" and gp


def test_unique_path_never_overwrites(tmp_path):
    p = tmp_path / "m.safetensors"
    assert _unique_path(p) == p
    p.write_bytes(b"x")
    p2 = _unique_path(p)
    assert p2.name == "m_1.safetensors" and not p2.exists()


def test_execute_recipe_graceful_without_safetensors(tmp_path):
    """Engine must fail with a clear FATAL line (never a raw exception)
    whether safetensors is installed (invalid header -> actionable
    message) or absent (dependency message). Deterministic on Mac,
    Linux, Colab and CI."""
    f1, f2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    f1.write_bytes(b"\x00" * 16)            # invalid header either way
    f2.write_bytes(b"\x00" * 16)

    class _P:
        lora_output_dir = str(tmp_path / "out")
        base_model = ""
    rec = make_recipe("t", [(str(f1), 0.7), (str(f2), 0.3)],
                      hash_files=False)
    lines = list(execute_recipe(_P(), tmp_path, rec))   # must not raise
    assert any(line.startswith("FATAL") for line in lines)


def test_execute_recipe_actionable_on_corrupt_file(tmp_path):
    """Simulate an installed safetensors whose loader rejects the file:
    the engine must convert the raw error into an actionable FATAL."""
    import sys
    import types

    f1, f2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    f1.write_bytes(b"\x00" * 16)
    f2.write_bytes(b"\x00" * 16)

    class _FakeSafetensorError(Exception):
        pass

    def _raise(path):
        raise _FakeSafetensorError(
            "Error while deserializing header: invalid JSON in header")

    fake_st = types.ModuleType("safetensors")
    fake_np = types.ModuleType("safetensors.numpy")
    fake_np.load_file = _raise
    fake_np.save_file = lambda *a, **k: None
    fake_st.numpy = fake_np
    saved = {k: sys.modules.get(k) for k in ("safetensors",
                                             "safetensors.numpy")}
    sys.modules["safetensors"] = fake_st
    sys.modules["safetensors.numpy"] = fake_np
    try:
        class _P:
            lora_output_dir = str(tmp_path / "out")
            base_model = ""
        rec = make_recipe("t", [(str(f1), 0.7), (str(f2), 0.3)],
                          hash_files=False)
        lines = list(execute_recipe(_P(), tmp_path, rec))
        fatal = [l for l in lines if l.startswith("FATAL")]
        assert fatal, lines
        assert "not a readable .safetensors" in fatal[0]
        assert "re-export or re-download" in fatal[0]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
