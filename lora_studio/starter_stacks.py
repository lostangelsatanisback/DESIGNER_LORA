"""Recommended starter stacks - explainable, identity-first templates.

Suggests clean production stacks from the user's actual LoRA library using
sidecar metadata first (concept family, priority hints, known conflicts),
filename inference second.  No ML required; every recommendation carries
reason codes and resolves through stack intelligence for a preservation
score before it is ever shown.
"""
from __future__ import annotations

from typing import Optional

from .lora_explorer import LoraCard
from .stack_intelligence import resolve_stack

# (template name, concept family, optional second family)
STARTER_TEMPLATES: list[tuple[str, str, Optional[str]]] = [
    ("Identity + Editorial Style", "style", None),
    ("Identity + Studio Lighting", "lighting", None),
    ("Identity + Fashion Study", "fashion", None),
    ("Identity + Form Study", "pose", None),
    ("Identity + Material/Fabric Study", "texture", None),
    ("Identity + Environment Style", "environment", None),
    ("Identity + Composition Study", "composition", None),
    ("Identity + Fashion + Lighting", "fashion", "lighting"),
]


def _pick_identity(cards: list[LoraCard]) -> Optional[LoraCard]:
    """Best identity anchor: anchor-hinted first, then identity family,
    most recently modified wins."""
    ids = [c for c in cards if c.profile.family in ("identity", "character")]
    if not ids:
        return None
    return sorted(ids, key=lambda c: (
        getattr(c.profile, "priority_hint", "") != "anchor",
        -(0 if not c.modified_at else 1), c.lora_id))[0]


def _pick_concept(cards: list[LoraCard], family: str,
                  chosen: list[LoraCard]) -> Optional[LoraCard]:
    """Best concept card for a family: lowest identity risk first, sidecar
    metadata preferred; respects known conflicts against already-chosen
    LoRAs and families."""
    risk_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    chosen_ids = {c.lora_id for c in chosen}
    chosen_fams = {c.profile.family for c in chosen}
    pool = []
    for c in cards:
        if c.profile.family != family or c.lora_id in chosen_ids:
            continue
        if set(c.profile.known_conflicts) & chosen_ids:
            continue
        if set(getattr(c.profile, "conflict_families", ())) & chosen_fams:
            continue
        pool.append(c)
    if not pool:
        return None
    return sorted(pool, key=lambda c: (
        risk_rank.get(c.profile.identity_risk, 2),
        c.metadata_source == "inferred",      # sidecar-backed first
        c.lora_id))[0]


def recommend_starter_stacks(cards: list[LoraCard],
                             base_model: Optional[str] = None,
                             limit: int = 6) -> list[dict]:
    """Starter stack recommendations from the available library."""
    identity = _pick_identity(cards)
    if identity is None:
        return [{"name": "No identity anchor available",
                 "available": False, "stack": [], "reasons": [
                     "no_identity_lora_found",
                     "Train or import a character identity LoRA first - "
                     "starter stacks are built around an identity anchor."]}]
    out: list[dict] = []
    for name, fam_a, fam_b in STARTER_TEMPLATES:
        chosen = [identity]
        reasons = ["identity_anchor_selected",
                   f"identity anchor: {identity.lora_id}"]
        a = _pick_concept(cards, fam_a, chosen)
        if a is None:
            continue
        chosen.append(a)
        reasons.append(f"{fam_a}_concept_matched: {a.lora_id} "
                       f"({a.profile.identity_risk} risk, "
                       f"{'sidecar' if a.metadata_source != 'inferred' else 'inferred'} metadata)")
        if fam_b:
            b = _pick_concept(cards, fam_b, chosen)
            if b is None:
                continue
            chosen.append(b)
            reasons.append(f"{fam_b}_concept_matched: {b.lora_id}")
        # conservative weights: identity default; concepts capped at the
        # midpoint of their recommended range (studio-safe starting point)
        weights = {identity.lora_id: identity.profile.weight_default}
        for c in chosen[1:]:
            mid = round((c.profile.weight_min + c.profile.weight_max) / 2, 2)
            weights[c.lora_id] = min(c.profile.weight_default, mid)
        st = resolve_stack(chosen, weights, base_model)
        reasons.append("conservative_concept_weights")
        out.append({
            "name": name, "available": True,
            "identity": identity.lora_id,
            "stack": ([[st.identity_anchor.lora_id,
                        st.identity_anchor.weight]]
                      if st.identity_anchor else [])
            + [[i.lora_id, i.weight] for i in st.concept_loras],
            "weights": {k: round(v, 2) for k, v in weights.items()},
            "preservation_score": st.identity_preservation_score,
            "risk_level": st.risk_level,
            "reasons": reasons,
            "warnings": [w.message for w in st.warnings
                         if w.severity in ("caution", "critical")],
        })
        if len(out) >= limit:
            break
    if not out:
        out.append({"name": "Library needs concept LoRAs",
                    "available": False, "stack": [], "reasons": [
                        "no_concept_loras_matched",
                        "No suitable style/fashion/lighting concepts found "
                        "alongside the identity anchor."]})
    return out
