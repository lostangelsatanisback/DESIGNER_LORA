"""
grokkie_weight_engine.py — Weight Intelligence & Semantic Isolation
====================================================================
Codename: Z. Phase 4 - The 10x Combinator

Calculates optimal LoRA weights and performs Semantic Isolation,
binding specific macOS Vision masks to specific LoRAs based on their
inferred category to prevent concept bleed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

W_MIN = 0.1
W_MAX = 1.0
WEIGHT_CAP_SDXL = 2.5
WEIGHT_CAP_SD15 = 2.0
DEFAULT_WEIGHT_CAP = 2.5

class WeightSignal(Enum):
    PROMPT_SEMANTIC = "prompt_semantic"
    IMAGE_CONTEXT = "image_context"
    HISTORICAL_PERF = "historical_perf"
    TASTE_AFFINITY = "taste_affinity"

@dataclass
class LoraWeightAssignment:
    lora_name: str
    fused_weight: float
    is_primary: bool = False
    signals: Dict[str, float] = field(default_factory=dict)
    conflict_adjustments: List[str] = field(default_factory=list)
    # Phase 4: Semantic Isolation Binding
    region_binding: str = "full_image" 
    requires_mask: bool = False

class WeightEngine:
    def __init__(self, target_base: str = "SDXL"):
        self.target_base = target_base.upper()
        self.weight_cap = WEIGHT_CAP_SDXL if self.target_base == "SDXL" else WEIGHT_CAP_SD15

    def _determine_semantic_isolation(self, category: str, routed_region: str) -> tuple[str, bool]:
        """
        Phase 4 10x Feature: Maps a LoRA category to a physical mask.
        Prevents a "silk blouse" LoRA from turning the background into silk.
        """
        cat = category.lower()
        if "upper" in cat or "shirt" in cat:
            return "upper_body", True
        if "lower" in cat or "pants" in cat:
            return "lower_body", True
        if "face" in cat or "identity" in cat:
            return "face", True
        if "background" in cat or "environment" in cat:
            return "background", True
            
        # Styles, lighting, and full outfits apply to the whole canvas
        return "full_image", False

    def compute_weights(
        self,
        prompt: str,
        lora_candidates: List[Dict[str, Any]],
        edit_type: str = "unknown",
        routed_region: str = "full_image",
        taste_profile: Optional[Any] = None,
    ) -> List[LoraWeightAssignment]:
        """Computes weights and binds macOS Vision masks to the candidates."""
        assignments = []
        total_weight = 0.0

        # Sort candidates by their initial confidence
        sorted_cands = sorted(lora_candidates, key=lambda x: x.get("confidence", 0.5), reverse=True)

        for i, cand in enumerate(sorted_cands):
            name = cand.get("name", "unknown")
            category = cand.get("category", "unknown")
            
            # Base weight calculation
            computed_w = min(W_MAX, cand.get("confidence", 0.5) * 1.2)
            
            # Phase 4: Bind the mask region
            binding_region, needs_mask = self._determine_semantic_isolation(category, routed_region)

            # Cap management
            if total_weight + computed_w > self.weight_cap:
                computed_w = max(W_MIN, self.weight_cap - total_weight)
                
            total_weight += computed_w

            assignments.append(LoraWeightAssignment(
                lora_name=name,
                fused_weight=round(computed_w, 3),
                is_primary=(i == 0),
                region_binding=binding_region,
                requires_mask=needs_mask
            ))

        return assignments

# --------------------------------------------------------------------------- #
# Singleton accessor
# --------------------------------------------------------------------------- #

_singleton: WeightEngine | None = None

def get_weight_engine(target_base: str = "SDXL") -> WeightEngine:
    global _singleton
    if _singleton is None:
        _singleton = WeightEngine(target_base)
    return _singleton


if __name__ == "__main__":
    print("WeightEngine (Phase 4 Semantic Isolation) Ready.")
