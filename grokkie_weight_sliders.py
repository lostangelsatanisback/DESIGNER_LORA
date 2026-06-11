"""
grokkie_weight_sliders.py — Per-LoRA Slider System for Redresser Studio Deluxe.

Phase-3: manages per-LoRA slider state, session lifecycle, user adjustment
processing, rebalancing coordination, and weight budget tracking.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RebalanceMode(Enum):
    PROPORTIONAL = "proportional"
    INDEPENDENT = "independent"


@dataclass
class LoraSliderState:
    """Full state and metadata for a single LoRA weight slider."""
    lora_name: str
    lora_path: str = ""
    category: str = "unknown"
    base_model: str = "unknown"
    computed_weight: float = 0.7
    adjusted_weight: Optional[float] = None
    final_weight: float = 0.7
    weight_rationale: str = ""
    signal_breakdown: Dict[str, Any] = field(default_factory=dict)
    conflict_flags: List[str] = field(default_factory=list)
    locked: bool = False
    enabled: bool = True
    min_weight: float = 0.1
    max_weight: float = 1.0
    sensitivity: float = 0.5
    is_primary: bool = False
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lora_name": self.lora_name,
            "lora_path": self.lora_path,
            "category": self.category,
            "base_model": self.base_model,
            "computed_weight": round(self.computed_weight, 3),
            "adjusted_weight": round(self.adjusted_weight, 3) if self.adjusted_weight is not None else None,
            "final_weight": round(self.final_weight, 3),
            "weight_rationale": self.weight_rationale,
            "signal_breakdown": self.signal_breakdown,
            "conflict_flags": self.conflict_flags,
            "locked": self.locked,
            "enabled": self.enabled,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "sensitivity": round(self.sensitivity, 3),
            "is_primary": self.is_primary,
            "rank": self.rank,
        }


@dataclass
class WeightBudget:
    """Current weight budget status for the LoRA stack."""
    total_weight: float = 0.0
    weight_cap: float = 2.5
    enabled_count: int = 0
    locked_count: int = 0
    status: str = "safe"  # safe | caution | danger
    headroom: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_weight": round(self.total_weight, 3),
            "weight_cap": round(self.weight_cap, 3),
            "enabled_count": self.enabled_count,
            "locked_count": self.locked_count,
            "status": self.status,
            "headroom": round(self.headroom, 3),
        }


class SliderSession:
    """Manages a set of per-LoRA sliders for a single generation request.

    A session is created from WeightEngine output, tracks user adjustments,
    coordinates rebalancing, and produces final weights for generation.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        weight_cap: float = 2.5,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self.weight_cap = weight_cap
        self.sliders: List[LoraSliderState] = []
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.rebalance_mode: RebalanceMode = RebalanceMode.PROPORTIONAL
        self._adjustment_history: List[Dict[str, Any]] = []

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        weight_assignments: List[Any],
        weight_cap: float = 2.5,
    ) -> "SliderSession":
        """Create a SliderSession from WeightEngine output (WeightAssignment list)."""
        session = cls(weight_cap=weight_cap)

        for wa in weight_assignments:
            breakdown_dict = {}
            if hasattr(wa, "signal_breakdown"):
                for sig_name, sc in wa.signal_breakdown.items():
                    breakdown_dict[sig_name] = {
                        "raw_value": getattr(sc, "raw_value", 0.5),
                        "alpha": getattr(sc, "alpha", 0.0),
                        "weighted": getattr(sc, "weighted", 0.0),
                        "available": getattr(sc, "available", True),
                        "reason": getattr(sc, "reason", ""),
                    }

            slider = LoraSliderState(
                lora_name=wa.lora_name,
                lora_path=getattr(wa, "lora_path", ""),
                category=getattr(wa, "category", "unknown"),
                base_model=getattr(wa, "base_model", "unknown"),
                computed_weight=wa.fused_weight,
                adjusted_weight=None,
                final_weight=wa.fused_weight,
                weight_rationale=getattr(wa, "rationale", ""),
                signal_breakdown=breakdown_dict,
                conflict_flags=getattr(wa, "conflict_flags", []),
                locked=False,
                enabled=True,
                min_weight=0.1,
                max_weight=1.0,
                sensitivity=getattr(wa, "sensitivity", 0.5),
                is_primary=getattr(wa, "is_primary", False),
                rank=getattr(wa, "rank", 0),
            )
            session.sliders.append(slider)

        session._recalculate_final_weights()
        logger.info("SliderSession %s created with %d sliders (cap=%.1f)",
                     session.session_id, len(session.sliders), weight_cap)
        return session

    # ── Slider Operations ──────────────────────────────────────────────

    def adjust(
        self,
        lora_name: str,
        new_weight: float,
        mode: Optional[str] = None,
    ) -> List[LoraSliderState]:
        """Adjust a single LoRA's weight and trigger rebalancing.

        Args:
            lora_name: Name of the LoRA to adjust.
            new_weight: New weight value (clamped to slider range).
            mode: Override rebalance mode for this adjustment.

        Returns:
            Updated list of all slider states.
        """
        slider = self._get_slider(lora_name)
        if slider is None:
            logger.warning("adjust: LoRA '%s' not found in session", lora_name)
            return self.sliders

        # Clamp to slider range
        new_weight = max(slider.min_weight, min(slider.max_weight, new_weight))
        old_weight = slider.final_weight
        slider.adjusted_weight = new_weight
        slider.final_weight = new_weight
        slider.locked = True  # Lock after manual adjustment

        effective_mode = RebalanceMode(mode) if mode else self.rebalance_mode

        # Record adjustment
        self._adjustment_history.append({
            "lora_name": lora_name,
            "old_weight": round(old_weight, 3),
            "new_weight": round(new_weight, 3),
            "delta": round(new_weight - old_weight, 3),
            "mode": effective_mode.value,
            "timestamp": time.time(),
        })

        # Rebalance other sliders if proportional mode
        if effective_mode == RebalanceMode.PROPORTIONAL:
            self._rebalance_proportional(lora_name, new_weight, old_weight)

        self._recalculate_final_weights()
        self.updated_at = time.time()

        logger.info("Session %s: adjusted '%s' %.2f -> %.2f (mode=%s)",
                     self.session_id, lora_name, old_weight, new_weight,
                     effective_mode.value)
        return self.sliders

    def toggle_lora(self, lora_name: str, enabled: bool) -> List[LoraSliderState]:
        """Enable or disable a LoRA in the stack."""
        slider = self._get_slider(lora_name)
        if slider is None:
            return self.sliders

        slider.enabled = enabled
        if not enabled:
            slider.final_weight = 0.0
        else:
            slider.final_weight = slider.adjusted_weight or slider.computed_weight

        self._recalculate_final_weights()
        self.updated_at = time.time()

        logger.info("Session %s: toggled '%s' enabled=%s",
                     self.session_id, lora_name, enabled)
        return self.sliders

    def lock_weight(self, lora_name: str) -> None:
        """Lock a LoRA's weight to prevent auto-rebalancing from changing it."""
        slider = self._get_slider(lora_name)
        if slider:
            slider.locked = True
            self.updated_at = time.time()

    def unlock_weight(self, lora_name: str) -> None:
        """Unlock a LoRA's weight for auto-rebalancing."""
        slider = self._get_slider(lora_name)
        if slider:
            slider.locked = False
            self.updated_at = time.time()

    def set_rebalance_mode(self, mode: str) -> None:
        """Set the default rebalance mode for the session."""
        self.rebalance_mode = RebalanceMode(mode)

    # ── Output ─────────────────────────────────────────────────────────

    def get_final_weights(self) -> Dict[str, float]:
        """Return {lora_name: final_weight} for all enabled LoRAs."""
        return {
            s.lora_name: round(s.final_weight, 3)
            for s in self.sliders
            if s.enabled and s.final_weight > 0
        }

    def get_budget_status(self) -> WeightBudget:
        """Compute current weight budget status."""
        enabled = [s for s in self.sliders if s.enabled]
        total = sum(s.final_weight for s in enabled)
        headroom = self.weight_cap - total
        locked = sum(1 for s in enabled if s.locked)

        if total <= self.weight_cap * 0.8:
            status = "safe"
        elif total <= self.weight_cap:
            status = "caution"
        else:
            status = "danger"

        return WeightBudget(
            total_weight=round(total, 3),
            weight_cap=self.weight_cap,
            enabled_count=len(enabled),
            locked_count=locked,
            status=status,
            headroom=round(headroom, 3),
        )

    def get_adjustment_history(self) -> List[Dict[str, Any]]:
        """Return the full adjustment history for this session."""
        return list(self._adjustment_history)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire session for API transport."""
        return {
            "session_id": self.session_id,
            "weight_cap": self.weight_cap,
            "rebalance_mode": self.rebalance_mode.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sliders": [s.to_dict() for s in self.sliders],
            "budget": self.get_budget_status().to_dict(),
            "adjustment_count": len(self._adjustment_history),
        }

    # ── Internal Helpers ───────────────────────────────────────────────

    def _get_slider(self, lora_name: str) -> Optional[LoraSliderState]:
        """Find a slider by LoRA name."""
        for s in self.sliders:
            if s.lora_name == lora_name:
                return s
        return None

    def _rebalance_proportional(
        self,
        adjusted_name: str,
        new_weight: float,
        old_weight: float,
    ) -> None:
        """Proportionally rebalance other sliders after an adjustment."""
        delta = new_weight - old_weight
        other_sliders = [
            s for s in self.sliders
            if s.lora_name != adjusted_name
            and s.enabled
            and not s.locked
        ]

        if not other_sliders:
            return

        current_total = sum(s.final_weight for s in other_sliders)
        new_total = sum(s.final_weight for s in self.sliders if s.enabled)

        if new_total > self.weight_cap and current_total > 0:
            # Over budget — reduce others proportionally
            excess = new_total - self.weight_cap
            for s in other_sliders:
                proportion = s.final_weight / current_total
                reduction = excess * proportion
                s.final_weight = round(max(s.min_weight, s.final_weight - reduction), 3)

        elif delta < 0 and current_total > 0:
            # User decreased — redistribute freed budget
            freed = abs(delta)
            for s in other_sliders:
                proportion = s.final_weight / current_total
                boost = freed * proportion * 0.5
                s.final_weight = round(min(s.max_weight, s.final_weight + boost), 3)

    def _recalculate_final_weights(self) -> None:
        """Ensure final_weight is consistent for all sliders."""
        for s in self.sliders:
            if not s.enabled:
                s.final_weight = 0.0
            elif s.adjusted_weight is not None:
                s.final_weight = s.adjusted_weight
            else:
                s.final_weight = s.computed_weight


# ── Self-Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    print("=" * 72)
    print("grokkie_weight_sliders — Phase-3 self-test")
    print("=" * 72)

    # Simulate WeightAssignment-like objects
    class MockWA:
        def __init__(self, name, weight, category="unknown", is_primary=False, rank=1):
            self.lora_name = name
            self.lora_path = f"/models/{name}.safetensors"
            self.category = category
            self.base_model = "sdxl"
            self.fused_weight = weight
            self.rationale = f"Test rationale for {name}"
            self.signal_breakdown = {}
            self.conflict_flags = []
            self.is_primary = is_primary
            self.rank = rank
            self.sensitivity = 0.5

    # Test 1: Session creation
    print("\n--- Test 1: Session Creation ---")
    assignments = [
        MockWA("silk_blouse_xl", 0.82, "garment_upper", True, 1),
        MockWA("denim_jeans_xl", 0.65, "garment_lower", False, 2),
        MockWA("detail_enhancer_xl", 0.50, "detail", False, 3),
        MockWA("cinematic_light_xl", 0.40, "lighting", False, 4),
    ]
    session = SliderSession.create(assignments, weight_cap=2.5)
    budget = session.get_budget_status()
    print(f"  Session ID: {session.session_id}")
    print(f"  Sliders: {len(session.sliders)}")
    print(f"  Budget: total={budget.total_weight:.3f}, cap={budget.weight_cap:.1f}, "
          f"status={budget.status}")
    for s in session.sliders:
        print(f"    {s.lora_name:30s} computed={s.computed_weight:.3f}  "
              f"final={s.final_weight:.3f}  locked={s.locked}")

    # Test 2: Proportional adjustment
    print("\n--- Test 2: Proportional Adjustment ---")
    session.adjust("silk_blouse_xl", 0.95, "proportional")
    budget = session.get_budget_status()
    print(f"  After adjusting silk_blouse_xl to 0.95:")
    print(f"  Budget: total={budget.total_weight:.3f}, status={budget.status}")
    for s in session.sliders:
        print(f"    {s.lora_name:30s} final={s.final_weight:.3f}  locked={s.locked}")

    # Test 3: Independent adjustment
    print("\n--- Test 3: Independent Adjustment ---")
    session.set_rebalance_mode("independent")
    session.adjust("denim_jeans_xl", 0.90, "independent")
    budget = session.get_budget_status()
    print(f"  After adjusting denim_jeans_xl to 0.90 (independent):")
    print(f"  Budget: total={budget.total_weight:.3f}, status={budget.status}")
    for s in session.sliders:
        print(f"    {s.lora_name:30s} final={s.final_weight:.3f}")

    # Test 4: Toggle LoRA
    print("\n--- Test 4: Toggle LoRA ---")
    session.toggle_lora("cinematic_light_xl", False)
    weights = session.get_final_weights()
    print(f"  Active LoRAs: {list(weights.keys())}")
    budget = session.get_budget_status()
    print(f"  Budget: total={budget.total_weight:.3f}, enabled={budget.enabled_count}")

    # Test 5: Lock/unlock
    print("\n--- Test 5: Lock/Unlock ---")
    session.unlock_weight("silk_blouse_xl")
    slider = session._get_slider("silk_blouse_xl")
    print(f"  silk_blouse_xl locked: {slider.locked}")
    session.lock_weight("silk_blouse_xl")
    print(f"  silk_blouse_xl locked: {slider.locked}")

    # Test 6: Serialization
    print("\n--- Test 6: Serialization ---")
    data = session.to_dict()
    print(f"  Keys: {list(data.keys())}")
    print(f"  Session ID: {data['session_id']}")
    print(f"  Slider count: {len(data['sliders'])}")
    print(f"  Budget status: {data['budget']['status']}")
    print(f"  Adjustment count: {data['adjustment_count']}")

    # Test 7: Adjustment history
    print("\n--- Test 7: Adjustment History ---")
    history = session.get_adjustment_history()
    for h in history:
        print(f"  {h['lora_name']}: {h['old_weight']:.3f} -> {h['new_weight']:.3f} "
              f"(delta={h['delta']:.3f}, mode={h['mode']})")

    # Test 8: Over-budget scenario
    print("\n--- Test 8: Over-Budget Scenario ---")
    session2 = SliderSession.create(assignments[:3], weight_cap=1.5)
    session2.adjust("silk_blouse_xl", 1.0, "proportional")
    session2.adjust("denim_jeans_xl", 0.9, "proportional")
    budget2 = session2.get_budget_status()
    print(f"  Budget: total={budget2.total_weight:.3f}, cap={budget2.weight_cap:.1f}, "
          f"status={budget2.status}, headroom={budget2.headroom:.3f}")
    for s in session2.sliders:
        if s.enabled:
            print(f"    {s.lora_name:30s} final={s.final_weight:.3f}")

    print("\n" + "=" * 72)
    print("All slider self-tests completed.")
    print("=" * 72)
