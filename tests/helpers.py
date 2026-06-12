"""Shared test fixtures/helpers - importable both as `tests.helpers`
(package-style runners) and as `helpers` (pytest rootdir-style sys.path
insertion on Mac/Linux/Colab/CI).  No cross-test-module imports."""
import json
import struct
from pathlib import Path

from lora_studio.lora_explorer import LoraCard, LoraInfluenceProfile


def _fake_safetensors(path: Path, meta: dict) -> None:
    """Minimal valid safetensors header (8-byte LE length + JSON)."""
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
