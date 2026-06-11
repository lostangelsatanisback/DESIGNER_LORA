"""Unit tests for Phase 2/3 pure logic: k-means, quota allocation,
cluster round-robin, caption rules engine. No model downloads required."""

import sys
from pathlib import Path

import pytest  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_studio.caption.wd14 import build_caption, parse_rules, PONY_PREFIX  # noqa: E402
from lora_studio.config import CaptionConfig  # noqa: E402
from lora_studio.curate.diversity import auto_k, kmeans  # noqa: E402
from lora_studio.packager import (  # noqa: E402
    allocate_quota, parse_quota, round_robin_clusters,
)


def test_kmeans_separates_clusters():
    a = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]]
    b = [[9.0, 9.0], [9.1, 9.0], [9.0, 9.1], [9.1, 9.1]]
    labels = kmeans(a + b, 2, max_iter=20)
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[4]


def test_auto_k_bounds():
    assert auto_k(4) == 2
    assert 2 <= auto_k(500) <= 40
    assert auto_k(1_000_000) == 40


def test_parse_quota_normalizes():
    q = parse_quota("closeup=3,portrait=1")
    assert abs(q["closeup"] - 0.75) < 1e-9
    assert abs(sum(q.values()) - 1.0) < 1e-9
    assert parse_quota("") == {}


def test_allocate_quota_redistributes():
    avail = {"closeup": 2, "portrait": 50, "upper_body": 50, "full_body": 0, "none": 10}
    quota = {"closeup": 0.30, "portrait": 0.30, "upper_body": 0.25, "full_body": 0.15}
    alloc = allocate_quota(avail, 40, quota)
    assert sum(alloc.values()) == 40
    assert alloc["closeup"] == 2                  # capped by availability
    assert alloc.get("full_body", 0) == 0         # nothing available
    for b, n in alloc.items():
        assert n <= avail[b]                       # never exceeds availability


def test_round_robin_clusters_diversity():
    # rows: (frame_id, source_id, path, sharpness, cluster_id)
    rows = (
        [(f"a{i}", "s", "p", 100 - i, 0) for i in range(10)]
        + [(f"b{i}", "s", "p", 50 - i, 1) for i in range(10)]
        + [(f"c{i}", "s", "p", 10 - i, 2) for i in range(10)]
    )
    picked = round_robin_clusters(rows, 6)
    clusters = [r[4] for r in picked]
    assert len(picked) == 6
    assert set(clusters) == {0, 1, 2}              # all clusters represented
    assert clusters.count(0) == 2                  # evenly spread


def _cfg(**kw) -> CaptionConfig:
    base = dict(output_base=Path("/tmp"), trigger="spook", class_word="person")
    base.update(kw)
    return CaptionConfig(**base)


def test_caption_trigger_first():
    cap = build_caption([("smile", 0.9), ("outdoors", 0.5)], _cfg())
    assert cap.startswith("spook person, ")
    assert "smile" in cap and "outdoors" in cap


def test_caption_rules_blacklist_remap_prune():
    cfg = _cfg(blacklist="watermark", remap="1girl:woman", prune="blue_eyes")
    rules = parse_rules(cfg)
    cap = build_caption(
        [("1girl", 0.99), ("blue eyes", 0.95), ("watermark", 0.9), ("smile", 0.8)],
        cfg, rules,
    )
    assert "woman" in cap            # remapped
    assert "1girl" not in cap
    assert "blue eyes" not in cap    # pruned (permanent trait -> binds to trigger)
    assert "watermark" not in cap    # blacklisted
    assert "smile" in cap


def test_caption_pony_prefix_and_max_tags():
    cfg = _cfg(pony_prefix=True, max_tags=2)
    cap = build_caption([("a", 0.9), ("b", 0.8), ("c", 0.7)], cfg)
    parts = cap.split(", ")
    assert parts[:3] == PONY_PREFIX
    assert "c" not in parts          # truncated at max_tags


def test_caption_dedupe_after_remap():
    cfg = _cfg(remap="lying:on_back")
    cap = build_caption([("lying", 0.9), ("on back", 0.85)], cfg)
    assert cap.count("on back") == 1
