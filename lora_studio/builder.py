"""Phase 4: Dataset Composer - recipe-driven, versioned, reproducible builds.

A *recipe* ([recipes.NAME] in the project file) = filters + quotas + repeats
+ caption policy. A *build* materializes one recipe (or a multi-concept combo)
into DATASET/vNNN__name/ with a dataset.json snapshot and manifest rows.

Guarantees:
  - Deterministic: identical manifest state -> identical content hash.
  - Val split is stable per-frame (hash of frame_id), not random.
  - Non-destructive: hardlinks; sources and frames untouched.

CLI:
  lora-studio build --recipe character_v1
  lora-studio datasets
  lora-studio diff 2 3
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

from . import manifest
from .config import Project
from .packager import allocate_quota, parse_quota, round_robin_clusters
from .util import now_iso, safe_slug, setup_logging


@dataclass
class BuildConfig:
    output_base: Path
    recipe: str
    note: str = ""
    dataset_root: str = "DATASET"
    link_mode: str = "hardlink"


RECIPE_DEFAULTS = {
    "type": "character",
    "repeats": 10,
    "max_total": 0,
    "max_per_video": 40,
    "quota": "",
    "framing": "",
    "include_clusters": "",
    "exclude_clusters": "",
    "min_identity": 0.0,
    "min_sharpness": 0.0,
    "val_fraction": 0.0,
    "token": "",
    "class_word": "",
    "caption_static": "",
    "smart_crop": False,       # subject-centered crop to SDXL buckets
    "bucket_base": 1024,       # 1024 standard, 768 for lighter runs
    # Study Intelligence Layer filters (v7) - blank/0 = inactive
    "study_primary": "",       # e.g. figure_study_candidate
    "study_min_confidence": 0.0,
    "study_export_only": False,  # require study_export_eligible = 1
}


def resolve_concepts(prj: Project, name: str) -> list[tuple[str, dict]]:
    """A recipe either IS a concept, or lists other recipes via `concepts`."""
    recipes = prj.recipes or {}
    if name not in recipes:
        raise KeyError(
            f"Recipe '{name}' not found. Available: {', '.join(sorted(recipes)) or '(none)'}"
        )
    r = {**RECIPE_DEFAULTS, **recipes[name]}
    concept_list = str(r.get("concepts", "")).strip()
    if concept_list:
        out = []
        for cname in (c.strip() for c in concept_list.split(",") if c.strip()):
            if cname not in recipes:
                raise KeyError(f"Concept recipe '{cname}' (in '{name}') not found")
            sub = {**RECIPE_DEFAULTS, **recipes[cname]}
            # combo-level val_fraction cascades unless concept overrides
            if not sub.get("val_fraction") and r.get("val_fraction"):
                sub["val_fraction"] = r["val_fraction"]
            out.append((cname, sub))
        return out
    return [(name, r)]


def _csv_set(spec) -> set[str]:
    return {s.strip() for s in str(spec or "").split(",") if s.strip()}


def stable_split(frame_id: str, val_fraction: float) -> str:
    """Deterministic train/val assignment from the frame id itself."""
    if val_fraction <= 0:
        return "train"
    bucket = int(hashlib.sha1(frame_id.encode()).hexdigest()[:6], 16) % 10000
    return "val" if bucket < int(val_fraction * 10000) else "train"


def select_for_recipe(conn, rcp: dict) -> list:
    """Returns rows (frame_id, source_id, path, sharpness, cluster_id, framing,
    caption) after filters + quota/diversity selection. Deterministic."""
    rows = conn.execute(
        "SELECT f.frame_id, f.source_id, f.path, f.sharpness, f.cluster_id, "
        "d.framing, c.caption_text, d.identity_sim, d.bbox, "
        "s.study_primary, s.study_confidence, s.study_export_eligible "
        "FROM frames f "
        "LEFT JOIN detections d ON d.frame_id = f.frame_id "
        "LEFT JOIN captions c ON c.frame_id = f.frame_id "
        "LEFT JOIN study_labels s ON s.frame_id = f.frame_id "
        "WHERE f.status IN ('selected','packaged') "
        "ORDER BY f.frame_id"
    ).fetchall()

    framing_allow = _csv_set(rcp["framing"])
    inc_clusters = _csv_set(rcp["include_clusters"])
    exc_clusters = _csv_set(rcp["exclude_clusters"])
    min_id = float(rcp["min_identity"] or 0)
    min_sharp = float(rcp["min_sharpness"] or 0)

    study_primary = str(rcp.get("study_primary") or "")
    study_min_conf = float(rcp.get("study_min_confidence") or 0)
    study_export_only = bool(rcp.get("study_export_only"))

    filtered = []
    for r in rows:
        (frame_id, sid, path, sharp, cluster, framing, caption, ident, _bbox,
         s_primary, s_conf, s_export) = r
        r = r[:9]   # downstream selection logic uses the original 9 columns
        if study_primary and (s_primary or "") != study_primary:
            continue
        if study_min_conf > 0 and (s_conf is None or s_conf < study_min_conf):
            continue
        if study_export_only and not s_export:
            continue
        if framing_allow and (framing or "none") not in framing_allow:
            continue
        cstr = str(cluster) if cluster is not None else ""
        if inc_clusters and cstr not in inc_clusters:
            continue
        if exc_clusters and cstr and cstr in exc_clusters:
            continue
        if min_id > 0 and (ident is None or ident < min_id):
            continue
        if min_sharp > 0 and (sharp or 0) < min_sharp:
            continue
        filtered.append(r)

    has_clusters = any(r[4] is not None for r in filtered)
    has_framing = any(r[5] for r in filtered)

    # per-video cap with cluster diversity
    per_source: dict[str, list] = defaultdict(list)
    for r in filtered:
        per_source[r[1]].append(r)
    capped: list = []
    max_pv = int(rcp["max_per_video"] or 0)
    for sid in sorted(per_source):
        frames = sorted(per_source[sid], key=lambda r: (-(r[3] or 0.0), r[0]))
        cap = max_pv if max_pv > 0 else len(frames)
        capped.extend(round_robin_clusters(frames, cap) if has_clusters
                      else frames[:cap])

    max_total = int(rcp["max_total"] or 0)
    quota = parse_quota(str(rcp["quota"])) if (rcp["quota"] and has_framing) else {}
    if quota and max_total > 0:
        buckets: dict[str, list] = defaultdict(list)
        for r in capped:
            buckets[r[5] or "none"].append(r)
        alloc = allocate_quota({b: len(fs) for b, fs in buckets.items()},
                               min(max_total, len(capped)), quota)
        chosen = []
        for b in sorted(alloc):
            frames = sorted(buckets.get(b, []), key=lambda r: (-(r[3] or 0.0), r[0]))
            n = alloc[b]
            chosen.extend(round_robin_clusters(frames, n) if has_clusters
                          else frames[:n])
    elif max_total > 0:
        capped.sort(key=lambda r: (-(r[3] or 0.0), r[0]))
        chosen = capped[:max_total]
    else:
        chosen = capped

    chosen.sort(key=lambda r: r[0])  # deterministic order for hashing
    return chosen


def content_hash(records: list[dict], concepts: list[tuple[str, dict]]) -> str:
    """Hash of everything that defines the dataset content (not time/version)."""
    payload = {
        "records": [
            {k: rec[k] for k in ("frame_id", "concept", "split", "caption", "name")}
            for rec in records
        ],
        "concepts": [
            {"name": n, "recipe": {k: rcp[k] for k in sorted(RECIPE_DEFAULTS)}}
            for n, rcp in concepts
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def _place(src: Path, dest: Path, link_mode: str) -> None:
    if dest.exists():
        return
    if link_mode == "hardlink":
        try:
            os.link(src, dest)
            return
        except OSError:
            pass
    shutil.copy2(src, dest)


def build_generator(prj: Project, cfg: BuildConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)

    try:
        concepts = resolve_concepts(prj, cfg.recipe)
    except KeyError as exc:
        yield f"FATAL: {exc}"
        return

    # assemble records
    records: list[dict] = []
    for cname, rcp in concepts:
        token = str(rcp["token"] or prj.trigger_token)
        class_word = str(rcp["class_word"] or prj.class_word)
        static = str(rcp["caption_static"] or f"{token} {class_word}".strip())
        chosen = select_for_recipe(conn, rcp)
        for frame_id, sid, path, _sh, _cl, _fr, caption, _id, bbox in chosen:
            records.append({
                "frame_id": frame_id, "concept": cname,
                "path": path, "name": f"{sid}_{Path(path).name}",
                "caption": caption or static,
                "split": stable_split(frame_id, float(rcp["val_fraction"] or 0)),
                "folder": f"{int(rcp['repeats'])}_{safe_slug(token)} {safe_slug(class_word)}".strip(),
                "bbox": bbox,
                "smart_crop": bool(rcp["smart_crop"]),
                "bucket_base": int(rcp["bucket_base"] or 1024),
            })
        yield f"Concept '{cname}': {len(chosen)} frames selected"

    if not records:
        yield "FATAL: no frames matched the recipe filters. Run curate first or relax filters."
        return

    chash = content_hash(records, concepts)

    # already built identically?
    prior = conn.execute(
        "SELECT version, dir FROM datasets WHERE content_hash = ?", (chash,)
    ).fetchone()
    if prior:
        yield (f"\nIdentical dataset already exists: v{prior[0]:03d} ({prior[1]})\n"
               f"Content hash {chash} - nothing new to build.")
        return

    row = conn.execute("SELECT COALESCE(MAX(version),0) FROM datasets").fetchone()
    version = int(row[0]) + 1
    ds_dir = cfg.output_base / cfg.dataset_root / f"v{version:03d}__{safe_slug(cfg.recipe)}"

    n_train = n_val = errors = 0
    for rec in records:
        src = Path(rec["path"])
        if not src.exists():
            errors += 1
            continue
        sub = "img" if rec["split"] == "train" else "val_img"
        dest_dir = ds_dir / sub / rec["folder"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / rec["name"]
        try:
            if rec.get("smart_crop"):
                from .crop import smart_crop_image
                if not dest.exists():
                    smart_crop_image(src, dest, rec.get("bbox"),
                                     base=rec.get("bucket_base", 1024))
            else:
                _place(src, dest, cfg.link_mode)
            dest.with_suffix(".txt").write_text(rec["caption"], encoding="utf-8")
            if rec["split"] == "train":
                n_train += 1
            else:
                n_val += 1
        except Exception as exc:
            errors += 1
            logging.error("build place failed %s: %s", src, exc)

    # snapshot
    snapshot = {
        "version": version,
        "recipe": cfg.recipe,
        "concepts": {n: r for n, r in concepts},
        "content_hash": chash,
        "built_at": now_iso(),
        "note": cfg.note,
        "counts": {"train": n_train, "val": n_val, "errors": errors},
        "frames": [
            {k: rec[k] for k in ("frame_id", "concept", "split", "name", "caption")}
            for rec in records
        ],
    }
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "dataset.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    conn.execute(
        "INSERT INTO datasets(version, recipe_name, recipe_json, content_hash, "
        "built_at, image_count, val_count, dir, note) VALUES (?,?,?,?,?,?,?,?,?)",
        (version, cfg.recipe, json.dumps({n: r for n, r in concepts}),
         chash, now_iso(), n_train, n_val, str(ds_dir), cfg.note),
    )
    conn.executemany(
        "INSERT OR REPLACE INTO dataset_frames(version, frame_id, concept, split, caption) "
        "VALUES (?,?,?,?,?)",
        [(version, r["frame_id"], r["concept"], r["split"], r["caption"])
         for r in records],
    )
    conn.commit()

    yield (
        f"\nBUILD v{version:03d} DONE.\n"
        f"Recipe: {cfg.recipe} ({len(concepts)} concept(s))\n"
        f"Train: {n_train} | Val: {n_val} | Errors: {errors}\n"
        f"Content hash: {chash}\n"
        f"Dir: {ds_dir}\n"
        f"Snapshot: {ds_dir / 'dataset.json'}\n"
        f"\nkohya 'Image folder': {ds_dir / 'img'}"
    )


# -----------------------------
# Listing & diff
# -----------------------------

def list_datasets(conn) -> list[dict]:
    return [
        {"version": r[0], "recipe": r[1], "hash": r[2], "built_at": r[3],
         "train": r[4], "val": r[5], "dir": r[6], "note": r[7] or ""}
        for r in conn.execute(
            "SELECT version, recipe_name, content_hash, built_at, image_count, "
            "val_count, dir, note FROM datasets ORDER BY version"
        )
    ]


def diff_datasets(conn, va: int, vb: int) -> dict:
    def load(v: int) -> dict[tuple, dict]:
        rows = conn.execute(
            "SELECT frame_id, concept, split, caption FROM dataset_frames "
            "WHERE version = ?", (v,),
        ).fetchall()
        if not rows and not conn.execute(
            "SELECT 1 FROM datasets WHERE version=?", (v,)
        ).fetchone():
            raise KeyError(f"dataset v{v} not found")
        return {(r[0], r[1]): {"split": r[2], "caption": r[3]} for r in rows}

    A, B = load(va), load(vb)
    added = sorted(k for k in B if k not in A)
    removed = sorted(k for k in A if k not in B)
    common = [k for k in A if k in B]
    caption_changed = sorted(k for k in common if A[k]["caption"] != B[k]["caption"])
    split_changed = sorted(k for k in common if A[k]["split"] != B[k]["split"])

    def fmix(d: dict) -> dict:
        mix: dict[str, int] = defaultdict(int)
        for (fid, concept) in d:
            mix[concept] += 1
        return dict(mix)

    return {
        "a": va, "b": vb,
        "added": [list(k) for k in added],
        "removed": [list(k) for k in removed],
        "caption_changed": [list(k) for k in caption_changed],
        "split_changed": [list(k) for k in split_changed],
        "concept_mix_a": fmix(A), "concept_mix_b": fmix(B),
        "summary": (f"v{va} -> v{vb}: +{len(added)} added, -{len(removed)} removed, "
                    f"{len(caption_changed)} captions changed, "
                    f"{len(split_changed)} split changes"),
    }
