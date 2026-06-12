"""Dataset insights - health report, caption lint, cluster naming,
orientation check.  Pure stdlib SQL over the manifest; pre-build QA."""
from __future__ import annotations

import json
from collections import Counter


def dataset_health(conn) -> dict:
    """One score (0-100) with the reasons it isn't 100."""
    score, reasons = 100, []
    sel = conn.execute("SELECT COUNT(*) FROM frames "
                       "WHERE status IN ('selected','packaged')").fetchone()[0]
    if sel == 0:
        return {"score": 0, "selected": 0,
                "reasons": ["no curated frames - run curate first"]}
    # framing balance
    fm = dict(conn.execute(
        "SELECT d.framing, COUNT(*) FROM frames f JOIN detections d "
        "ON d.frame_id=f.frame_id WHERE f.status IN ('selected','packaged') "
        "AND d.framing IS NOT NULL GROUP BY d.framing"))
    fb = fm.get("full_body", 0) / max(1, sum(fm.values()))
    if fb < 0.08:
        score -= 15; reasons.append(
            f"full-body coverage thin ({fb:.0%}) - pose flexibility suffers")
    # identity quality
    row = conn.execute(
        "SELECT AVG(identity_sim), COUNT(*) FROM detections d JOIN frames f "
        "ON f.frame_id=d.frame_id WHERE f.status IN ('selected','packaged') "
        "AND identity_sim IS NOT NULL").fetchone()
    if row[0] and row[0] < 0.45:
        score -= 15; reasons.append(
            f"mean identity similarity low ({row[0]:.2f})")
    # caption coverage
    cap = conn.execute(
        "SELECT COUNT(*) FROM frames f JOIN captions c ON "
        "c.frame_id=f.frame_id WHERE f.status IN ('selected','packaged') "
        "AND c.caption_text != ''").fetchone()[0]
    cov = cap / sel
    if cov < 0.95:
        score -= 20; reasons.append(f"caption coverage {cov:.0%} - "
                                    "finish captioning before build")
    # source dominance (temporal bias)
    dom = conn.execute(
        "SELECT source_id, COUNT(*) c FROM frames WHERE status IN "
        "('selected','packaged') GROUP BY source_id ORDER BY c DESC "
        "LIMIT 1").fetchone()
    if dom and dom[1] / sel > 0.25:
        score -= 10; reasons.append(
            f"one source dominates ({dom[1]}/{sel} frames) - "
            "max_per_video in the recipe will cap it at build time")
    # multi-person contamination
    multi = conn.execute(
        "SELECT COUNT(*) FROM frames f JOIN captions c ON "
        "c.frame_id=f.frame_id WHERE f.status IN ('selected','packaged') "
        "AND c.caption_text LIKE '%1boy%'").fetchone()[0]
    if multi / sel > 0.15:
        score -= 10; reasons.append(
            f"{multi} frames include a second person - review before build")
    return {"score": max(0, score), "selected": sel,
            "framing_mix": fm, "caption_coverage": round(cov, 2),
            "reasons": reasons or ["dataset is build-ready"]}


def caption_lint(conn, limit: int = 200) -> list[dict]:
    """Frames whose tags contradict detections."""
    out = []
    for fid, cap, faces in conn.execute(
            "SELECT f.frame_id, c.caption_text, d.face_count FROM frames f "
            "JOIN captions c ON c.frame_id=f.frame_id "
            "JOIN detections d ON d.frame_id=f.frame_id "
            "WHERE f.status IN ('selected','packaged')"):
        cap = cap or ""
        issues = []
        if "solo" in cap and (faces or 0) >= 2:
            issues.append("tagged solo but 2+ faces detected")
        if "no_humans" in cap and (faces or 0) >= 1:
            issues.append("tagged no_humans but a face was detected")
        if issues:
            out.append({"frame_id": fid, "issues": issues})
            if len(out) >= limit:
                break
    return out


def name_clusters(conn, top_n: int = 4) -> dict:
    """Auto-name clusters from their most distinctive frequent tags;
    stored in meta as cluster_names for the Review UI."""
    rows = conn.execute(
        "SELECT f.cluster_id, c.caption_text FROM frames f JOIN captions c "
        "ON c.frame_id=f.frame_id WHERE f.status IN ('selected','packaged') "
        "AND f.cluster_id IS NOT NULL").fetchall()
    global_c, per = Counter(), {}
    for cid, cap in rows:
        toks = [t.strip() for t in (cap or "").split(",")
                if t.strip() and "score_" not in t]
        per.setdefault(cid, Counter()).update(toks)
        global_c.update(toks)
    names = {}
    for cid, cnt in per.items():
        total = sum(cnt.values()) or 1
        # distinctiveness: local share vs global share
        ranked = sorted(cnt.items(), key=lambda kv: -(
            kv[1] / total - global_c[kv[0]] / max(1, sum(global_c.values()))))
        names[str(cid)] = ", ".join(t for t, _ in ranked[:top_n])
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES "
                 "('cluster_names', ?)", (json.dumps(names),))
    conn.commit()
    return names


def check_orientation(image_path):
    """Upside-down face check via landmarks (eyes vs mouth).  Returns
    suggested rotation in degrees (0/180) or None when undetectable."""
    try:
        import cv2
        from insightface.app import FaceAnalysis
        from .identity_integration import _SWAPPER
        if _SWAPPER["app"] is None:
            _SWAPPER["app"] = FaceAnalysis(name="buffalo_l", providers=_rt_providers())
            _SWAPPER["app"].prepare(ctx_id=0, det_size=(640, 640))
        faces = _SWAPPER["app"].get(cv2.imread(str(image_path)))
        if not faces:
            return None
        from .curate.smart import _upright_rotation_from_kps
        return _upright_rotation_from_kps(faces[0].kps)
    except Exception:
        return None


def global_search(conn, prj, q: str, limit: int = 6) -> dict:
    """One query across frames, LoRAs, presets, and variation batches."""
    like = f"%{q}%"
    out = {"frames": [], "loras": [], "presets": [], "batches": []}
    try:
        out["frames"] = [
            {"frame_id": r[0], "caption": (r[1] or "")[:90]}
            for r in conn.execute(
                "SELECT f.frame_id, c.caption_text FROM frames f JOIN "
                "captions c ON c.frame_id=f.frame_id WHERE f.status IN "
                "('selected','packaged') AND c.caption_text LIKE ? LIMIT ?",
                (like, limit))]
        out["presets"] = [r[0] for r in conn.execute(
            "SELECT name FROM concept_control_presets WHERE name LIKE ? "
            "AND kind != 'stack_history' LIMIT ?", (like, limit))]
        out["batches"] = [r[0] for r in conn.execute(
            "SELECT batch_id FROM variation_batches WHERE batch_id LIKE ? "
            "LIMIT ?", (like, limit))]
    except Exception:
        pass
    try:
        from .lora_explorer import filter_cards, scan_loras_cached
        out["loras"] = [c.lora_id for c in filter_cards(
            scan_loras_cached(prj), search=q)][:limit]
    except Exception:
        pass
    return out


def log_stack_history(conn, summary: dict, keep: int = 100) -> None:
    """Append a resolved-stack snapshot to the timeline (kind
    stack_history in the presets store), pruned to the newest `keep`."""
    import time
    import uuid
    from .concept_control import save_preset
    save_preset(conn, f"hist_{time.strftime('%Y%m%d_%H%M%S')}_"
                      f"{uuid.uuid4().hex[:6]}",
                "stack_history", summary)
    conn.execute(
        "DELETE FROM concept_control_presets WHERE kind='stack_history' "
        "AND name NOT IN (SELECT name FROM concept_control_presets WHERE "
        "kind='stack_history' ORDER BY name DESC LIMIT ?)", (keep,))
    conn.commit()


def _hamming(a: str, b: str) -> int:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except Exception:
        return 64


def aesthetic_proxy(row: dict) -> float:
    """Lightweight aesthetic score (0-1) from technical signals: sharpness,
    exposure band, framing readability.  Registered as a study SIGNAL_HOOK
    and used as the duplicate-collapse tiebreak.  No ML required."""
    sharp = min(float(row.get("sharpness") or 0), 150.0) / 150.0
    b = row.get("brightness")
    expo = 1.0 - min(abs((float(b) if b is not None else 110) - 110) / 110,
                     1.0)
    framing = {"full_body": 1.0, "upper_body": 0.9, "portrait": 0.8,
               "closeup": 0.7}.get(row.get("framing") or "", 0.5)
    return round(0.5 * sharp + 0.3 * expo + 0.2 * framing, 3)


try:                                    # register as an optional probe
    from .study import SIGNAL_HOOKS as _SH
    _SH.setdefault("aesthetic_proxy",
                   lambda row: {"aesthetic_score": aesthetic_proxy(row)})
except Exception:
    pass


def collapse_near_duplicates(conn, max_hamming: int = 6,
                             apply: bool = False) -> dict:
    """Duplicate-pose collapse: within each source, consecutive frames
    whose dHash differs by <= max_hamming bits are near-identical poses -
    keep the highest aesthetic score, reject the rest (reversible:
    status rejected_neardup)."""
    rows = conn.execute(
        "SELECT f.frame_id, f.source_id, f.path, f.dhash, f.sharpness, "
        "f.brightness, d.framing FROM frames f LEFT JOIN detections d ON "
        "d.frame_id=f.frame_id WHERE f.status IN ('selected','packaged') "
        "AND f.dhash IS NOT NULL ORDER BY f.source_id, f.path").fetchall()
    rejected, kept_groups = [], 0
    group: list = []

    def flush():
        nonlocal kept_groups
        if len(group) > 1:
            kept_groups += 1
            best = max(group, key=lambda r: aesthetic_proxy(
                {"sharpness": r[4], "brightness": r[5], "framing": r[6]}))
            rejected.extend(r[0] for r in group if r[0] != best[0])
        group.clear()

    prev = None
    for r in rows:
        if (prev is not None and r[1] == prev[1]
                and _hamming(r[3], prev[3]) <= max_hamming):
            if not group:
                group.append(prev)
            group.append(r)
        else:
            flush()
        prev = r
    flush()
    if apply and rejected:
        conn.executemany(
            "UPDATE frames SET status='rejected_neardup' WHERE frame_id=?",
            [(fid,) for fid in rejected])
        conn.commit()
    return {"groups": kept_groups, "would_reject": len(rejected),
            "applied": bool(apply and rejected)}


def fix_rotations(conn, prj, apply: bool = False,
                  limit: int = 0):
    """Orientation pass over curated frames: detect upside-down faces via
    landmarks and rotate the extracted frame file 180 degrees in place
    (source media untouched).  Yields progress; resumable by nature
    (correct frames are no-ops)."""
    rows = conn.execute(
        "SELECT f.frame_id, f.path FROM frames f JOIN detections d ON "
        "d.frame_id=f.frame_id WHERE f.status IN ('selected','packaged') "
        "AND d.face_count >= 1").fetchall()
    if limit:
        rows = rows[:limit]
    yield f"Orientation check: {len(rows)} frames"
    flipped = 0
    for i, (fid, path) in enumerate(rows, 1):
        rot = check_orientation(path)
        if rot:
            flipped += 1
            if apply:
                try:
                    from PIL import Image
                    img = Image.open(path).rotate(rot, expand=True)
                    img.save(path)
                    yield f"  rotated {fid} by {rot} deg"
                except Exception as exc:
                    yield f"  {fid}: rotate failed ({exc})"
            else:
                yield f"  {fid}: needs {rot} deg rotation (dry run)"
        if i % 500 == 0:
            yield f"  {i}/{len(rows)} checked..."
    yield (f"Done: {flipped} upside-down frame(s) "
           f"{'rotated' if apply else 'found (use --apply to fix)'}")


def _rt_providers():
    from .runtime import onnx_providers
    return onnx_providers()
