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
            _SWAPPER["app"] = FaceAnalysis(name="buffalo_l")
            _SWAPPER["app"].prepare(ctx_id=0, det_size=(640, 640))
        faces = _SWAPPER["app"].get(cv2.imread(str(image_path)))
        if not faces:
            return None
        kps = faces[0].kps          # [l_eye, r_eye, nose, l_mouth, r_mouth]
        eyes_y = (kps[0][1] + kps[1][1]) / 2
        mouth_y = (kps[3][1] + kps[4][1]) / 2
        return 180 if eyes_y > mouth_y else 0
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
