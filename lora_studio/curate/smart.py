"""Smart curation: face detection + identity filtering (Phase 2, first slice).

Backend: InsightFace 'buffalo_l' bundle —
  - SCRFD-10G face detector
  - ArcFace R50 (w600k) identity embedding
running on onnxruntime with CoreML EP on Apple Silicon, CPU fallback.

Install:  pip install 'lora-studio[ai]'   (insightface, onnxruntime, opencv, numpy)
Models auto-download to ~/.insightface/models/buffalo_l on first run.

Pattern credit: framing thresholds adapted from the focal-point logic in
reForge modules/textual_inversion/autocrop.py (face-weighted POI), upgraded
from YuNet/Haar to SCRFD + identity embeddings.

Workflow:
  1. lora-studio anchor --anchor-dir /path/to/15_reference_images
  2. lora-studio curate --smart
Frames already 'selected' by basic curation are scanned; frames with no
subject, tiny faces, or wrong identity are demoted with explicit statuses.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Optional

from .. import manifest
from ..config import SmartCurateConfig
from ..util import now_iso, setup_logging

_BACKEND = None
_BACKEND_ERR: Optional[str] = None


def _load_backend(det_size: int = 640):
    """Lazy global InsightFace app. Returns (app, numpy, cv2_or_none) or raises."""
    global _BACKEND, _BACKEND_ERR
    if _BACKEND is not None:
        return _BACKEND
    if _BACKEND_ERR is not None:
        raise RuntimeError(_BACKEND_ERR)
    try:
        import numpy as np
        from insightface.app import FaceAnalysis
        try:
            import cv2
        except Exception:
            cv2 = None
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0, det_size=(det_size, det_size))
        _BACKEND = (app, np, cv2)
        return _BACKEND
    except Exception as exc:
        _BACKEND_ERR = (
            f"Smart curation unavailable: {exc}\n"
            "Install extras:  pip install 'lora-studio[ai]'"
        )
        raise RuntimeError(_BACKEND_ERR)


def smart_available() -> tuple[bool, str]:
    try:
        import numpy  # noqa: F401
        import insightface  # noqa: F401
        import onnxruntime  # noqa: F401
        return True, "ok"
    except Exception as exc:
        return False, f"missing [ai] extras: {exc}"


def _read_bgr(path: Path, np, cv2):
    if cv2 is not None:
        img = cv2.imread(str(path))
        if img is not None:
            return img
    try:
        from PIL import Image
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        return rgb[:, :, ::-1].copy()  # RGB -> BGR
    except Exception:
        return None


def _classify_framing(face_h_ratio: float) -> str:
    if face_h_ratio >= 0.40:
        return "closeup"
    if face_h_ratio >= 0.25:
        return "portrait"
    if face_h_ratio >= 0.12:
        return "upper_body"
    return "full_body"


# -----------------------------
# Anchor (subject identity)
# -----------------------------

def build_anchor(cfg: SmartCurateConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    if not cfg.anchor_dir or not Path(cfg.anchor_dir).exists():
        yield "FATAL: --anchor-dir must point to a folder of subject reference images."
        return
    app, np, cv2 = _load_backend(cfg.det_size)
    conn = manifest.connect(cfg.output_base)

    embeds = []
    refs = sorted(
        p for p in Path(cfg.anchor_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    yield f"Building identity anchor from {len(refs)} reference images..."
    for ref in refs:
        img = _read_bgr(ref, np, cv2)
        if img is None:
            yield f"  skip (unreadable): {ref.name}"
            continue
        faces = app.get(img)
        if not faces:
            yield f"  skip (no face): {ref.name}"
            continue
        # largest face wins
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        embeds.append(face.normed_embedding)
        yield f"  ok: {ref.name} (det {face.det_score:.2f})"

    if len(embeds) < 3:
        yield f"FATAL: only {len(embeds)} usable references; need at least 3 (5-15 ideal)."
        return

    anchor = np.mean(np.stack(embeds), axis=0)
    anchor = anchor / np.linalg.norm(anchor)
    manifest.meta_set(conn, "identity_anchor", anchor.tolist())
    manifest.meta_set(conn, "identity_anchor_info", {
        "built_at": now_iso(), "refs_used": len(embeds),
        "anchor_dir": str(cfg.anchor_dir),
    })
    # self-consistency: how tight is the reference set?
    sims = [float(np.dot(anchor, e)) for e in embeds]
    yield (
        f"\nANCHOR BUILT from {len(embeds)} refs.\n"
        f"Self-similarity: min {min(sims):.3f} / mean {sum(sims)/len(sims):.3f}\n"
        f"(threshold {cfg.identity_threshold} should sit comfortably below the min)\n"
        "Next: lora-studio curate --smart"
    )


# -----------------------------
# Smart pass over selected frames
# -----------------------------

def _upright_rotation_from_kps(kps) -> int:
    """CCW degrees needed to make the face upright (0/90/180/270) from
    the eyes->mouth axis.  0 means already upright."""
    ex, ey = (kps[0][0] + kps[1][0]) / 2, (kps[0][1] + kps[1][1]) / 2
    mx, my = (kps[3][0] + kps[4][0]) / 2, (kps[3][1] + kps[4][1]) / 2
    dx, dy = mx - ex, my - ey
    if abs(dy) >= abs(dx):
        return 0 if dy > 0 else 180
    # mouth left of eyes -> face's down points left -> rotate 270 CCW
    return 270 if dx < 0 else 90


def _rotate_bgr(img, ccw_degrees: int, cv2):
    code = {90: cv2.ROTATE_90_COUNTERCLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_CLOCKWISE}[ccw_degrees]
    return cv2.rotate(img, code)


def smart_curate_generator(cfg: SmartCurateConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    ok, reason = smart_available()
    if not ok:
        yield f"Smart curation skipped: {reason}"
        return
    app, np, cv2 = _load_backend(cfg.det_size)
    conn = manifest.connect(cfg.output_base)

    anchor_list = manifest.meta_get(conn, "identity_anchor")
    anchor = np.asarray(anchor_list, dtype=np.float32) if anchor_list else None
    if anchor is None:
        yield ("No identity anchor found - running face-presence filtering only.\n"
               "Build one with: lora-studio anchor --anchor-dir <refs>  "
               "to also reject wrong-identity frames.\n")

    if cfg.rescan:
        rows = conn.execute(
            "SELECT f.frame_id, f.path FROM frames f "
            "WHERE f.status IN ('selected','rejected_noface','rejected_smallface',"
            "'rejected_identity') ORDER BY f.path"
        ).fetchall()
        conn.executemany(
            "UPDATE frames SET status='selected' WHERE frame_id=?",
            [(r[0],) for r in rows],
        )
        conn.commit()
    else:
        rows = conn.execute(
            "SELECT f.frame_id, f.path FROM frames f "
            "LEFT JOIN detections d ON d.frame_id = f.frame_id "
            "WHERE f.status='selected' AND d.frame_id IS NULL ORDER BY f.path"
        ).fetchall()

    if not rows:
        yield "Nothing to scan (no selected frames without detections). Use --rescan to redo."
        return

    yield (f"Smart scan: {len(rows)} frames | backend buffalo_l "
           f"(SCRFD det + ArcFace id) | anchor: {'yes' if anchor is not None else 'no'}\n")

    n_noface = n_small = n_wrongid = n_kept = 0
    t0 = time.time()

    for i, (frame_id, path_str) in enumerate(rows, start=1):
        img = _read_bgr(Path(path_str), np, cv2)
        if img is None:
            conn.execute(
                "UPDATE frames SET status='decode_failed' WHERE frame_id=?", (frame_id,)
            )
            continue
        ih, iw = img.shape[:2]
        faces = app.get(img)

        # orientation correction: landmark check on detected faces, and
        # 90/180/270 rescue when the detector finds nothing (SCRFD is
        # rotation-sensitive - sideways faces read as no-face)
        rotated_by = 0
        if getattr(cfg, "auto_rotate", True):
            if faces:
                rot = _upright_rotation_from_kps(
                    max(faces, key=lambda f: f.det_score).kps)
                if rot:
                    img = _rotate_bgr(img, rot, cv2)
                    new_faces = app.get(img)
                    if new_faces:
                        faces, rotated_by = new_faces, rot
                        ih, iw = img.shape[:2]
            else:
                for rot in (90, 180, 270):
                    cand = _rotate_bgr(img, rot, cv2)
                    new_faces = app.get(cand)
                    if new_faces and not _upright_rotation_from_kps(
                            max(new_faces, key=lambda f: f.det_score).kps):
                        img, faces, rotated_by = cand, new_faces, rot
                        ih, iw = img.shape[:2]
                        break
        if rotated_by:
            try:
                cv2.imwrite(path_str, img)
            except Exception:
                rotated_by = 0

        if not faces:
            conn.execute(
                "INSERT OR REPLACE INTO detections(frame_id, face_count, face_area, "
                "det_conf, identity_sim, framing, detected_at) VALUES (?,0,0,0,NULL,'none',?)",
                (frame_id, now_iso()),
            )
            conn.execute(
                "UPDATE frames SET status='rejected_noface' WHERE frame_id=?", (frame_id,)
            )
            n_noface += 1
        else:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            bw = float(face.bbox[2] - face.bbox[0])
            bh = float(face.bbox[3] - face.bbox[1])
            area_ratio = (bw * bh) / float(iw * ih)
            framing = _classify_framing(bh / ih)
            sim = float(np.dot(anchor, face.normed_embedding)) if anchor is not None else None

            import json as _json
            bbox_norm = [
                max(0.0, float(face.bbox[0]) / iw), max(0.0, float(face.bbox[1]) / ih),
                min(1.0, float(face.bbox[2]) / iw), min(1.0, float(face.bbox[3]) / ih),
            ]
            conn.execute(
                "INSERT OR REPLACE INTO detections(frame_id, face_count, face_area, "
                "det_conf, identity_sim, framing, embed, detected_at, bbox) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (frame_id, len(faces), area_ratio, float(face.det_score), sim,
                 framing, face.normed_embedding.astype(np.float32).tobytes(),
                 now_iso(), _json.dumps(bbox_norm)),
            )

            if area_ratio < cfg.min_face_area:
                conn.execute(
                    "UPDATE frames SET status='rejected_smallface' WHERE frame_id=?",
                    (frame_id,),
                )
                n_small += 1
            elif sim is not None and sim < cfg.identity_threshold:
                conn.execute(
                    "UPDATE frames SET status='rejected_identity' WHERE frame_id=?",
                    (frame_id,),
                )
                n_wrongid += 1
            else:
                n_kept += 1

        if i % cfg.batch_log_every == 0 or i == len(rows):
            conn.commit()
            rate = i / max(0.001, time.time() - t0)
            yield (f"Scanned {i}/{len(rows)} ({rate:.1f} img/s) | kept {n_kept} | "
                   f"no-face {n_noface} | small {n_small} | wrong-id {n_wrongid}")
    conn.commit()

    framing_counts = dict(conn.execute(
        "SELECT d.framing, COUNT(*) FROM detections d "
        "JOIN frames f ON f.frame_id = d.frame_id "
        "WHERE f.status='selected' GROUP BY d.framing"
    ).fetchall())

    yield (
        "\nSMART CURATION DONE.\n"
        f"Kept (subject confirmed): {n_kept}\n"
        f"Rejected no-face: {n_noface}\n"
        f"Rejected small-face: {n_small}\n"
        f"Rejected wrong-identity: {n_wrongid}\n"
        f"Framing mix of survivors: {framing_counts}\n"
        "\nNext: lora-studio package"
    )
