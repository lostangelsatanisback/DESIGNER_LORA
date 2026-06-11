"""Basic curation: dHash perceptual dedup, Laplacian blur scoring,
exposure filtering. Pure stdlib (Pillow used opportunistically)."""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, Optional

from .. import manifest
from ..config import CurateConfig
from ..util import HAVE_PIL, now_iso, setup_logging

if HAVE_PIL:
    from PIL import Image


def load_gray(path: Path, w: int, h: int) -> Optional[list[int]]:
    """Grayscale w*h pixel list. Pillow if available, else ffmpeg rawvideo."""
    if HAVE_PIL:
        try:
            with Image.open(path) as im:
                im = im.convert("L").resize((w, h), Image.BILINEAR)
                return list(im.getdata())
        except Exception:
            return None
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", str(path),
             "-vf", f"scale={w}:{h}", "-pix_fmt", "gray",
             "-f", "rawvideo", "-"],
            capture_output=True,
        )
        data = proc.stdout
        if len(data) < w * h:
            return None
        return list(data[: w * h])
    except Exception:
        return None


def dhash_bits(gray9x8: list[int]) -> int:
    bits = 0
    for row in range(8):
        for col in range(8):
            left = gray9x8[row * 9 + col]
            right = gray9x8[row * 9 + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def laplacian_variance(gray: list[int], w: int, h: int) -> float:
    vals: list[float] = []
    for y in range(1, h - 1):
        base = y * w
        for x in range(1, w - 1):
            c = gray[base + x]
            lap = (gray[base + x - 1] + gray[base + x + 1]
                   + gray[base - w + x] + gray[base + w + x] - 4 * c)
            vals.append(float(lap))
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def score_frame(path_str: str) -> Optional[tuple[str, float, float]]:
    p = Path(path_str)
    g9 = load_gray(p, 9, 8)
    if g9 is None:
        return None
    dh = dhash_bits(g9)
    g = load_gray(p, 96, 96)
    if g is None:
        return None
    sharp = laplacian_variance(g, 96, 96)
    bright = sum(g) / len(g)
    return (f"{dh:016x}", sharp, bright)


def curate_generator(cfg: CurateConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)

    where = "status = 'new'" if not cfg.rescore else "1=1"
    rows = conn.execute(
        f"SELECT frame_id, source_id, path FROM frames WHERE {where} "
        "ORDER BY source_id, path"
    ).fetchall()
    if not rows:
        yield "No frames to curate. Run extract/photos first."
        return

    yield f"Scoring {len(rows)} frames (workers={cfg.workers}, pillow={HAVE_PIL})...\n"
    done = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(score_frame, r[2]): r for r in rows}
        batch: list[tuple] = []
        for fut in as_completed(futures):
            frame_id, source_id, path = futures[fut]
            try:
                res = fut.result()
            except Exception:
                res = None
            if res is None:
                failed += 1
                batch.append((None, None, None, "decode_failed", now_iso(), frame_id))
            else:
                dh, sharp, bright = res
                batch.append((dh, sharp, bright, "scored", now_iso(), frame_id))
            done += 1
            if len(batch) >= 200:
                conn.executemany(
                    "UPDATE frames SET dhash=?, sharpness=?, brightness=?, "
                    "status=?, scored_at=? WHERE frame_id=?",
                    batch,
                )
                conn.commit()
                batch.clear()
                rate = done / max(0.001, time.time() - t0)
                yield f"Scored {done}/{len(rows)} ({rate:.0f} img/s, {failed} failed)"
        if batch:
            conn.executemany(
                "UPDATE frames SET dhash=?, sharpness=?, brightness=?, "
                "status=?, scored_at=? WHERE frame_id=?",
                batch,
            )
            conn.commit()

    yield f"Scoring done: {done} frames, {failed} failed. Applying quality filters...\n"

    rejected_blur = conn.execute(
        "UPDATE frames SET status='rejected_blur' "
        "WHERE status='scored' AND ? > 0 AND sharpness < ?",
        (cfg.min_sharpness, cfg.min_sharpness),
    ).rowcount
    rejected_exposure = conn.execute(
        "UPDATE frames SET status='rejected_exposure' "
        "WHERE status='scored' AND (brightness < ? OR brightness > ?)",
        (cfg.min_brightness, cfg.max_brightness),
    ).rowcount
    conn.commit()

    yield "Deduplicating (dHash)...\n"
    dup_count = 0
    sources = [r[0] for r in conn.execute(
        "SELECT DISTINCT source_id FROM frames WHERE status='scored'"
    )]
    for sid in sources:
        frames = conn.execute(
            "SELECT frame_id, dhash, sharpness FROM frames "
            "WHERE source_id=? AND status='scored' ORDER BY path",
            (sid,),
        ).fetchall()
        kept: list[tuple[str, int, float]] = []
        dups: list[str] = []
        for frame_id, dh_hex, sharp in frames:
            if not dh_hex:
                continue
            h = int(dh_hex, 16)
            is_dup = any(
                hamming(h, kh) <= cfg.hamming_threshold
                for _, kh, _ in kept[-12:]
            )
            if is_dup:
                dups.append(frame_id)
            else:
                kept.append((frame_id, h, sharp or 0.0))
        if dups:
            conn.executemany(
                "UPDATE frames SET status='duplicate' WHERE frame_id=?",
                [(d,) for d in dups],
            )
            dup_count += len(dups)
        conn.executemany(
            "UPDATE frames SET status='selected' WHERE frame_id=?",
            [(k[0],) for k in kept],
        )
        conn.commit()

    sel = conn.execute("SELECT COUNT(*) FROM frames WHERE status='selected'").fetchone()[0]
    yield (
        "\nBASIC CURATION DONE.\n"
        f"Selected: {sel}\n"
        f"Duplicates removed: {dup_count}\n"
        f"Rejected (blur): {rejected_blur}\n"
        f"Rejected (exposure): {rejected_exposure}\n"
        f"Decode failures: {failed}\n"
        "\nNext: lora-studio curate --smart   (identity filtering, needs [ai] extras)"
    )
