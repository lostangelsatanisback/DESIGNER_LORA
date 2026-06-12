"""SQLite manifest: connection handling, versioned schema migrations, helpers.

Schema version lives in PRAGMA user_version. Migrations are additive and
idempotent-safe; a v1.0 single-file manifest upgrades in place, no data loss.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from .config import DB_NAME
from .util import human_bytes, now_iso, stable_id

# -----------------------------
# Migrations
# -----------------------------

MIGRATIONS: dict[int, str] = {
    # v1 - core pipeline (matches legacy single-file schema)
    1: """
        CREATE TABLE IF NOT EXISTS sources (
            source_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            kind TEXT NOT NULL,
            size_bytes INTEGER,
            mtime REAL,
            discovered_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'discovered',
            output_dir TEXT,
            frames_extracted INTEGER DEFAULT 0,
            error TEXT,
            meta_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sources_kind ON sources(kind);
        CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);

        CREATE TABLE IF NOT EXISTS frames (
            frame_id TEXT PRIMARY KEY,
            source_id TEXT,
            path TEXT NOT NULL,
            dhash TEXT,
            sharpness REAL,
            brightness REAL,
            status TEXT NOT NULL DEFAULT 'new',
            scored_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_frames_source ON frames(source_id);
        CREATE INDEX IF NOT EXISTS idx_frames_status ON frames(status);

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT,
            ts TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL
        );
    """,
    # v2 - smart curation (face/identity) + future captioning + key/value meta
    2: """
        CREATE TABLE IF NOT EXISTS detections (
            frame_id TEXT PRIMARY KEY,
            face_count INTEGER,
            face_area REAL,
            det_conf REAL,
            identity_sim REAL,
            framing TEXT,
            embed BLOB,
            detected_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_det_sim ON detections(identity_sim);

        CREATE TABLE IF NOT EXISTS captions (
            frame_id TEXT PRIMARY KEY,
            tags_json TEXT,
            caption_text TEXT,
            edited INTEGER DEFAULT 0,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """,
    # v3 - CLIP diversity clustering (Phase 2.4)
    3: """
        ALTER TABLE frames ADD COLUMN cluster_id INTEGER;

        CREATE TABLE IF NOT EXISTS clip_embeds (
            frame_id TEXT PRIMARY KEY,
            embed BLOB,
            embedded_at TEXT
        );
    """,
    # v4 - versioned dataset builds (Phase 4)
    4: """
        CREATE TABLE IF NOT EXISTS datasets (
            version INTEGER PRIMARY KEY,
            recipe_name TEXT NOT NULL,
            recipe_json TEXT,
            content_hash TEXT,
            built_at TEXT,
            image_count INTEGER,
            val_count INTEGER,
            dir TEXT,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS dataset_frames (
            version INTEGER NOT NULL,
            frame_id TEXT NOT NULL,
            concept TEXT NOT NULL,
            split TEXT NOT NULL DEFAULT 'train',
            caption TEXT,
            PRIMARY KEY (version, frame_id, concept)
        );
        CREATE INDEX IF NOT EXISTS idx_df_version ON dataset_frames(version);
    """,
    # v5 - subject bbox for smart-crop + training runs (Phase 5)
    5: """
        ALTER TABLE detections ADD COLUMN bbox TEXT;

        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dataset_version INTEGER,
            preset TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            config_json TEXT,
            command TEXT,
            output_dir TEXT,
            started_at TEXT,
            finished_at TEXT,
            current_step INTEGER DEFAULT 0,
            total_steps INTEGER DEFAULT 0,
            current_epoch INTEGER DEFAULT 0,
            total_epochs INTEGER DEFAULT 0,
            last_loss REAL,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS run_metrics (
            run_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            loss REAL,
            ts TEXT,
            PRIMARY KEY (run_id, step)
        );
    """,
    # v6 - evaluation / test generation (Phase 6)
    6: """
        CREATE TABLE IF NOT EXISTS evals (
            eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
            lora TEXT NOT NULL,
            label TEXT,
            category TEXT,
            prompt TEXT,
            seed INTEGER,
            backend TEXT,
            image_path TEXT,
            likeness REAL,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_evals_lora ON evals(lora);
    """,
    # v7 - Study Intelligence Layer (professional study classification)
    7: """
        CREATE TABLE IF NOT EXISTS study_labels (
            frame_id TEXT PRIMARY KEY,
            study_primary TEXT,
            study_tags TEXT,
            study_confidence REAL,
            study_reason_codes TEXT,
            figure_study_score REAL,
            fashion_study_score REAL,
            lingerie_fashion_score REAL,
            pose_clarity_score REAL,
            silhouette_clarity_score REAL,
            garment_visibility_score REAL,
            identity_lock_score REAL,
            study_review_status TEXT DEFAULT 'auto',
            study_export_eligible INTEGER DEFAULT 0,
            manual_override INTEGER DEFAULT 0,
            classified_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_study_primary
            ON study_labels(study_primary);
        CREATE INDEX IF NOT EXISTS idx_study_conf
            ON study_labels(study_confidence);
    """,
    # v8 - Concept Control Layer (visual LoRA explorer + controlled variation)
    8: """
        CREATE TABLE IF NOT EXISTS lora_influence_profiles (
            lora_id TEXT PRIMARY KEY,
            path TEXT,
            family TEXT,
            influence_tags TEXT,
            weight_min REAL,
            weight_max REAL,
            weight_default REAL,
            identity_risk TEXT,
            known_conflicts TEXT,
            notes TEXT,
            preview TEXT,
            updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_lip_family
            ON lora_influence_profiles(family);

        CREATE TABLE IF NOT EXISTS concept_control_presets (
            name TEXT PRIMARY KEY,
            kind TEXT,
            payload TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS variation_batches (
            batch_id TEXT PRIMARY KEY,
            spec TEXT,
            base_model TEXT,
            job_count INTEGER,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS variation_jobs (
            batch_id TEXT,
            variation_id TEXT,
            prompt TEXT,
            negative TEXT,
            seed INTEGER,
            loras TEXT,
            slider_state TEXT,
            warnings TEXT,
            output_path TEXT,
            created_at TEXT,
            PRIMARY KEY (batch_id, variation_id)
        );
    """,
    # v9 - Batch Variation Controller (modes, scoring, resumability)
    9: """
        ALTER TABLE variation_batches ADD COLUMN mode TEXT;
        ALTER TABLE variation_batches ADD COLUMN source_state TEXT;
        ALTER TABLE variation_batches ADD COLUMN identity_anchor TEXT;
        ALTER TABLE variation_batches ADD COLUMN hard_cap INTEGER;
        ALTER TABLE variation_jobs ADD COLUMN preservation_score REAL;
        ALTER TABLE variation_jobs ADD COLUMN risk_level TEXT;
        ALTER TABLE variation_jobs ADD COLUMN status TEXT DEFAULT 'planned';
    """,
    # v10 - Wardrobe Variation & Selective Region Editing
    10: """
        CREATE TABLE IF NOT EXISTS wardrobe_edits (
            edit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            mask_path TEXT,
            region_id TEXT,
            edit_mode TEXT,
            prompt TEXT,
            negative TEXT,
            loras TEXT,
            denoise REAL,
            seed INTEGER,
            preserve_background INTEGER,
            preserve_pose INTEGER,
            identity_score REAL,
            risk_level TEXT,
            readiness TEXT,
            output_path TEXT,
            created_at TEXT
        );
    """,
    # v11 - measured QA: face similarity on generated variations
    11: """
        ALTER TABLE variation_jobs ADD COLUMN measured_face_sim REAL;
    """,
}

SCHEMA_VERSION = max(MIGRATIONS)


def connect(output_base: Path) -> sqlite3.Connection:
    output_base.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_base / DB_NAME, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("SELECT 1;").fetchone()
    except sqlite3.OperationalError:
        # WAL unsupported on some network/exFAT volumes
        conn.close()
        conn = sqlite3.connect(output_base / DB_NAME, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=DELETE;")
        conn.execute("PRAGMA synchronous=FULL;")
        logging.info("SQLite WAL unavailable on this volume; using DELETE journal")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version;").fetchone()[0]
    for version in sorted(MIGRATIONS):
        if version > current:
            conn.executescript(MIGRATIONS[version])
            conn.execute(f"PRAGMA user_version = {version};")
            conn.commit()
            logging.info("Manifest migrated to schema v%d", version)


# -----------------------------
# Meta key/value
# -----------------------------

def meta_set(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value)),
    )
    conn.commit()


def meta_get(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


# -----------------------------
# Sources / frames / events
# -----------------------------

def event(conn: sqlite3.Connection, source_id: str, level: str, message: str) -> None:
    conn.execute(
        "INSERT INTO events(source_id, ts, level, message) VALUES (?, ?, ?, ?)",
        (source_id, now_iso(), level, message),
    )
    conn.commit()


def upsert_source(
    conn: sqlite3.Connection,
    path: Path,
    kind: str,
    status: str = "discovered",
    meta: Optional[dict] = None,
) -> str:
    sid = stable_id(path)
    st = path.stat()
    conn.execute(
        """
        INSERT INTO sources(
            source_id, path, kind, size_bytes, mtime, discovered_at, status, meta_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            path=excluded.path, kind=excluded.kind,
            size_bytes=excluded.size_bytes, mtime=excluded.mtime
        """,
        (sid, str(path), kind, st.st_size, st.st_mtime, now_iso(), status,
         json.dumps(meta or {}, ensure_ascii=False)),
    )
    conn.commit()
    return sid


def get_status(conn: sqlite3.Connection, source_id: str) -> Optional[str]:
    row = conn.execute(
        "SELECT status FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()
    return row[0] if row else None


def mark(
    conn: sqlite3.Connection,
    source_id: str,
    status: str,
    *,
    output_dir: Optional[Path] = None,
    frames_extracted: Optional[int] = None,
    error: Optional[str] = None,
    meta: Optional[dict] = None,
) -> None:
    updates = ["status = ?"]
    values: list[object] = [status]
    if output_dir is not None:
        updates.append("output_dir = ?")
        values.append(str(output_dir))
    if frames_extracted is not None:
        updates.append("frames_extracted = ?")
        values.append(frames_extracted)
    if error is not None:
        updates.append("error = ?")
        values.append(error)
    if meta is not None:
        updates.append("meta_json = ?")
        values.append(json.dumps(meta, ensure_ascii=False))
    values.append(source_id)
    conn.execute(f"UPDATE sources SET {', '.join(updates)} WHERE source_id = ?", values)
    conn.commit()


def stats(conn: sqlite3.Connection) -> dict:
    out: dict = {"sources": {}, "frames": {}, "totals": {}}
    for kind, status, n, size in conn.execute(
        "SELECT kind, status, COUNT(*), COALESCE(SUM(size_bytes),0) "
        "FROM sources GROUP BY kind, status"
    ):
        out["sources"].setdefault(kind, {})[status] = {"count": n, "bytes": size}
    for status, n in conn.execute("SELECT status, COUNT(*) FROM frames GROUP BY status"):
        out["frames"][status] = n
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(frames_extracted),0), COALESCE(SUM(size_bytes),0) "
        "FROM sources"
    ).fetchone()
    out["totals"] = {
        "sources": row[0],
        "frames_extracted": row[1],
        "source_bytes": row[2],
        "source_bytes_h": human_bytes(row[2]),
    }
    try:
        out["totals"]["faces_scanned"] = conn.execute(
            "SELECT COUNT(*) FROM detections"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        pass
    return out
