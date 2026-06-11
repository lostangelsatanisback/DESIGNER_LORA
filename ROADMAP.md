# LoRA Designer Studio — Production Roadmap

**Project:** SPOOKUMS_STUDIO · personal LoRA design pipeline for Pony Diffusion V6 XL
**Hardware:** Apple Silicon M2 Max 32GB (MPS) · 2TB+ source library on CURATOR_SSD
**Doctrine:** local & private · non-destructive · resumable · manifest-tracked · zero mandatory deps in the core

---

## Part 0 — Current State Assessment (v1.0)

### What exists today (`lora_studio.py`, 1,699 lines, verified end-to-end)

| Stage | Status | Notes |
|---|---|---|
| Frame extraction | ✅ Production | HDR-aware tonemap, VideoToolbox, segmented, resumable, ffmpeg 4–7 compatible |
| Photo import | ✅ Production | hardlink/copy/symlink, cross-device fallback |
| Manifest | ✅ Production | SQLite (WAL + fallback), `sources`/`frames`/`events` tables |
| Curation | ✅ v1 | dHash dedup, Laplacian blur rejection, exposure filter — *no semantic understanding yet* |
| Packaging | ✅ v1 | kohya `N_token class` structure, captions, per-video balancing |
| Web UI | ✅ v1 | stdlib dashboard, job runner, live stats — *no image preview yet* |
| Training | ❌ | Not started — datasets are handed off to kohya_ss manually |
| Captioning | ⚠️ Static | Single fixed caption; no tagger |
| Evaluation | ❌ | Not started |

### Honest gaps blocking "studio grade"

1. **Curation is blind.** It knows sharpness, not subjects. It cannot tell a perfect face frame from a sharp photo of a wall.
2. **Captions are static.** Pony/SDXL LoRAs live or die on caption quality; one repeated string under-trains flexibility.
3. **No feedback loop.** Train → look at outputs → guess → retrain is manual. The studio should close this loop.
4. **Single file is at its ceiling.** 1.7k lines now; Phases 2–6 would push it past 6k. Migration required first.
5. **No dataset versioning.** You can't answer "which dataset produced my best LoRA?"

---

## Guiding Principles (apply to every phase)

- **Core stays stdlib.** Heavy deps (torch, onnxruntime, CLIP) live in *optional* extras; every AI feature has a graceful "not installed → stage disabled with instructions" path so the tool still runs anywhere.
- **Never touch source media.** All derived data goes to EXTRACTED_FRAMES and the manifest.
- **Everything resumable.** Any stage killed mid-run continues from the manifest.
- **MPS first.** All model inference targets `mps` device with fp16, batch sizes tuned for 32GB unified memory.
- **Each phase ships usable.** No phase depends on a later one to be valuable.

---

## Phase 1 — Package Migration & Hardening
*Foundation. Everything else builds on this.*

**Goal:** Convert the single file into a maintainable package without losing drop-anywhere convenience.

```
lora_studio/
├── pyproject.toml            # extras: [ai], [caption], [train], [eval], [all]
├── lora_studio/
│   ├── __main__.py           # python -m lora_studio  (same CLI)
│   ├── config.py             # project.toml loader (paths, token, presets)
│   ├── manifest.py           # typed DB layer + schema migrations
│   ├── extract.py            # current extraction stage
│   ├── photos.py
│   ├── curate/basic.py       # dHash/blur/exposure (current v1)
│   ├── package.py
│   ├── ui/server.py  ui/static/
│   └── util.py
└── tests/                    # pytest: synthetic-video e2e, manifest, slug/hash units
```

**Key tasks**
- `project.toml` per project (SPOOKUMS_STUDIO) — paths, trigger word, defaults. No more editing constants in code. CLI: `lora-studio --project spookums.toml <stage>`.
- Manifest **schema migrations table** (`schema_version`) so future phases can alter tables safely on a 2TB-scale live manifest.
- `pip install -e .` plus a `make single` script that re-bundles to one file (stitcher) for portability when wanted.
- Test suite from the verification work already done (synthetic videos, dry-run/resume semantics, ffmpeg version matrix).

**Acceptance:** all v1.0 commands behave identically; tests green; manifest from v1.0 auto-migrates.
**Effort:** 1–2 sessions.

---

## Phase 2 — Smart Curation Engine (AI eyes)  ✅ SHIPPED (v2.2)
*2.1 detection ✅ · 2.2 identity ✅ · 2.3 framing ✅ (face-geometry; body-pose deferred) · 2.4 CLIP clustering ✅ · 2.5 review grid ✅ · composition quotas in packaging ✅*

**Goal:** Curation that understands *subjects*: who is in frame, how they're framed, how diverse the selection is.

**2.1 Person & face detection** — `[ai]` extra
- InsightFace (`buffalo_l`) or MediaPipe on MPS/CoreML; per-frame: face count, bbox, landmark quality, face area ratio.
- New `detections` table: `frame_id, face_count, face_area, face_conf, embed BLOB`.
- Reject frames with zero subject; flag multi-person frames for review.

**2.2 Identity filtering**
- You provide 5–15 reference images → mean face embedding ("anchor").
- Every frame scored by cosine similarity to anchor → automatically excludes other people, background strangers, TV screens. Threshold tunable in UI.

**2.3 Framing & pose classification**
- Classify each frame: `closeup / portrait / upper-body / full-body / pov / back / profile` from face bbox geometry + body keypoints (lightweight YOLO-pose or MediaPipe Pose).
- Stored as `framing` tag → packaging can enforce **composition quotas** (e.g. 30% closeup, 30% upper, 25% full, 15% profile/back — the classic balanced character-LoRA recipe).

**2.4 Diversity-aware selection**
- CLIP image embeddings (OpenCLIP ViT-B/32 on MPS) → cluster by scene/outfit/lighting (HDBSCAN or k-means).
- Selection maximizes coverage across clusters instead of "sharpest N per video" — kills the #1 LoRA failure mode (1000 near-identical frames from one couch angle).

**2.5 Visual review UI**
- Dashboard gains a **grid view**: thumbnails (lazy-generated 256px), filter by status/cluster/framing/similarity, click to approve/reject, keyboard-driven triage. Manual verdicts override automation and are remembered.

**Acceptance:** run on a real video set → selection visibly balanced across framing types, zero non-subject frames, review grid usable at 10k+ frames.
**Effort:** 3–4 sessions. **Depends:** Phase 1.

---

## Phase 3 — Auto-Captioning System  ✅ SHIPPED (v2.3)
*3.1 WD14 v3 tagger ✅ · 3.2 rules engine ✅ · 3.3 inline caption editor ✅ + Tags frequency dashboard ✅ (edited captions are never overwritten and flow into builds).*

**3.1 Tagger backend** — `[caption]` extra
- WD14 tagger (SwinV2/ConvNeXt ONNX) via onnxruntime (CoreML EP on Apple Silicon); threshold-tuned booru tags per frame → `captions` table (`frame_id, tags_json, caption_text, edited INTEGER`).

**3.2 Caption rules engine**
- Declarative rules in `project.toml`:
  - always prepend trigger (`spookums`), inject class word;
  - Pony quality prefix policy (`score_9, score_8_up, …`) on/off;
  - tag blacklist/whitelist, tag remapping (e.g. `1girl → spookums`);
  - **prune-what-you-train rule**: drop tags describing permanent character traits (so they bind to the trigger), keep tags for variable things (outfit, pose, background).
- Per-framing-type rule overrides (POV frames get POV-relevant tags kept, etc.).

**3.3 Caption editor in UI**
- Grid view shows caption under thumbnail; inline edit; bulk find/replace across selection; tag frequency dashboard ("`smile` appears in 92% — consider pruning").

**Acceptance:** packaged dataset has unique, rule-compliant captions per image; tag frequency view works; manual edits survive re-runs.
**Effort:** 2–3 sessions. **Depends:** Phase 1 (Phase 2 enriches it but not required).

---

## Phase 4 — Dataset Composer & Versioning  ✅ SHIPPED (v2.4)
*Recipes ✅ · versioned snapshots ✅ · identical hashes ✅ · diff ✅ · stable val split ✅ · multi-concept ✅ · subject-centered SDXL-bucket smart-crop ✅ (`smart_crop = true` in any recipe; face-bbox POI from schema v5, center fallback).*

- **Aspect-ratio bucketing aware:** group by native AR, optional smart-crop (subject-centered using Phase 2 bboxes) to SDXL bucket resolutions instead of blind center-crop.
- **Dataset recipes:** named recipe in `project.toml` = filters + quotas + caption ruleset + repeats. `lora-studio build --recipe character_v3`.
- **Versioned snapshots:** every build writes `DATASET/v003__character_v3/` + `dataset.json` (content hash, frame list, recipe, stats). Manifest table `datasets` links builds → frames.
- **Dataset diff:** `lora-studio diff v002 v003` → added/removed frames, caption changes, quota shifts.
- **Train/val split** with held-out frames for Phase 6 eval.
- **Multi-concept support:** several folders (`10_spookums person`, `5_outfitX`) from one recipe for outfit/style sub-LoRAs.

**Acceptance:** two builds from identical state produce identical hashes; diff is accurate; recipes cover the 5 LoRA types (character/style/outfit/pose/detail).
**Effort:** 2 sessions. **Depends:** Phases 1–3.

---

## Phase 5 — Training Orchestration  ✅ SHIPPED (v2.4)
*5.1 sd-scripts adapter ✅ (swappable `trainer_cmd` backend interface) · 5.2 all 5 M2-Max presets ✅ · 5.3 queue + live loss parsing into `runs`/`run_metrics`, loss-curve canvas in Train tab, per-run metadata JSON, auto-naming `{token}_{preset}_v{NNN}` ✅. Set `sd_scripts_dir` + `base_model` in Settings to activate.*

**5.1 Backend adapter** — `[train]` extra
- Wraps `sd-scripts` (`sdxl_train_network.py`) as subprocess; generates its TOML config; captures stdout to manifest events. Adapter interface so a future Draw Things / OneTrainer backend can slot in.

**5.2 M2-Max-32GB preset library** (per LoRA type)
| Preset | dim/alpha | LR (UNet/TE) | Batch | Notes |
|---|---|---|---|---|
| Character | 32/16 | 1e-4 / 5e-5 | 1–2 | cosine, min_snr_gamma 5, ~2500 steps |
| Style | 16/8 | 8e-5 / 0 | 2 | TE off, more epochs |
| Outfit | 16/8 | 1e-4 / 4e-5 | 1–2 | tight captions |
| Pose/POV | 8/4 | 1e-4 / 0 | 2 | low dim, regularization images |
| Detail refiner | 64/32 | 5e-5 / 2e-5 | 1 | short runs, high-res subset |
All presets: fp16, gradient checkpointing, `--mem_eff_attn`, cache latents to disk, MPS env guards (`PYTORCH_ENABLE_MPS_FALLBACK=1`).

**5.3 Training queue & monitoring**
- `runs` table; queue multiple recipes × presets overnight; UI tab shows live loss curve (parsed from logs), step rate, ETA, per-epoch checkpoint list; safe stop = finish epoch then halt.
- Auto-name outputs `spookums_char_v003_e08.safetensors` + write run metadata JSON next to each checkpoint.

**Acceptance:** one click from recipe → queued run → checkpoints + loss curve in UI; a full character preset completes on the M2 Max without OOM.
**Effort:** 3–4 sessions. **Depends:** Phase 4.

---

## Phase 6 — Evaluation & Iteration Loop  ✅ CORE SHIPPED (v2.5)
*Forge API client ✅ · local diffusers backend ✅ (pipeline cache, 3-tier LoRA fallback w/ negative weights, MPS hygiene — patterns from the Dressing Studio analysis) · 5-category test matrix w/ fixed seeds ✅ · anchor-based likeness scoring + overfit gap check ✅ · labeled grids + `evals` table ✅ · Test tab (txt2img/img2img, LoRA stack, matrix runner, results) ✅. Remaining: A/B blind compare UI, auto best-epoch sweep.*

- **Forge/A1111 API client** (`[eval]` extra): for each checkpoint epoch, generate a fixed prompt matrix (likeness set, flexibility set, style-bleed set) at fixed seeds → `evals` table + image grid per epoch.
- **Likeness scoring:** face embedding similarity between generations and the Phase 2 anchor → numeric likeness per epoch; plot likeness-vs-epoch next to loss curve (the elbow finds your best epoch, not vibes).
- **Flexibility probes:** prompts the dataset doesn't contain (new outfits, settings) — detect overfit when likeness is high but flexibility collapses.
- **A/B compare UI:** two epochs/runs side-by-side, same seeds; blind-pick mode; verdicts stored.
- **Regression tracking:** dataset v002 → v003 — did likeness/flexibility improve? Closes the loop back to Phase 4 recipes.

**Acceptance:** after a queued run, the studio recommends a best epoch with grids and scores to back it; comparisons reproducible (fixed seeds).
**Effort:** 3 sessions. **Depends:** Phase 5 + running Forge instance.

---

## Phase 7 — Studio Polish & Scale  ✅ SHIPPED (v2.6)
*Pipeline DAG w/ gates + resume ✅ · watch/auto-ingest ✅ · Merging Lab ✅ (concat method, mixed ranks, negative weights, te/down/mid/up block multipliers, one-click combo) · model registry + model cards ✅ · A/B blind compare ✅ · gc + space dashboard ✅ · .pyz bundling ✅. Original scope below for reference.*

### Original scope
- **Pipeline DAG:** `lora-studio pipeline run full` = extract → curate → caption → build → train → eval as one resumable chain with per-stage gates ("pause for manual review after curation").
- **Watch folders:** new media dropped into CURATOR_SSD auto-ingests on next run.
- **Global dedup index:** cross-video/cross-photo dHash + embedding index — the same moment captured by video and Live Photo only enters a dataset once.
- **Model registry:** every trained LoRA with its recipe, dataset version, eval scores, sample grid — searchable in UI; export "model card" per LoRA.
- **Multi-project workspaces:** N characters/styles, isolated manifests, shared cache.
- **Disk hygiene:** `lora-studio gc` (orphaned thumbnails, superseded dataset builds), space dashboards for the 2TB volume.

**Effort:** 2–3 sessions, parallelizable with 5/6.

---

## Phase 7.5 — Creator Wizard  ✅ SHIPPED (v2.7)
*The offline Civitai-creator flow: dataset analysis from the manifest, explainable type auto-detection (character/style/outfit/pose/detail/explicit), type-tuned recipe + preset injection, one-click build→train→matrix→model card, merge previews. UI Create tab + `lora-studio analyze` / `lora-studio wizard`.*

## Phase 8 — Frontier ("go wild")  ← NEXT
Concrete extension plan, ordered by payoff (each builds on shipped pieces):

1. **Best-epoch sweep** ✅ SHIPPED (v2.8): `lora-studio sweep --run NAME` — evaluates every epoch checkpoint (resumable; skips already-evaluated), epoch scoreboard with likeness/flexibility/gap, recommends best epoch with overfit guard. Also first-class in the Grokkie Dataset Factory (Step 6).
2. **Active-learning curation** (2 sessions): mine eval failures (low-likeness categories) → embed failure prompts with CLIP → search unscanned frames in the 2TB pool by similarity → propose "v+1 candidates" in the Review tab. Uses stored CLIP + face embeddings; zero re-scanning.
3. **Autopilot** (2 sessions): `autopilot --target-likeness 0.82 --max-cycles 4` — build → train → sweep → adjust recipe (quota shift / caption prune from tag stats) → repeat, with the DAG's gates as human checkpoints.
4. **Scene-aware sampling** (1 session): PySceneDetect shot boundaries → per-shot sharpest-frame extraction; plugs into ExtractConfig.
5. **ControlNet asset factory** (1 session): export pose skeletons/depth maps from best frames via reForge's bundled preprocessors (`extensions-builtin/forge_legacy_preprocessors`).
6. **Prompt lab** (1 session): measure which Pony quality-prefix variants actually score best for YOUR LoRA via the eval matrix — data, not folklore.

### Original frontier notes

1. **Active-learning curation.** After an eval, the studio identifies *failure modes* (e.g. weak profile likeness) and searches the unscored 2TB pool for frames that fill that gap (embedding similarity to failure prompts) → proposes "dataset v004 candidates". Curation becomes targeted, not exhaustive.
2. **Auto-iteration loop.** `lora-studio autopilot --target-likeness 0.82 --max-runs 4`: build → train → eval → adjust recipe (quota shifts, caption pruning) → repeat, with human gates between cycles.
3. **Scene-aware video sampling.** PySceneDetect shot boundaries → sample per shot instead of per second; motion-peak frame picking inside each shot (sharpest frame of each distinct moment).
4. **ControlNet asset factory.** Export pose skeletons / depth maps from your best frames → reusable pose library for generation-time composition control.
5. **LoRA lab.** Merge/extract tools (weight blending, block-weight merge presets), strength-sweep grids, interference matrix when stacking character+outfit+pose LoRAs.
6. **Prompt lab.** Pony prompt template manager wired to eval results — which quality-tag prefix actually scores best *for your LoRA*, measured not assumed.
7. **Studio agent (optional, local-first).** Natural-language control surface: "build me a closeup-heavy v5 without the red couch cluster and queue a character run" → translated to recipe + queue ops. Claude API behind an explicit opt-in flag; nothing leaves the machine otherwise.

---

## Sequencing & Effort Summary

```
P1 Foundation ──► P2 Smart Curation ──► P3 Captioning ──► P4 Composer ──► P5 Training ──► P6 Eval
                                                                              │
                                              P7 Polish ◄────────────────────┘ (parallel)
                                              P8 Frontier (after P6 loop closes)
```

| Phase | Sessions (est.) | Unlocks |
|---|---|---|
| 1 Foundation | 1–2 | maintainability, config, tests |
| 2 Smart curation | 3–4 | subject-aware datasets |
| 3 Captioning | 2–3 | per-frame Pony captions |
| 4 Composer | 2 | reproducible versioned datasets |
| 5 Training | 3–4 | one-click queued kohya runs |
| 6 Eval | 3 | measured best-epoch selection |
| 7 Polish | 2–3 | full pipeline automation |
| 8 Frontier | open | self-improving studio |

**Total to a closed train-eval loop: ~14–18 working sessions.**

## Risk Register
- **MPS instability in sd-scripts** → mitigation: pinned torch version per preset, CPU-fallback env flags, smoke-test run (50 steps) before queueing long runs.
- **onnxruntime/CoreML quirks for WD14** → mitigation: CPU EP fallback, cache all tagger outputs in manifest (never re-run).
- **Manifest scale (millions of frame rows)** → mitigation: indexes already in place; add batch writers + `PRAGMA mmap_size`; thumbnails on disk, never in DB.
- **Disk pressure on 2TB volume** → mitigation: Phase 7 gc + hardlink-everywhere policy (already default).

---

*Next concrete step: Phase 1 — package migration. Say the word and we start.*
