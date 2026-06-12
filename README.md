# LoRA Designer Studio (v3.7)

## Concept Metadata & Sidecar Hardening (v3.7 - Concept Control Feature D)

Every LoRA's metadata now normalizes into one rich, backward-compatible
shape (`lora_studio/concept_metadata.py` - `ConceptMetadata`).

- **Schema v2 sidecars** - `schema_version`, `display_name`,
  `concept_family`, `concept_tags`, `control_axes` (axis_id, label,
  description, recommended_range, safe_default, response_curve,
  identity_sensitivity), `priority_hint`
  (anchor/primary/normal/supporting/experimental), `identity_risk_level`,
  `recommended_weight_range`, structured `known_conflicts`
  (concept_family or lora_id + reason + severity), `preview_images`,
  reserved `signal_hooks`, `notes`. Every field optional.
- **Legacy normalization** - old `tags` -> `concept_tags`, `preview` ->
  `preview_images.default`, missing family -> `general_concept`, missing
  risk -> `medium`, missing range -> conservative `[0.2, 0.7]`. Unknown
  enum values never crash: they normalize to safe defaults and surface in
  `normalization_warnings`. Discovery order unchanged and deterministic
  (`<stem>.concept.json` > `<stem>.json` > previews-folder `meta.json` >
  inferred).
- **Response curve registry** - linear / slow_start / fast_start / damped /
  stepped, ready for per-axis control without another schema rewrite.
- **Resolver integration (conservative)** - `priority_hint` adjusts
  resolver priority (supporting -15, experimental -25, anchor +40 and
  protected from auto-trim); a supporting LoRA outweighing the identity
  anchor raises a caution; experimental concepts get an advisory;
  family-level known conflicts warn when the conflicting family is active
  in the stack. v2 ranges/risk flow into the influence profile.
- **Explorer payload** - every card carries `concept_meta` (the full
  normalized object); cards summarize control axes, non-default priority
  hints, and known-conflict counts.

## Batch Variation Controller (v3.6 - Concept Control Feature C)

Production-grade controlled variation on top of the Concept Control stack.

- **Smart variation modes** - Low-Risk Studio Sweep (cap 24, value ceiling
  0.60, strictest preservation; the default), Balanced Exploration (cap 64,
  ceiling 0.85), Creative Exploration (cap 128, full range, explicit
  selection required). Mode caps are hard - oversized grids return guidance
  instead of jobs.
- **Variation axes** - up to three axes per grid, each as an explicit value
  list or `min:max:step`; values are validated against the slider spec and
  clipped to the mode ceiling. The identity anchor axis is protected: it
  can only be swept inside its safe band and is never weakened below the
  minimum. Axis metadata (`variation_axes()`) exposes recommended ranges
  and identity impact per attribute.
- **Every job resolves through stack intelligence** - identity anchor held
  fixed, per-job preservation score, risk level (`info` / `caution` /
  `high_risk` / `blocked_or_needs_review`), warnings, and resolved weights
  stored in the manifest (schema v9, additive columns). Deterministic job
  ids; `load_batch()` reloads any plan; generation is resumable (generated
  jobs skip; failures are marked).
- **UI** - mode selector, two axis rows with live job-count estimate vs
  cap, Create Variation Grid (dry-run: plan + manifest only), per-job
  preservation/risk table, and **Generate Variations** which queues the
  batch on the hub job queue against the live Forge engine.
- **CLI** - `lora-studio -p spookums.toml concept batch --spec grid.json
  --mode low_risk [--run]`.

## Concept Modulation & Weight Intelligence (v3.5 - Concept Control Feature B)

Identity preservation is now the resolver's highest-priority behavior.

- **Intelligent slider mapping** - sliders no longer map 0-1 linearly:
  concept families ride a smoothstep response curve inside their
  recommended operating range with soft caps (range max) and a hard cap
  (max x1.15); identity rides a protected linear band that never drops
  below half-range. When the stack is already heavy, additional concept
  weight is automatically damped to preserve identity headroom.
  Professional family aliases (`identity_anchor`, `garment_style`,
  `lighting_mood`, `pose_energy`, `form_emphasis`, `material_finish`,
  `rendering_style`, `camera_treatment`, ...) map onto canonical families;
  unknown families fall back to a conservative 0.10-0.30 studio-safe range.
  Numeric inputs and sliders stay in sync through exact curve inversion.
- **Stack intelligence v2** - concept priorities (identity 100 ... style 30,
  unknown 20): when total concept strength exceeds 1.60 the resolver trims
  lowest-priority concepts first and never reduces the identity anchor.
  Each item reports requested vs resolved weight, whether it was adjusted,
  and its pin state. Pinned (manual) weights are never auto-adjusted -
  only warned about. Four risk levels (stable / watch / elevated / high),
  four warning severities (info / advisory / caution / critical, sorted
  critical-first), influence-pressure ranking, conflict list, and a
  one-line studio summary.
- **Actionable recommendations** - raise identity anchor, lower total
  strength, deprioritize conflicting concept, reduce highest-risk LoRA,
  switch to a controlled variation batch - each with proposed weights.
  `recommended_weights` powers the UI's **Apply Suggested Balance** button.
- **UI** - live risk badge + preservation score, requested -> resolved
  weight column with adjustment/pin markers, click a weight to pin a
  manual value, warnings grouped by severity, recommendation panel,
  Apply Suggested Balance / Clear manual weights.

## Visual LoRA Explorer (v3.4 - Concept Control Feature A)

The Concept Control page's explorer is now a full visual browser:

- **Rich sidecars** - beside any LoRA, `<stem>.json` or `<stem>.concept.json`:

```json
{
  "display_name": "Example Style",
  "concept_tags": ["lighting_mood", "composition_flow"],
  "description": "Studio-facing visual style concept.",
  "recommended_weight": 0.65,
  "category": "lighting",
  "preview_images": {"default": "example_style.preview.png",
                     "low": "example_style.low.png"}
}
```

  Both schemas work (native `family`/`influence_tags` too); broken sidecars
  are tolerated; LoRAs with no metadata still appear (filename inference).
- **Strength previews** - `<stem>.preview.png` plus `<stem>.low/medium/
  high.png` beside the model, or a dedicated folder
  `<output_base>/previews/lora/<stem>/{default,low,medium,high}.png` with
  optional `meta.json`. Cards show level chips; missing previews show a
  clean placeholder.
- **Formats** - `.safetensors` (with stdlib metadata read), `.pt`, `.ckpt`.
- **Safe serving** - previews stream only through the scan index
  (`/api/concept/lora_preview/<id>/<level>`); no filesystem paths, no
  traversal.
- **Filters** - search, family, influence tag, preview availability; sort
  by name / modified / family / identity risk.
- **Add to stack** - one click adds at the recommended weight (identity
  LoRAs go to the front), updates the current stack for resolution, and
  persists the `explorer_stack` Playground preset.

## Concept Control Layer (v3.3)

Visually explore, combine, and finely control specialized concept LoRAs on
top of a strong identity anchor. New hub page **Concept Control** (next to
Playground) plus CLI. Pure stdlib; schema v8 (`lora_influence_profiles`,
`concept_control_presets`, `variation_batches`, `variation_jobs`).

```bash
lora-studio -p spookums.toml concept scan --write-sidecars
lora-studio -p spookums.toml concept stack --lora spookums_character_v001:0.75 --lora neon_style:0.3
lora-studio -p spookums.toml concept batch --spec grid.json --run
```

**Visual LoRA Explorer** scans `lora_output_dir` + Forge `models/Lora`,
reads .safetensors training metadata with a stdlib header parser (never
loads tensors), associates `<stem>.preview.png` thumbnails, and keeps a
*visual influence profile* per LoRA (concept family, influence tags,
recommended weight range, identity risk, known conflicts) in
`<stem>.concept.json` sidecars - sidecar edits override filename inference.
Filter by family/tag/search, sort by name, modified, family, identity risk.

**Attribute Controls** - 9 professional sliders (Identity Anchor Strength,
Garment Style Intensity, Form Emphasis, Lighting Mood, Style Intensity,
Texture Emphasis, Detail Refinement, Composition Strength, Background
Influence). Sliders resolve into per-LoRA weights inside each family's
recommended range; the identity anchor never drops below half its range.

**Stack Intelligence** - explained recommendations with reason codes:
identity anchor dominant, conservative concept weights, family-overlap and
conflicting-family detection, known-conflict pairs, high-risk stacking
warnings, automatic normalization above 1.60 total concept strength, and
an identity preservation score (warn < 0.75, strong warn < 0.60) with
safer-alternative suggestions. CoreShift block weights attached per item.

**Batch Variations** - sweep any slider across values x seeds (capped, 64),
identity anchor held fixed; every job stores prompt, negative, seed, resolved
stack, slider state, warnings, and the full CoreShift generation payload in
the manifest. Generation runs through the Forge adapter when live
(`concept batch --run`, resumable) - otherwise the manifest + payloads wait.

**Playground handoff** - "Send to Playground" writes a complete preset
(checkpoint, sampler/steps/CFG/clip-skip, modular prompt with identity
anchor + concept descriptors, identity-drift negative); stack and slider
presets persist in the manifest.

## Study Intelligence Layer (v3.2)

Professional study classification for artistic figure studies, fashion
editorial work, lingerie/fashion studies, and form-and-proportion studies —
identity-preserving throughout. Pure stdlib, resumable, manifest-tracked
(schema v7, `study_labels`); old manifests upgrade transparently.

```bash
lora-studio -p spookums.toml study classify          # label all curated frames
lora-studio -p spookums.toml study report            # category + score summary
lora-studio -p spookums.toml study recipes --apply   # add the 4 study recipes
lora-studio -p spookums.toml study stack --mode character_figure_study
lora-studio -p spookums.toml study presets           # 8 Playground presets
```

**Classifier** (`lora_studio/study.py`) fuses manifest signals — captions/
WD14 tags, framing, identity similarity, face count, sharpness, brightness —
into scores (`figure_study`, `fashion_study`, `lingerie_fashion`,
`pose_clarity`, `silhouette_clarity`, `garment_visibility`, `identity_lock`),
a primary category, study tags, reason codes (`clear_full_body_framing`,
`garment_structure_visible`, `lighting_quality_pass`, ...), a review status
and export eligibility. Ambiguous or sensitive frames route to
`needs_review` and are excluded from study exports until approved. Manual
overrides (Review tab "S" button or `set_manual_label`) always win and
survive re-classification. `SIGNAL_HOOKS` is the extension point for
optional local pose/aesthetic/CLIP scorers.

**Dataset recipes** (`study recipes --apply`, then build/wizard as usual):
`figure_study_v1` (40/30/20/10 full/upper/portrait/detail, identity-locked),
`fashion_editorial_v1` (wardrobe + textile detail), `lingerie_fashion_study_v1`
(export-eligible frames only, strongest identity floor),
`balanced_character_study_v1` (identity-first with study support frames).
Recipes accept `study_primary`, `study_min_confidence`, `study_export_only`.

**Training presets**: `figure_study`, `fashion_editorial`, `balanced_study`
(dim 32-48, low TE LR for identity retention, caption-dropout tuned per
goal; rationale + validation guidance in each preset's notes; validation
prompt set in `train/presets.py::STUDY_VALIDATION_PROMPTS`).

**Stack Planner modes** (`study stack --mode ...`): `character_identity`,
`character_fashion_study`, `character_figure_study`, `character_style`,
`character_fashion_style`, `character_figure_fashion_editorial` — identity
LoRA stays primary, support LoRAs secondary, merge order + conflict
warnings + CoreShift block weights included.

**Review tab**: Study filter (candidates per category, Needs Study Review,
Export Eligible, Strong Identity Lock, Strong Pose Clarity), sort by study
confidence / identity lock / figure / fashion scores, study chip with
confidence + reason codes on hover, "S" button to set a manual study label.

**Playground presets** (`study presets`): Identity-Locked Figure Study,
Fashion Editorial Study, Lingerie/Fashion Study, Character + Figure/Fashion
Consistency Tests, and Merge QA (Identity Preservation / Full-Body
Consistency / Editorial Fashion) — all tuned for CyberRealistic Pony v18.0
CoreShift with QA notes embedded.

Limitations: the baseline classifier is heuristic (caption + geometry
signals); pose estimation and aesthetic scoring land via `SIGNAL_HOOKS`
without new required dependencies.

## Base model: CyberRealistic Pony v18.0 CoreShift

The studio standard checkpoint. `lora_studio/base_models.py` auto-detects it
from `base_model` in your project file and retunes **everything**:

- Playground presets & eval matrix: **DPM++ SDE Karras, 32 steps, CFG 5,
  clip skip 2** (Forge gets clip skip via `override_settings`)
- Prompts: `score_9, score_8_up, score_7_up` + photoreal tags; photoreal
  negative (anti-cartoon/anime/3d)
- Stack Planner block weights that protect CoreShift's texture engine:
  flavor `te=0,down=0.2,mid=0.4,up=0.9` · wardrobe `te=0,down=1,mid=1,up=0.8`
  · refiner `te=0,down=0.1,mid=0.3,up=0.9`
- Sidebar **Engines** strip: live Forge / Playground / Factory dots on every
  page; click a dot to start the engine. Factory steps now stream full
  tracebacks into their output box on failure (state preserved, re-run safe).

Swap the checkpoint path and the profile follows (Pony V6 and generic SDXL
profiles included). Override per-run: `matrix --sampler --steps --cfg --clip-skip`.

## Complete End-to-End Workflow

One command, raw library to production pack:

```bash
conda activate a1111 && cd ~/DESIGNER_LORA
lora-studio -p spookums.toml ui          # the COMPLETE studio @ :7861
```

1. **Anchor** (once): drop 5-15 reference photos in `anchor_refs/`, run
   `lora-studio -p spookums.toml anchor`.
2. **Dataset Factory** (sidebar) — walk steps 1-7: ingest → analyze → stack
   plan → curate+caption → build → train + best-epoch sweep → merge & ship.
   Each step streams progress; everything is resumable.
3. **Review** anytime — thumbnail grid with per-frame and **bulk**
   keep/reject, caption editing, framing/cluster filters.
4. **Test** tab — the Forge backend (:7860) has its own Start/status badge;
   matrix and sweep use it automatically.
5. **Playground** — Presets → Refresh → load the shipped pack: base model,
   merged LoRAs, weights, trigger and sample prompts all pre-filled.

CLI equivalents for every step are listed below; terminal and UI share one
manifest, so you can mix freely.

## One command, the whole studio

```bash
lora-studio -p spookums.toml ui     # ← launches the COMPLETE studio @ :7861
```

The hub opens with a sidebar:

- **Playground** (default) — the Grokkie generation lab (unlimited LoRA stack,
  WeightEngine weighting, img2img, batch/vary, canvas), embedded and managed
  by the hub (click Start once; it runs as a child app in your Forge env).
- **Dataset Factory** — the guided 7-step flow, embedded the same way.
- **Create Wizard / Pipeline / Review / Tags / Builds / Train / Test / Lab /
  Settings** — the classic studio tools.

Deep integration: everything shares one manifest + project file. The Factory's
"Send to Playground" and the mega-pipeline write presets the Playground loads
directly (Presets → Refresh → Load). Best-Epoch Sweep results live in the
Train tab (and the Factory).

**One click, end to end:** Pipeline tab → **Run FULL STUDIO Pipeline** —
raw media → curate → caption → auto-typed build → train → best-epoch sweep →
Playground preset. CLI equivalent:

```bash
lora-studio -p spookums.toml pipeline run mega
```


Local, private, production-grade LoRA studio for Pony Diffusion V6 XL —
a full offline pipeline from raw 2TB media library to trained, merged,
eval-scored LoRAs. Core is **pure stdlib** (runs in any Python ≥3.10 env,
including your Forge conda env); every AI capability is an optional extra
with graceful degradation. M2 Max 32GB / MPS optimized throughout.

```
extract -> curate (identity + diversity) -> caption (WD14) -> build (versioned)
        -> train (kohya presets) -> eval (likeness matrix) -> merge -> model card
```

## Install / upgrade

```bash
cd ~/DESIGNER_LORA
pip install -e .                  # core (zero deps)
pip install -e '.[all]'           # ai + cluster + caption + eval extras
lora-studio init spookums.toml    # then edit paths, or use the Settings tab
lora-studio -p spookums.toml ui   # dashboard -> http://127.0.0.1:7861
```

Single-file bundle: `python -m zipapp lora_studio -m "lora_studio.cli:main" -o lora_studio.pyz`

## Creator Wizard (the Civitai-style flow, offline)

The **Create** tab (first tab in the UI) walks the whole flow:
Analyze dataset → pick type (auto-detected with explainable scores) →
trigger word → one click runs build → train → eval matrix → model card.

```bash
lora-studio -p spookums.toml analyze          # stats + type recommendation
lora-studio -p spookums.toml wizard           # auto type, full chain
lora-studio -p spookums.toml wizard --type detail --trigger spook
lora-studio -p spookums.toml wizard --no-train    # build the dataset only
```

Type detection reads the manifest only (no re-scanning): character = face
presence + identity consistency; style = scene diversity, low face
dependence; outfit = concentrated clusters + body framings; pose =
full-body share; detail = closeup share + sharpness; explicit = explicit
caption-tag rate (trains with character parameters). Each type maps to a
tuned recipe + M2-Max training preset; you can override the recommendation.

## The one-command pipeline (Phase 7)

```bash
# everything, with a review pause after curation:
lora-studio -p spookums.toml pipeline run full \
  --recipe character_v1 --preset character --gate curate

# review grids in the UI, then:
lora-studio -p spookums.toml pipeline resume
lora-studio -p spookums.toml pipeline status
```

Failed stages also pause the DAG; fix and `pipeline resume`. State survives
restarts (stored in the manifest).

## Walkthroughs per LoRA type

All five follow the same skeleton — only the recipe + preset change:

| Type | Recipe (project file) | Preset | Notes |
|---|---|---|---|
| Character | `character_v1` | `character` | quota-balanced framings, identity-filtered |
| Style | `style_v1` | `style` | TE frozen; broad sampling, low repeats |
| Outfit | `outfit_v1` | `outfit` | set `include_clusters` to the outfit's CLIP clusters |
| Pose | `pose_v1` | `pose` | full-body framing filter |
| Detail | `detail_v1` | `detail` | closeup + high sharpness floor; `smart_crop = true` |

Example — Character end-to-end, manually staged:

```bash
lora-studio -p spookums.toml extract --dry-run && lora-studio -p spookums.toml extract
lora-studio -p spookums.toml anchor --anchor-dir ~/refs_of_subject
lora-studio -p spookums.toml curate --smart
lora-studio -p spookums.toml cluster
lora-studio -p spookums.toml caption --pony-prefix --prune "$(echo see Tags tab >80% tags)"
lora-studio -p spookums.toml build --recipe character_v1
lora-studio -p spookums.toml train --dataset 1 --preset character
# eval every epoch checkpoint, pick the best:
for f in ~/LORA_OUTPUT/*character_v001*.safetensors; do
  lora-studio -p spookums.toml matrix --lora "$f" --label "$(basename $f .safetensors)"
done
lora-studio -p spookums.toml evals
```

Outfit tip: run `cluster`, open the Review tab, note which cluster IDs show
the outfit, put them in the recipe's `include_clusters`, rebuild.

## Merging Lab

```bash
# weighted merge with block control (te/down/mid/up):
lora-studio -p spookums.toml merge \
  --lora ~/LORA_OUTPUT/tok_character_v001.safetensors:1.0 \
  --lora ~/LORA_OUTPUT/tok_style_v002.safetensors:0.4 \
  --blocks "te=0,down=0.3,mid=0.5,up=1.0" --name char_styled_v1

# one-click final character (identity + style surface + outfit):
lora-studio -p spookums.toml combo \
  --character ~/LORA_OUTPUT/tok_character_v001.safetensors \
  --style ~/LORA_OUTPUT/tok_style_v002.safetensors \
  --outfit ~/LORA_OUTPUT/tok_outfit_v001.safetensors --name spook_final_v1

# merge with instant preview (likeness grid on the result):
lora-studio -p spookums.toml merge --lora A.safetensors:1.0 --lora B.safetensors:0.4 \
  --name test_mix --preview

# always verify the merge:
lora-studio -p spookums.toml matrix --lora ~/LORA_OUTPUT/spook_final_v1.safetensors
```

Concat method: mixed ranks merge correctly; negative weights subtract.
Also available in the UI **Lab** tab, next to **A/B blind compare**.

## Registry, cards, maintenance

```bash
lora-studio -p spookums.toml registry          # all LoRAs + likeness scores
lora-studio -p spookums.toml card spook_final_v1   # markdown model card
lora-studio -p spookums.toml watch --interval 300 --auto-curate  # auto-ingest
lora-studio -p spookums.toml space             # disk dashboard
lora-studio -p spookums.toml gc                # dry-run; add --apply to delete
```

## UI tabs

**Create (wizard)** · Pipeline (queue + live output) · Review (grid,
keep/reject, caption edit) · Tags (frequency, prune candidates) · Builds
(versions, diff) · Train (presets, loss curves) · Test (txt2img/img2img,
matrix) · Lab (merge + preview, A/B blind compare) · Settings.

## Tests

`pytest tests/ -v` — 48 tests, no model downloads required.
