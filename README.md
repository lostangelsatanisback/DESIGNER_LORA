# LoRA Designer Studio (v3.0)

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
