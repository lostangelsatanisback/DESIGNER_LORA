#!/usr/bin/env python3
"""
grokkie_dataset_factory.py — Grokkie Dataset Factory
====================================================
Guided 7-step flow: raw personal media library -> production LoRA stack,
pre-loaded into the Grokkie Playground. The Civitai creator experience,
fully local.

    Step 1  Ingest        extraction with live progress (resumable)
    Step 2  Analyze       face rate / identity / framing / diversity dashboard
    Step 3  Stack Plan    type detection + recommended multi-LoRA stack
    Step 4  Curate        identity filtering + clustering + captioning
    Step 5  Recipe+Build  visual recipe -> versioned dataset
    Step 6  Train+Sweep   kohya preset run + BEST-EPOCH SWEEP
    Step 7  Merge & Ship  block-weight merge -> "Send to Grokkie Playground"

All heavy lifting is delegated to the verified `lora_studio` package; the
WeightEngine (if present) prices the final Playground weights. Every import
is guarded — missing pieces disable their step with a clear message instead
of crashing.

Run next to the lora_studio package (and optionally your grokkie modules):

    python grokkie_dataset_factory.py --project spookums.toml --port 7875
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("factory")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _try(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:                                    # noqa: BLE001
        logger.info("optional module '%s' unavailable: %s", name, exc)
        return None


# lora_studio is the engine room (required)
try:
    from lora_studio import manifest
    from lora_studio.config import (CaptionConfig, ClusterConfig, CurateConfig,
                                    ExtractConfig, SmartCurateConfig,
                                    load_project)
    from lora_studio.wizard import (analyze, detect_type, make_recipe,
                                    suggest_stack, TYPE_TEMPLATES)
    HAVE_STUDIO = True
except Exception as exc:                                        # noqa: BLE001
    HAVE_STUDIO = False
    _STUDIO_ERR = str(exc)

G_WEIGHTS = _try("grokkie_weight_engine")
G_SLIDERS = _try("grokkie_weight_sliders")

PLAYGROUND_PRESETS = HERE / "outputs" / "playground_presets.json"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class FactorySession:
    """Carries project + step state across the guided flow."""

    def __init__(self, project_path: Optional[str]) -> None:
        self.project_path = project_path
        self.prj = load_project(project_path) if HAVE_STUDIO else None
        self.analysis: dict = {}
        self.ranking: list[dict] = []
        self.plan: dict = {}
        self.built_versions: dict[str, int] = {}     # type -> dataset version
        self.run_names: dict[str, str] = {}          # type -> run name
        self.best_epochs: dict[str, dict] = {}       # type -> sweep best
        self.merged_path: str = ""

    @property
    def base(self) -> Path:
        return self.prj.output_path

    def conn(self):
        return manifest.connect(self.base)


SESSION: Optional[FactorySession] = None


def get_session(project_path: Optional[str] = None) -> FactorySession:
    global SESSION
    if SESSION is None:
        SESSION = FactorySession(project_path)
    return SESSION


def _stream(gen):
    """Stream a lora_studio generator into a growing transcript string."""
    lines: list[str] = []
    for update in gen:
        lines.append(str(update))
        yield "\n".join(lines[-40:])


# ---------------------------------------------------------------------------
# Step handlers (UI-framework-free; testable)
# ---------------------------------------------------------------------------

def step_ingest(sess: FactorySession, dry_run: bool):
    cfg = ExtractConfig(output_base=sess.base, dry_run=bool(dry_run))
    from lora_studio.extract import pipeline_generator
    yield from _stream(pipeline_generator(sess.prj.video_dirs,
                                          sess.prj.photos_dir, cfg))


def step_analyze(sess: FactorySession) -> tuple[str, str]:
    conn = sess.conn()
    sess.analysis = analyze(conn)
    sess.ranking = detect_type(sess.analysis)
    a = sess.analysis
    dash = (
        f"| metric | value |\n|---|---|\n"
        f"| curated frames | {a['selected']} |\n"
        f"| face rate | {a['face_rate']:.0%} |\n"
        f"| identity mean / std | {a['id_mean']} / "
        f"{a['id_std'] if a['id_std'] is None else round(a['id_std'], 3)} |\n"
        f"| framing mix | {a['framing_mix']} |\n"
        f"| clusters (entropy) | {a['clusters']} ({a['cluster_entropy']}) |\n"
        f"| explicit-tag rate | {a['explicit_rate']:.0%} |\n"
        f"| sharpness mean | {a['sharpness_mean']} |\n"
    )
    types = "\n".join(
        f"- **{r['type']}** — {r['score']:.2f} ({r['reason']})"
        for r in sess.ranking
    )
    return dash, types


def step_plan(sess: FactorySession) -> str:
    if not sess.ranking:
        return "Run Analyze first."
    sess.plan = suggest_stack(sess.ranking, sess.analysis)
    lines = ["### Recommended production stack\n",
             "| LoRA | role | playground weight | merge blocks |", "|---|---|---|---|"]
    for s in sess.plan["stack"]:
        lines.append(f"| {s['type']} | {s['role']} | {s['weight']} | {s['blocks']} |")
    lines.append("\n**Why:** " + " · ".join(sess.plan["rationale"]))
    lines.append(f"\nMerged pack name: `{sess.plan['merge_name']}`")
    return "\n".join(lines)


def step_curate(sess: FactorySession, smart: bool, cluster: bool, caption: bool):
    from lora_studio.curate import curate_generator, smart_curate_generator

    def chain():
        yield from curate_generator(CurateConfig(output_base=sess.base))
        if smart:
            yield from smart_curate_generator(SmartCurateConfig(output_base=sess.base))
        if cluster:
            from lora_studio.curate.diversity import cluster_generator
            yield from cluster_generator(ClusterConfig(output_base=sess.base))
        if caption:
            from lora_studio.caption import caption_generator
            yield from caption_generator(CaptionConfig(
                output_base=sess.base, trigger=sess.prj.trigger_token,
                class_word=sess.prj.class_word))
    yield from _stream(chain())


def step_build(sess: FactorySession, lora_type: str, repeats: int,
               max_total: int, quota: str, val_fraction: float,
               smart_crop: bool):
    recipe = make_recipe(lora_type, sess.prj.trigger_token, sess.prj.class_word)
    recipe.update(repeats=int(repeats), max_total=int(max_total),
                  quota=quota, val_fraction=float(val_fraction),
                  smart_crop=bool(smart_crop))
    rname = f"factory_{lora_type}"
    sess.prj.recipes[rname] = recipe
    from lora_studio.builder import BuildConfig, build_generator
    transcript = ""
    for chunk in _stream(build_generator(
            sess.prj, BuildConfig(output_base=sess.base, recipe=rname,
                                  note="dataset factory"))):
        transcript = chunk
        yield transcript
    row = sess.conn().execute("SELECT MAX(version) FROM datasets").fetchone()
    if row and row[0]:
        sess.built_versions[lora_type] = int(row[0])
        yield transcript + f"\n\n[factory] dataset v{row[0]:03d} registered for '{lora_type}'"


def step_train_sweep(sess: FactorySession, lora_type: str, do_sweep: bool,
                     backend: str):
    version = sess.built_versions.get(lora_type)
    if version is None:
        conn = sess.conn()
        row = conn.execute("SELECT MAX(version) FROM datasets").fetchone()
        version = int(row[0]) if row and row[0] else None
    if version is None:
        yield "Build a dataset first (Step 5)."
        return
    preset = TYPE_TEMPLATES[lora_type]["preset"]
    name = f"{sess.prj.trigger_token}_{lora_type}_v{version:03d}"
    sess.run_names[lora_type] = name

    from lora_studio.train.kohya import TrainConfig, train_generator

    def chain():
        yield from train_generator(sess.prj, TrainConfig(
            output_base=sess.base, dataset_version=version,
            preset=preset, name=name))
        if do_sweep:
            from lora_studio.sweep import SweepConfig, sweep_generator
            yield from sweep_generator(sess.prj, SweepConfig(
                output_base=sess.base, run=name, backend=backend))
    transcript = ""
    for chunk in _stream(chain()):
        transcript = chunk
        yield transcript
    if do_sweep:
        from lora_studio.sweep import sweep_summary
        s = sweep_summary(sess.conn(), name)
        if s and s.get("best"):
            sess.best_epochs[lora_type] = s["best"]
            yield transcript + (f"\n\n[factory] best epoch for {lora_type}: "
                                f"{s['best']['epoch']} ({s['best']['label']})")


def _checkpoint_for(sess: FactorySession, lora_type: str) -> Optional[Path]:
    lora_dir = Path(sess.prj.lora_output_dir
                    or (sess.base / "LORA_OUTPUT")).expanduser()
    best = sess.best_epochs.get(lora_type)
    if best:
        p = lora_dir / f"{best['label']}.safetensors"
        if p.exists():
            return p
    run = sess.run_names.get(lora_type)
    if run:
        cands = sorted(lora_dir.glob(f"{run}*.safetensors"))
        if cands:
            return cands[-1]
    return None


def step_merge(sess: FactorySession, name: str):
    if not sess.plan:
        yield "Run Stack Plan first (Step 3)."
        return
    name = name or sess.plan.get("merge_name", "factory_pack_v1")
    loras: list[tuple[str, float]] = []
    block_weights: dict = {}
    missing = []
    for s in sess.plan["stack"]:
        p = _checkpoint_for(sess, s["type"])
        if p is None:
            missing.append(s["type"])
            continue
        loras.append((str(p), s["weight"]))
        block_weights[str(p)] = s["blocks"]
    if missing:
        yield (f"Missing trained checkpoints for: {missing}. "
               "Run Step 6 for each, or proceed with what exists.")
    if len(loras) < 2:
        if len(loras) == 1:
            sess.merged_path = loras[0][0]
            yield (f"Only one LoRA in the stack - nothing to merge. "
                   f"Using {sess.merged_path} directly; go to Ship.")
        return
    from lora_studio.merge import MergeConfig, merge_generator
    transcript = ""
    for chunk in _stream(merge_generator(sess.prj, MergeConfig(
            output_base=sess.base, loras=loras, output_name=name,
            block_weights=block_weights))):
        transcript = chunk
        yield transcript
    lora_dir = Path(sess.prj.lora_output_dir
                    or (sess.base / "LORA_OUTPUT")).expanduser()
    from lora_studio.util import safe_slug
    merged = lora_dir / f"{safe_slug(name)}.safetensors"
    if merged.exists():
        sess.merged_path = str(merged)
        yield transcript + f"\n\n[factory] merged pack ready: {merged}"


def step_ship(sess: FactorySession, preset_name: str,
              presets_path: str = "") -> str:
    """Write a Grokkie Playground preset pre-loading the production stack."""
    target = Path(presets_path or PLAYGROUND_PRESETS).expanduser()
    stack: list[tuple[str, float]] = []

    if sess.merged_path:
        stack.append((Path(sess.merged_path).stem, 0.85))
    else:
        for s in sess.plan.get("stack", []):
            p = _checkpoint_for(sess, s["type"])
            if p is not None:
                stack.append((p.stem, s["weight"]))
    if not stack:
        return "Nothing to ship - merge or train first."

    # let the WeightEngine refine the weights when available
    if G_WEIGHTS is not None and len(stack) > 1:
        try:
            eng = G_WEIGHTS.get_weight_engine("SDXL")
            cands = [{"name": n, "category": "unknown", "confidence": 0.7}
                     for n, _ in stack]
            assignments = eng.compute_weights(
                f"{sess.prj.trigger_token} {sess.prj.class_word}", cands)
            refined = {a.lora_name: a.fused_weight for a in assignments}
            stack = [(n, refined.get(n, w)) for n, w in stack]
        except Exception:                                       # noqa: BLE001
            pass

    preset = {
        "checkpoint": None,                       # keep playground's selection
        "prompt": f"score_9, score_8_up, {sess.prj.trigger_token} "
                  f"{sess.prj.class_word}, ",
        "negative": ("lowres, bad anatomy, bad hands, deformed, blurry, "
                     "watermark, worst quality"),
        "sampler": "DPM++ 2M Karras", "steps": 30, "cfg": 7.0,
        "width": 1024, "height": 1024,
        "loras": [[n, round(float(w), 2)] for n, w in stack],
        "_factory": {"project": sess.project_path, "shipped_at": time.time(),
                     "best_epochs": sess.best_epochs},
    }
    presets = {}
    try:
        presets = json.loads(target.read_text())
    except Exception:
        pass
    pname = preset_name or f"factory_{sess.prj.trigger_token}"
    presets[pname] = preset
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(presets, indent=2))
    return (f"Shipped preset '{pname}' -> {target}\n"
            f"Stack: {preset['loras']}\n"
            "Open Grokkie Playground -> Presets -> Load. "
            "(LoRA files must be visible in the playground's LoRA folders.)")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui(project_path: Optional[str]):
    import gradio as gr
    if not HAVE_STUDIO:
        raise RuntimeError(f"lora_studio package not importable: {_STUDIO_ERR}")
    sess = get_session(project_path)
    types = list(TYPE_TEMPLATES)

    with gr.Blocks(title="Grokkie Dataset Factory") as demo:
        gr.Markdown(f"# Grokkie Dataset Factory\nproject: "
                    f"`{project_path or '(defaults)'}` · output: `{sess.base}` · "
                    f"WeightEngine: {'on' if G_WEIGHTS else 'off'}")

        with gr.Tab("1 · Ingest"):
            dry = gr.Checkbox(True, label="Dry run first (recommended)")
            b1 = gr.Button("Ingest library", variant="primary")
            o1 = gr.Textbox(lines=18, label="Progress")
            b1.click(lambda d: step_ingest(sess, d), inputs=dry, outputs=o1)

        with gr.Tab("2 · Analyze"):
            b2 = gr.Button("Analyze dataset", variant="primary")
            dash = gr.Markdown()
            tdet = gr.Markdown()
            b2.click(lambda: step_analyze(sess), outputs=[dash, tdet])

        with gr.Tab("3 · Stack Plan"):
            b3 = gr.Button("Suggest production stack", variant="primary")
            plan_md = gr.Markdown()
            b3.click(lambda: step_plan(sess), outputs=plan_md)

        with gr.Tab("4 · Curate & Caption"):
            c_smart = gr.Checkbox(True, label="Identity filtering ([ai])")
            c_clust = gr.Checkbox(True, label="CLIP clustering ([cluster])")
            c_capt = gr.Checkbox(True, label="WD14 captioning ([caption])")
            b4 = gr.Button("Run curation chain", variant="primary")
            o4 = gr.Textbox(lines=18, label="Progress")
            b4.click(lambda s, cl, cp: step_curate(sess, s, cl, cp),
                     inputs=[c_smart, c_clust, c_capt], outputs=o4)
            gr.Markdown("Review/override frames and captions in the LoRA Studio "
                        "UI (`lora-studio ui`) - same manifest, live.")

        with gr.Tab("5 · Recipe & Build"):
            r_type = gr.Dropdown(types, value="character", label="LoRA type")
            with gr.Row():
                r_rep = gr.Slider(2, 20, 10, step=1, label="Repeats")
                r_max = gr.Slider(0, 1000, 400, step=10, label="Max images (0=all)")
                r_val = gr.Slider(0.0, 0.2, 0.05, step=0.01, label="Val fraction")
            r_quota = gr.Textbox(
                value="closeup=0.30,portrait=0.30,upper_body=0.25,full_body=0.15",
                label="Framing quota (blank = off)")
            r_crop = gr.Checkbox(False, label="Subject-centered SDXL smart-crop")
            b5 = gr.Button("Build versioned dataset", variant="primary")
            o5 = gr.Textbox(lines=16, label="Progress")
            b5.click(lambda t, rp, mx, q, vf, sc:
                     step_build(sess, t, rp, mx, q, vf, sc),
                     inputs=[r_type, r_rep, r_max, r_quota, r_val, r_crop],
                     outputs=o5)

        with gr.Tab("6 · Train + Best-Epoch Sweep"):
            t_type = gr.Dropdown(types, value="character", label="LoRA type")
            t_sweep = gr.Checkbox(True, label="Best-Epoch Sweep after training")
            t_backend = gr.Dropdown(["forge", "diffusers"], value="forge",
                                    label="Sweep backend")
            b6 = gr.Button("Train + sweep", variant="primary")
            o6 = gr.Textbox(lines=18, label="Progress")
            b6.click(lambda t, s, b: step_train_sweep(sess, t, s, b),
                     inputs=[t_type, t_sweep, t_backend], outputs=o6)

        with gr.Tab("7 · Merge & Ship"):
            m_name = gr.Textbox(label="Pack name (blank = from plan)")
            b7a = gr.Button("Merge production stack (best epochs)",
                            variant="primary")
            o7 = gr.Textbox(lines=12, label="Merge progress")
            b7a.click(lambda n: step_merge(sess, n), inputs=m_name, outputs=o7)
            gr.Markdown("---")
            s_name = gr.Textbox(label="Playground preset name")
            s_path = gr.Textbox(value=str(PLAYGROUND_PRESETS),
                                label="Playground presets file")
            b7b = gr.Button("Send to Grokkie Playground", variant="primary")
            o7b = gr.Markdown()
            b7b.click(lambda n, p: step_ship(sess, n, p),
                      inputs=[s_name, s_path], outputs=o7b)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Grokkie Dataset Factory")
    parser.add_argument("--project", "-p", default=None)
    parser.add_argument("--port", type=int, default=7875)
    args = parser.parse_args()
    demo = build_ui(args.project)
    demo.launch(server_name="127.0.0.1", server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
