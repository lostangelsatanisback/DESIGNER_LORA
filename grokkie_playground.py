#!/usr/bin/env python3
"""
grokkie_playground.py — Grokkie Playground Studio
==================================================
Unified local generation studio: base model + unlimited LoRA stack with the
Grokkie WeightEngine/SliderSession system, full txt2img / img2img / inpaint /
outpaint, batch + variations + upscale. Local Civitai/Midjourney-style flow,
MPS-first.

Drop this file NEXT TO your grokkie_*.py modules and run:

    python grokkie_playground.py                 # http://127.0.0.1:7870
    python grokkie_playground.py --port 7871

Design notes
------------
* PlaygroundPipeline is a SUPERSET of RedresserPipeline's interface, so the
  existing BatchEngine / CanvasEngine plug in unchanged — and it fixes the
  gap where BatchEngine.vary_image() calls load_reference_image(), which the
  current RedresserPipeline doesn't implement.
* Every grokkie module import is guarded: missing modules degrade their
  feature with a visible status instead of crashing the app.
* MPS hygiene follows the Redresser patterns: fp16, attention slicing, VAE
  tiling, CPU-seeded generators, explicit unload with gc + mps.empty_cache.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("playground")

GROKKIE_DIR = Path(__file__).resolve().parent
if str(GROKKIE_DIR) not in sys.path:
    sys.path.insert(0, str(GROKKIE_DIR))


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:                                    # noqa: BLE001
        logger.info("optional module '%s' unavailable: %s", name, exc)
        return None


G_CONFIG = _try_import("grokkie_config")
G_WEIGHTS = _try_import("grokkie_weight_engine")
G_SLIDERS = _try_import("grokkie_weight_sliders")
G_BATCH = _try_import("grokkie_batch")
G_CANVAS = _try_import("grokkie_canvas")
G_PRESETS = _try_import("grokkie_presets")

# ---------------------------------------------------------------------------
# Paths / discovery
# ---------------------------------------------------------------------------

MODEL_EXTS = {".safetensors", ".ckpt"}
LORA_EXTS = {".safetensors"}

EXTRA_MODEL_DIRS = [
    "/Users/snoozeybnuiadmin/FORGE_REPO/stable-diffusion-webui-reForge/models/Stable-diffusion",
]
EXTRA_LORA_DIRS = [
    "/Users/snoozeybnuiadmin/FORGE_REPO/stable-diffusion-webui-reForge/models/Lora",
]


def _config_dir(attr: str, fallback: str) -> Path:
    if G_CONFIG is not None and hasattr(G_CONFIG, attr):
        return Path(getattr(G_CONFIG, attr))
    return GROKKIE_DIR / fallback


MODELS_DIR = _config_dir("MODELS_DIR", "models")
LORA_ROOT = _config_dir("LORA_ROOT", "loras")
OUTPUT_DIR = _config_dir("OUTPUT_DIR", "outputs")
PRESETS_PATH = OUTPUT_DIR / "playground_presets.json"


def detect_base_type(path: Path) -> str:
    """Heuristic base-model detection from filename + size."""
    name = path.name.lower()
    if "flux" in name:
        return "FLUX"
    if "pony" in name:
        return "PONY"
    if "xl" in name or "sdxl" in name:
        return "SDXL"
    try:
        if path.is_file() and path.stat().st_size > 4_500_000_000:
            return "SDXL"
    except OSError:
        pass
    return "SD15"


class ModelScanner:
    def __init__(self) -> None:
        self.checkpoints: dict[str, Path] = {}
        self.loras: dict[str, Path] = {}

    def scan(self) -> "ModelScanner":
        self.checkpoints.clear()
        self.loras.clear()
        for d in [MODELS_DIR, *map(Path, EXTRA_MODEL_DIRS)]:
            if d.exists():
                for f in sorted(d.rglob("*")):
                    if f.suffix.lower() in MODEL_EXTS and f.is_file():
                        label = f"{f.stem}  [{detect_base_type(f)}]"
                        self.checkpoints[label] = f
        for d in [LORA_ROOT, *map(Path, EXTRA_LORA_DIRS)]:
            if d.exists():
                for f in sorted(d.rglob("*")):
                    if f.suffix.lower() in LORA_EXTS and f.is_file():
                        self.loras[f.stem] = f
        logger.info("scan: %d checkpoints, %d LoRAs",
                    len(self.checkpoints), len(self.loras))
        return self

    def checkpoint_path(self, label: str) -> Optional[Path]:
        return self.checkpoints.get(label)

    def lora_pool(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.loras.items()}


SCANNER = ModelScanner()


def guess_category(name: str) -> str:
    n = name.lower()
    for cat, keys in {
        "character": ("char", "person", "face", "identity"),
        "style": ("style", "aesthetic", "art"),
        "outfit": ("outfit", "cloth", "dress", "wear"),
        "pose": ("pose", "position"),
        "detail": ("detail", "skin", "texture", "enhance", "refine"),
    }.items():
        if any(k in n for k in keys):
            return cat
    return "unknown"


# ---------------------------------------------------------------------------
# PlaygroundPipeline (Redresser-compatible superset)
# ---------------------------------------------------------------------------

@dataclass
class GenResult:
    images: list = field(default_factory=list)
    seed_used: int = -1
    generation_time_seconds: float = 0.0
    peak_memory_gb: float = 0.0
    success: bool = True
    error: str = ""
    notes: list = field(default_factory=list)


def _mps_flush() -> None:
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


class PlaygroundPipeline:
    """txt2img + img2img + inpaint with stacked LoRAs (negative weights OK).

    Interface superset of RedresserPipeline so BatchEngine / CanvasEngine
    work unchanged: load_checkpoint, load_loras, generate, peak_gb,
    load_reference_image.
    """

    SAMPLERS = ["default", "Euler a", "DPM++ 2M Karras", "DDIM", "UniPC"]

    def __init__(self, device: str = "auto") -> None:
        import torch
        if device == "auto":
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = torch.float16 if device != "cpu" else torch.float32
        self._pipe = None
        self.base_model = "SDXL"
        self.checkpoint_path = ""
        self.adapters: list[tuple[str, float]] = []
        self._reference_image = None
        self._peak = 0.0

    # ----- checkpoint -----

    def load_checkpoint(self, path: str, base_model: str = "auto") -> dict:
        ckpt = Path(path).expanduser()
        if not ckpt.exists():
            return {"loaded": False, "error": f"Checkpoint not found: {ckpt}"}
        if str(ckpt) == self.checkpoint_path and self._pipe is not None:
            return {"loaded": True, "base_model": self.base_model, "cached": True}

        self.unload()
        detected = detect_base_type(ckpt) if base_model == "auto" else base_model.upper()
        if detected == "FLUX":
            return {"loaded": False,
                    "error": "FLUX checkpoints need the flux pipeline (not wired yet); "
                             "pick an SDXL/Pony model."}
        try:
            import torch
            from diffusers import (AutoPipelineForText2Image,
                                   StableDiffusionXLPipeline,
                                   StableDiffusionPipeline)
            if ckpt.is_file():
                cls = (StableDiffusionXLPipeline if detected in ("SDXL", "PONY")
                       else StableDiffusionPipeline)
                self._pipe = cls.from_single_file(
                    str(ckpt), torch_dtype=self.dtype, use_safetensors=True,
                )
            else:
                self._pipe = AutoPipelineForText2Image.from_pretrained(
                    str(ckpt), torch_dtype=self.dtype,
                )
            self._pipe.to(self.device)
            self._apply_optimizations()
            self.checkpoint_path = str(ckpt)
            self.base_model = detected
            self.adapters = []
            logger.info("loaded %s [%s] on %s", ckpt.name, detected, self.device)
            return {"loaded": True, "base_model": detected}
        except Exception as exc:                                # noqa: BLE001
            logger.exception("checkpoint load failed")
            return {"loaded": False, "error": str(exc)}

    def _apply_optimizations(self) -> None:
        for fn in ("enable_attention_slicing", "enable_vae_tiling"):
            try:
                getattr(self._pipe, fn)()
            except Exception:
                pass

    # ----- LoRA stack -----

    def load_loras(self, lora_names: list[str], weights: list[float],
                   lora_pool: dict[str, str]) -> dict:
        if self._pipe is None:
            return {"ok": False, "error": "load a checkpoint first"}
        try:
            self._pipe.unload_lora_weights()
        except Exception:
            pass
        self.adapters = []
        loaded, failed = [], []
        for name, weight in zip(lora_names, weights):
            path = lora_pool.get(name, name)
            if self._load_one_lora(path, float(weight)):
                loaded.append(f"{Path(path).stem}:{weight:g}")
            else:
                failed.append(Path(path).stem)
        return {"ok": True, "loaded": loaded, "failed": failed}

    def _load_one_lora(self, path: str, strength: float) -> bool:
        lp = Path(path).expanduser()
        adapter = lp.stem
        for ch in (".", " ", "-", "%20"):
            adapter = adapter.replace(ch, "_")
        if any(n == adapter for n, _ in self.adapters):
            adapter = f"{adapter}_{len(self.adapters)}"
        # 3-tier fallback: standard -> direct file -> UNet-only
        for attempt in range(3):
            try:
                if attempt == 0:
                    self._pipe.load_lora_weights(
                        str(lp.parent), weight_name=lp.name, adapter_name=adapter)
                elif attempt == 1:
                    self._pipe.load_lora_weights(str(lp), adapter_name=adapter)
                else:
                    from safetensors.torch import load_file
                    state = load_file(str(lp))
                    state = {k: v for k, v in state.items()
                             if "text_encoder" not in k}
                    self._pipe.load_lora_weights(state, adapter_name=adapter)
                self.adapters.append((adapter, strength))
                names = [n for n, _ in self.adapters]
                ws = [w for _, w in self.adapters]   # negative weights pass through
                self._pipe.set_adapters(names, adapter_weights=ws)
                return True
            except Exception as exc:                            # noqa: BLE001
                logger.info("LoRA tier %d failed for %s: %s", attempt, lp.name, exc)
        return False

    # ----- reference image (fixes BatchEngine.vary_image) -----

    def load_reference_image(self, image) -> None:
        self._reference_image = image

    # ----- samplers -----

    def set_sampler(self, name: str) -> None:
        if self._pipe is None or name in ("", "default"):
            return
        try:
            from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler,
                                   EulerAncestralDiscreteScheduler, UniPCMultistepScheduler)
            cfgs = self._pipe.scheduler.config
            if name == "Euler a":
                self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(cfgs)
            elif name == "DPM++ 2M Karras":
                self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    cfgs, use_karras_sigmas=True)
            elif name == "DDIM":
                self._pipe.scheduler = DDIMScheduler.from_config(cfgs)
            elif name == "UniPC":
                self._pipe.scheduler = UniPCMultistepScheduler.from_config(cfgs)
        except Exception as exc:                                # noqa: BLE001
            logger.info("sampler '%s' unavailable: %s", name, exc)

    # ----- generation -----

    def generate(self, **kw) -> GenResult:
        """Redresser-compatible. Modes:
        - txt2img: no init image
        - img2img: init image (or stored reference) + denoising_strength
        - inpaint: init image + mask
        """
        start = time.time()
        if self._pipe is None:
            return GenResult(success=False, error="Pipeline not loaded")

        prompt = kw.get("prompt", "") or ""
        if (self.base_model == "PONY" and kw.get("pony_tags", True)
                and "score_" not in prompt):
            prompt = "score_9, score_8_up, score_7_up, " + prompt

        init = kw.get("init_image") or kw.get("input_image") or self._reference_image
        mask = kw.get("mask_image")
        seed = int(kw.get("seed", -1))
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        n_images = max(1, int(kw.get("num_images", 1)))

        try:
            import torch
            self.set_sampler(kw.get("sampler", "default"))
            images = []
            for i in range(n_images):
                gen = torch.Generator("cpu").manual_seed(seed + i)
                common = dict(
                    prompt=prompt,
                    negative_prompt=kw.get("negative_prompt", "") or "",
                    num_inference_steps=int(kw.get("steps", 30)),
                    guidance_scale=float(kw.get("cfg", 7.0)),
                    generator=gen,
                )
                if init is not None and mask is not None:
                    from diffusers import AutoPipelineForInpainting
                    pipe = AutoPipelineForInpainting.from_pipe(self._pipe)
                    out = pipe(image=init, mask_image=mask,
                               strength=float(kw.get("denoising_strength", 0.85)),
                               width=int(kw.get("width", 1024)),
                               height=int(kw.get("height", 1024)), **common)
                elif init is not None:
                    from diffusers import AutoPipelineForImage2Image
                    pipe = AutoPipelineForImage2Image.from_pipe(self._pipe)
                    out = pipe(image=init,
                               strength=float(kw.get("denoising_strength", 0.6)),
                               **common)
                else:
                    out = self._pipe(width=int(kw.get("width", 1024)),
                                     height=int(kw.get("height", 1024)), **common)
                images.extend(out.images)
            return GenResult(
                images=images, seed_used=seed,
                generation_time_seconds=round(time.time() - start, 2),
                peak_memory_gb=self.peak_gb(), success=True,
                notes=[f"{self.base_model} | {len(self.adapters)} LoRAs | {self.device}"],
            )
        except Exception as exc:                                # noqa: BLE001
            logger.exception("generation failed")
            return GenResult(success=False, error=str(exc))
        finally:
            _mps_flush()

    def peak_gb(self) -> float:
        try:
            import torch
            if torch.backends.mps.is_available():
                return round(torch.mps.current_allocated_memory() / 1024**3, 2)
        except Exception:
            pass
        return 0.0

    def unload(self) -> str:
        self._pipe = None
        self.adapters = []
        self.checkpoint_path = ""
        _mps_flush()
        return "Pipeline unloaded - memory freed."


_PIPELINE: Optional[PlaygroundPipeline] = None


def get_pipeline() -> PlaygroundPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = PlaygroundPipeline()
    return _PIPELINE


# ---------------------------------------------------------------------------
# Weight system bridge
# ---------------------------------------------------------------------------

class _SimpleWA:
    """Stand-in WeightAssignment when grokkie_weight_engine is absent."""

    def __init__(self, name: str, path: str, weight: float, primary: bool, rank: int):
        self.lora_name = name
        self.lora_path = path
        self.fused_weight = weight
        self.is_primary = primary
        self.rank = rank
        self.category = guess_category(name)
        self.rationale = "uniform fallback (WeightEngine unavailable)"


class WeightPanel:
    """Bridges selected LoRAs -> WeightEngine -> SliderSession."""

    def __init__(self) -> None:
        self.session = None

    def suggest(self, prompt: str, lora_names: list[str]) -> tuple[dict, str]:
        """Returns ({name: weight}, budget_text)."""
        pool = SCANNER.lora_pool()
        names = [n for n in lora_names if n]
        if not names:
            self.session = None
            return {}, "No LoRAs selected."

        if G_WEIGHTS is not None:
            engine = G_WEIGHTS.get_weight_engine("SDXL")
            cands = [{"name": n, "category": guess_category(n),
                      "confidence": 0.75 if i == 0 else 0.6}
                     for i, n in enumerate(names)]
            assignments = engine.compute_weights(prompt, cands)
            for wa in assignments:                      # attach paths for UI
                wa.lora_path = pool.get(wa.lora_name, "")
        else:
            base = 0.85
            assignments = [
                _SimpleWA(n, pool.get(n, ""), round(max(0.3, base - 0.15 * i), 2),
                          i == 0, i)
                for i, n in enumerate(names)
            ]

        if G_SLIDERS is not None:
            self.session = G_SLIDERS.SliderSession.create(assignments, weight_cap=2.5)
            weights = self.session.get_final_weights()
            return weights, self.budget_text()
        weights = {a.lora_name: a.fused_weight for a in assignments}
        return weights, self._fallback_budget(weights)

    def adjust(self, name: str, value: float) -> tuple[dict, str]:
        if self.session is not None:
            self.session.adjust(name, float(value))
            return self.session.get_final_weights(), self.budget_text()
        return {}, ""

    def budget_text(self) -> str:
        if self.session is None:
            return ""
        try:
            b = self.session.get_budget_status()
            d = b.to_dict() if hasattr(b, "to_dict") else dict(b.__dict__)
            used = d.get("total_weight", d.get("used", 0))
            cap = d.get("weight_cap", d.get("cap", 2.5))
            state = d.get("status", "ok")
            bar = "#" * int(min(20, (used / max(cap, 0.01)) * 20))
            return (f"**Weight budget** `[{bar:<20}]` "
                    f"{used:.2f} / {cap:.1f} — {state}")
        except Exception:
            return ""

    @staticmethod
    def _fallback_budget(weights: dict) -> str:
        used = sum(abs(w) for w in weights.values())
        bar = "#" * int(min(20, used / 2.5 * 20))
        return f"**Weight budget** `[{bar:<20}]` {used:.2f} / 2.5 (fallback)"


PANEL = WeightPanel()

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def load_presets() -> dict:
    try:
        return json.loads(PRESETS_PATH.read_text())
    except Exception:
        return {}


def save_preset(name: str, data: dict) -> str:
    presets = load_presets()
    presets[name] = data
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRESETS_PATH.write_text(json.dumps(presets, indent=2))
    return f"Saved preset '{name}' ({len(presets)} total)"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

LORA_SLOTS = 10
DEFAULT_NEG = ("lowres, bad anatomy, bad hands, deformed, blurry, watermark, "
               "text, signature, worst quality")


def build_ui():
    import gradio as gr

    # ---- gradio 3.x (A1111 pins 3.41) vs 4.x compatibility ----
    GR4 = hasattr(gr, "ImageEditor")

    def make_mask_editor(label):
        if GR4:
            return gr.ImageEditor(type="pil", label=label)
        # gradio 3.x: sketch tool returns {"image": PIL, "mask": PIL}
        return gr.Image(type="pil", tool="sketch", label=label)

    def theme_kwargs():
        try:
            return {"theme": gr.themes.Base()}
        except Exception:
            return {}

    SCANNER.scan()
    ckpt_labels = list(SCANNER.checkpoints) or ["(no checkpoints found)"]
    lora_names = [""] + sorted(SCANNER.loras)

    # ---------- handlers ----------

    def rescan():
        SCANNER.scan()
        return (gr.update(choices=list(SCANNER.checkpoints) or ["(none)"]),
                *[gr.update(choices=[""] + sorted(SCANNER.loras))
                  for _ in range(LORA_SLOTS)],
                f"Rescanned: {len(SCANNER.checkpoints)} checkpoints, "
                f"{len(SCANNER.loras)} LoRAs")

    def collect_stack(*slot_vals):
        """slot_vals = name1..nameN, w1..wN, en1..enN -> (names, weights)."""
        n = LORA_SLOTS
        names, weights = [], []
        for i in range(n):
            name = slot_vals[i]
            w = slot_vals[n + i]
            enabled = slot_vals[2 * n + i]
            if name and enabled:
                names.append(name)
                weights.append(float(w))
        return names, weights

    def suggest_weights(prompt, *slot_vals):
        names, _ = collect_stack(*slot_vals)
        weights, budget = PANEL.suggest(prompt, names)
        updates = []
        n = LORA_SLOTS
        for i in range(n):
            name = slot_vals[i]
            if name and name in weights:
                updates.append(gr.update(value=round(weights[name], 2)))
            else:
                updates.append(gr.update())
        return (*updates, budget or "WeightEngine suggestions applied.")

    def do_generate(ckpt_label, prompt, negative, sampler, steps, cfg, seed,
                    width, height, n_images, init_img, strength, pony_tags,
                    *slot_vals):
        try:
            path = SCANNER.checkpoint_path(ckpt_label)
            if path is None:
                return None, "Pick a checkpoint (Rescan if list is empty)."
            pp = get_pipeline()
            res = pp.load_checkpoint(str(path))
            if not res.get("loaded"):
                return None, f"Checkpoint error: {res.get('error')}"
            names, weights = collect_stack(*slot_vals)
            lres = pp.load_loras(names, weights, SCANNER.lora_pool())
            lnote = (f"LoRAs: {', '.join(lres.get('loaded', [])) or 'none'}"
                     + (f" | FAILED: {lres['failed']}" if lres.get("failed") else ""))
            out = pp.generate(
                prompt=prompt, negative_prompt=negative, sampler=sampler,
                steps=steps, cfg=cfg, seed=seed, width=width, height=height,
                num_images=n_images, init_image=init_img,
                denoising_strength=strength, pony_tags=pony_tags,
            )
            if not out.success:
                return None, f"Generation failed: {out.error}"
            status = (f"seed {out.seed_used} | {out.generation_time_seconds}s | "
                      f"peak {out.peak_memory_gb} GB | {lnote}")
            return out.images, status
        except Exception as exc:                                # noqa: BLE001
            logger.exception("do_generate")
            return None, f"Error: {exc}"

    def do_unload():
        return get_pipeline().unload()

    def do_batch(ckpt_label, prompt, negative, steps, cfg, n, w, h, *slot_vals):
        names, weights = collect_stack(*slot_vals)
        path = SCANNER.checkpoint_path(ckpt_label)
        if path is None:
            return None, "Pick a checkpoint."
        if G_BATCH is not None:
            eng = G_BATCH.BatchEngine(pipeline=get_pipeline())
            r = eng.generate_batch(
                prompt=prompt, negative_prompt=negative, num_images=int(n),
                width=int(w), height=int(h), steps=int(steps), cfg=float(cfg),
                lora_names=names, lora_weights=weights,
                lora_pool=SCANNER.lora_pool(), checkpoint_path=str(path),
            )
            if not getattr(r, "success", False):
                return None, f"Batch failed: {getattr(r, 'error', '?')}"
            return r.images, f"batch {r.batch_id} | seeds {r.seeds}"
        # fallback: internal loop
        imgs, status = do_generate(ckpt_label, prompt, negative, "default",
                                   steps, cfg, -1, w, h, int(n), None, 0.6,
                                   True, *slot_vals)
        return imgs, status

    def do_vary(gallery, index, strength, prompt, negative, steps, cfg):
        try:
            if not gallery:
                return None, "Generate something first."
            idx = int(index) if index is not None else 0
            idx = max(0, min(idx, len(gallery) - 1))
            item = gallery[idx]
            img = item[0] if isinstance(item, (tuple, list)) else item
            if isinstance(img, str):
                from PIL import Image
                img = Image.open(img)
            pp = get_pipeline()
            if pp._pipe is None:
                return None, "Pipeline not loaded - generate once first."
            pp.load_reference_image(img)
            out = pp.generate(prompt=prompt, negative_prompt=negative,
                              steps=steps, cfg=cfg, num_images=4, seed=-1,
                              denoising_strength=float(strength))
            pp.load_reference_image(None)
            if not out.success:
                return None, f"Vary failed: {out.error}"
            return out.images, (f"4 variations @ strength {strength} | "
                                f"seed {out.seed_used}")
        except Exception as exc:                                # noqa: BLE001
            return None, f"Error: {exc}"

    def do_upscale(gallery, index, prompt, strength):
        try:
            if not gallery:
                return None, "Nothing to upscale."
            idx = max(0, min(int(index or 0), len(gallery) - 1))
            item = gallery[idx]
            img = item[0] if isinstance(item, (tuple, list)) else item
            if isinstance(img, str):
                from PIL import Image
                img = Image.open(img)
            from PIL import Image as PILImage
            w, h = img.size
            up = img.resize((int(w * 1.5) // 8 * 8, int(h * 1.5) // 8 * 8),
                            PILImage.LANCZOS)
            pp = get_pipeline()
            if pp._pipe is None:
                return [up], "Lanczos 1.5x only (load a model for hires refine)."
            pp.load_reference_image(up)
            out = pp.generate(prompt=prompt or "high quality, detailed",
                              steps=24, cfg=6.0, seed=-1,
                              denoising_strength=float(strength))
            pp.load_reference_image(None)
            if not out.success:
                return [up], f"Refine failed ({out.error}); Lanczos result shown."
            return out.images, f"hires-fix 1.5x @ denoise {strength}"
        except Exception as exc:                                # noqa: BLE001
            return None, f"Error: {exc}"

    def _editor_to_mask(editor_value):
        """Editor value -> (base PIL, mask PIL). Handles BOTH formats:
        gradio 4 ImageEditor {background, layers, composite} and
        gradio 3 sketch {image, mask}."""
        import numpy as np
        from PIL import Image
        if editor_value is None:
            return None, None
        if isinstance(editor_value, Image.Image):      # plain image, no mask
            return editor_value.convert("RGB"), None

        # gradio 3.x sketch dict
        if isinstance(editor_value, dict) and "image" in editor_value:
            img = editor_value.get("image")
            msk = editor_value.get("mask")
            if img is None:
                return None, None
            base = img.convert("RGB")
            mask = None
            if msk is not None:
                arr = np.asarray(msk)
                chan = arr[..., 3] if (arr.ndim == 3 and arr.shape[2] == 4) \
                    else (arr if arr.ndim == 2 else arr[..., 0])
                if chan.max() > 0:
                    mask = Image.fromarray((chan > 10).astype("uint8") * 255, "L")
            return base, mask

        # gradio 4.x ImageEditor dict
        bg = editor_value.get("background")
        layers = editor_value.get("layers") or []
        if bg is None:
            return None, None
        base = Image.fromarray(np.asarray(bg)[..., :3]) \
            if not isinstance(bg, Image.Image) else bg.convert("RGB")
        mask = None
        acc = None
        for layer in layers:
            arr = np.asarray(layer)
            if arr.ndim == 3 and arr.shape[2] == 4:
                alpha = arr[..., 3]
                acc = alpha if acc is None else np.maximum(acc, alpha)
        if acc is not None and acc.max() > 0:
            mask = Image.fromarray((acc > 10).astype("uint8") * 255, "L")
        return base, mask

    def do_inpaint(editor_value, prompt, negative, strength, steps, cfg, seed):
        try:
            base, mask = _editor_to_mask(editor_value)
            if base is None:
                return None, "Load an image into the canvas."
            if mask is None:
                return None, "Paint the region to edit (white brush)."
            pp = get_pipeline()
            if pp._pipe is None:
                return None, "Load a model first (Playground tab, Generate once)."
            if G_CANVAS is not None:
                eng = G_CANVAS.CanvasEngine(pipeline=pp)
                req = G_CANVAS.InpaintRequest(
                    image=base, mask=mask, prompt=prompt,
                    negative_prompt=negative,
                    denoising_strength=float(strength), steps=int(steps),
                    cfg=float(cfg), seed=int(seed),
                )
                r = eng.inpaint(req)
                if not getattr(r, "success", False):
                    return None, f"Inpaint failed: {getattr(r, 'error', '?')}"
                img = getattr(r, "image", None) or (r.images[0] if getattr(r, "images", None) else None)
                return [img], "inpaint done (CanvasEngine)"
            out = pp.generate(prompt=prompt, negative_prompt=negative,
                              init_image=base, mask_image=mask,
                              denoising_strength=strength, steps=steps,
                              cfg=cfg, seed=seed,
                              width=base.width, height=base.height)
            if not out.success:
                return None, f"Inpaint failed: {out.error}"
            return out.images, f"inpaint done | seed {out.seed_used}"
        except Exception as exc:                                # noqa: BLE001
            logger.exception("inpaint")
            return None, f"Error: {exc}"

    def do_outpaint(image, direction, pixels, prompt, strength, steps, cfg):
        try:
            if image is None:
                return None, "Provide an image."
            from PIL import Image, ImageOps
            pp = get_pipeline()
            if pp._pipe is None:
                return None, "Load a model first."
            if G_CANVAS is not None:
                eng = G_CANVAS.CanvasEngine(pipeline=pp)
                req = G_CANVAS.OutpaintRequest(
                    image=image, direction=direction, extend_pixels=int(pixels),
                    prompt=prompt, denoising_strength=float(strength),
                    steps=int(steps), cfg=float(cfg),
                )
                r = eng.outpaint(req)
                if getattr(r, "success", False):
                    img = getattr(r, "image", None) or (r.images[0] if getattr(r, "images", None) else None)
                    return [img], "outpaint done (CanvasEngine)"
            # fallback: pad + inpaint border
            px = int(pixels)
            pad = {"left": (px, 0, 0, 0), "right": (0, 0, px, 0),
                   "up": (0, px, 0, 0), "down": (0, 0, 0, px),
                   "all": (px, px, px, px)}[direction]
            big = ImageOps.expand(image.convert("RGB"), pad, fill=(127, 127, 127))
            mask = Image.new("L", big.size, 255)
            mask.paste(0, (pad[0], pad[1], pad[0] + image.width, pad[1] + image.height))
            out = pp.generate(prompt=prompt, init_image=big, mask_image=mask,
                              denoising_strength=strength, steps=steps, cfg=cfg,
                              width=big.width // 8 * 8, height=big.height // 8 * 8)
            if not out.success:
                return None, f"Outpaint failed: {out.error}"
            return out.images, "outpaint done (fallback engine)"
        except Exception as exc:                                # noqa: BLE001
            return None, f"Error: {exc}"

    def do_save_preset(name, ckpt, prompt, negative, sampler, steps, cfg,
                       width, height, *slot_vals):
        names, weights = collect_stack(*slot_vals)
        return save_preset(name or f"preset_{int(time.time())}", {
            "checkpoint": ckpt, "prompt": prompt, "negative": negative,
            "sampler": sampler, "steps": steps, "cfg": cfg,
            "width": width, "height": height,
            "loras": list(zip(names, weights)),
        })

    def do_load_preset(name):
        p = load_presets().get(name)
        if not p:
            return [gr.update()] * (9 + LORA_SLOTS * 3) + ["Preset not found."]
        slot_names = [""] * LORA_SLOTS
        slot_ws = [0.8] * LORA_SLOTS
        slot_en = [False] * LORA_SLOTS
        for i, (n, w) in enumerate(p.get("loras", [])[:LORA_SLOTS]):
            slot_names[i], slot_ws[i], slot_en[i] = n, w, True
        return ([gr.update(value=p.get("checkpoint")),
                 gr.update(value=p.get("prompt", "")),
                 gr.update(value=p.get("negative", DEFAULT_NEG)),
                 gr.update(value=p.get("sampler", "default")),
                 gr.update(value=p.get("steps", 30)),
                 gr.update(value=p.get("cfg", 7.0)),
                 gr.update(value=p.get("width", 1024)),
                 gr.update(value=p.get("height", 1024)),
                 gr.update()]                       # seed untouched
                + [gr.update(value=v) for v in slot_names]
                + [gr.update(value=v) for v in slot_ws]
                + [gr.update(value=v) for v in slot_en]
                + [f"Loaded preset '{name}'"])

    # ---------- layout ----------

    css = """
    .gradio-container {max-width: 1400px !important}
    footer {display: none !important}
    """
    with gr.Blocks(title="Grokkie Playground Studio", css=css,
                   **theme_kwargs()) as demo:
        gr.Markdown("# Grokkie Playground Studio\n"
                    "local generation lab · unlimited LoRA stack · "
                    "WeightEngine semantic weighting · MPS optimized")

        with gr.Tab("Playground"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Row():
                        ckpt_dd = gr.Dropdown(ckpt_labels, label="Base model "
                                              "(type auto-detected)", value=ckpt_labels[0])
                        rescan_btn = gr.Button("Rescan", scale=0)
                    prompt_tb = gr.Textbox(label="Prompt", lines=3)
                    neg_tb = gr.Textbox(label="Negative prompt", value=DEFAULT_NEG, lines=2)

                    gr.Markdown("### LoRA stack")
                    budget_md = gr.Markdown("")
                    slot_names, slot_weights, slot_enabled = [], [], []
                    for i in range(LORA_SLOTS):
                        with gr.Row():
                            slot_names.append(gr.Dropdown(
                                lora_names, value="", label=f"LoRA {i+1}",
                                scale=4))
                            slot_weights.append(gr.Slider(
                                -1.5, 2.0, value=0.8, step=0.05,
                                label="weight" if i == 0 else None, scale=3))
                            slot_enabled.append(gr.Checkbox(
                                value=True, label="on" if i == 0 else None,
                                scale=1))
                    suggest_btn = gr.Button("Suggest weights (WeightEngine)")

                    with gr.Accordion("Parameters", open=True):
                        with gr.Row():
                            sampler_dd = gr.Dropdown(PlaygroundPipeline.SAMPLERS,
                                                     value="DPM++ 2M Karras",
                                                     label="Sampler")
                            steps_sl = gr.Slider(8, 80, 30, step=1, label="Steps")
                            cfg_sl = gr.Slider(1, 15, 7.0, step=0.5, label="CFG")
                        with gr.Row():
                            seed_nb = gr.Number(value=-1, label="Seed (-1 random)")
                            width_sl = gr.Slider(512, 1536, 1024, step=64, label="Width")
                            height_sl = gr.Slider(512, 1536, 1024, step=64, label="Height")
                            count_sl = gr.Slider(1, 4, 1, step=1, label="Images")
                        pony_chk = gr.Checkbox(True, label="Auto Pony quality tags "
                                               "(when base = PONY)")

                    with gr.Accordion("img2img (optional init image)", open=False):
                        init_img = gr.Image(type="pil", label="Init image")
                        strength_sl = gr.Slider(0.05, 1.0, 0.6, step=0.05,
                                                label="Denoising strength")

                    with gr.Row():
                        gen_btn = gr.Button("Generate", variant="primary")
                        unload_btn = gr.Button("Unload / free memory")

                with gr.Column(scale=4):
                    gallery = gr.Gallery(label="Output", columns=2, height=560)
                    status_md = gr.Markdown("Ready.")

        with gr.Tab("Batch & Vary"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Batch (sequential on MPS)")
                    b_count = gr.Slider(2, 8, 4, step=1, label="Images")
                    b_btn = gr.Button("Generate batch", variant="primary")
                    gr.Markdown("### Vary selected")
                    v_index = gr.Number(value=0, label="Image # in gallery (0-based)")
                    v_strength = gr.Slider(0.1, 0.6, 0.3, step=0.05,
                                           label="Variation strength "
                                                 "(0.15 subtle / 0.45 strong)")
                    v_btn = gr.Button("Make 4 variations")
                    gr.Markdown("### Upscale selected (hires-fix 1.5x)")
                    u_strength = gr.Slider(0.1, 0.6, 0.35, step=0.05,
                                           label="Refine denoise")
                    u_btn = gr.Button("Upscale + refine")
                with gr.Column():
                    b_gallery = gr.Gallery(label="Results", columns=2, height=560)
                    b_status = gr.Markdown("")

        with gr.Tab("Canvas (inpaint / outpaint)"):
            with gr.Row():
                with gr.Column():
                    editor = make_mask_editor("Paint the edit region (brush)")
                    c_prompt = gr.Textbox(label="Prompt", lines=2)
                    c_neg = gr.Textbox(label="Negative", value=DEFAULT_NEG, lines=1)
                    with gr.Row():
                        c_strength = gr.Slider(0.3, 1.0, 0.85, step=0.05,
                                               label="Denoise")
                        c_steps = gr.Slider(10, 60, 30, step=1, label="Steps")
                        c_cfg = gr.Slider(1, 15, 7.5, step=0.5, label="CFG")
                        c_seed = gr.Number(value=-1, label="Seed")
                    inpaint_btn = gr.Button("Inpaint", variant="primary")
                    gr.Markdown("### Outpaint")
                    o_image = gr.Image(type="pil", label="Image to extend")
                    with gr.Row():
                        o_dir = gr.Dropdown(["left", "right", "up", "down", "all"],
                                            value="all", label="Direction")
                        o_px = gr.Slider(64, 512, 256, step=64, label="Pixels")
                    o_btn = gr.Button("Outpaint")
                with gr.Column():
                    c_gallery = gr.Gallery(label="Canvas results", columns=1,
                                           height=620)
                    c_status = gr.Markdown("")

        with gr.Tab("Presets"):
            with gr.Row():
                p_name = gr.Textbox(label="Preset name")
                p_save = gr.Button("Save current Playground setup")
            with gr.Row():
                p_pick = gr.Dropdown(sorted(load_presets()), label="Load preset",
                                     scale=4)
                p_refresh = gr.Button("Refresh", scale=1)
            p_load = gr.Button("Load")
            p_status = gr.Markdown("")
            p_refresh.click(
                lambda: gr.update(choices=sorted(load_presets())),
                outputs=p_pick)

        # ---------- wiring ----------
        slot_all = [*slot_names, *slot_weights, *slot_enabled]

        rescan_btn.click(rescan, outputs=[ckpt_dd, *slot_names, status_md])
        suggest_btn.click(suggest_weights, inputs=[prompt_tb, *slot_all],
                          outputs=[*slot_weights, budget_md])
        gen_btn.click(
            do_generate,
            inputs=[ckpt_dd, prompt_tb, neg_tb, sampler_dd, steps_sl, cfg_sl,
                    seed_nb, width_sl, height_sl, count_sl, init_img,
                    strength_sl, pony_chk, *slot_all],
            outputs=[gallery, status_md])
        unload_btn.click(do_unload, outputs=status_md)

        b_btn.click(do_batch,
                    inputs=[ckpt_dd, prompt_tb, neg_tb, steps_sl, cfg_sl,
                            b_count, width_sl, height_sl, *slot_all],
                    outputs=[b_gallery, b_status])
        v_btn.click(do_vary,
                    inputs=[b_gallery, v_index, v_strength, prompt_tb, neg_tb,
                            steps_sl, cfg_sl],
                    outputs=[b_gallery, b_status])
        u_btn.click(do_upscale, inputs=[b_gallery, v_index, prompt_tb, u_strength],
                    outputs=[b_gallery, b_status])

        inpaint_btn.click(do_inpaint,
                          inputs=[editor, c_prompt, c_neg, c_strength, c_steps,
                                  c_cfg, c_seed],
                          outputs=[c_gallery, c_status])
        o_btn.click(do_outpaint,
                    inputs=[o_image, o_dir, o_px, c_prompt, c_strength,
                            c_steps, c_cfg],
                    outputs=[c_gallery, c_status])

        p_save.click(do_save_preset,
                     inputs=[p_name, ckpt_dd, prompt_tb, neg_tb, sampler_dd,
                             steps_sl, cfg_sl, width_sl, height_sl, *slot_all],
                     outputs=p_status)
        p_load.click(do_load_preset, inputs=p_pick,
                     outputs=[ckpt_dd, prompt_tb, neg_tb, sampler_dd, steps_sl,
                              cfg_sl, width_sl, height_sl, seed_nb,
                              *slot_names, *slot_weights, *slot_enabled,
                              p_status])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Grokkie Playground Studio")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=args.port,
                share=args.share, show_error=True)


if __name__ == "__main__":
    main()
