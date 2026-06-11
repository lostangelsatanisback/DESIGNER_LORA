"""
grokkie_canvas.py — Inpainting + Outpainting
=============================================
Codename: Z. Phase 7D of Grokkie ULTIMA.

Midjourney's Vary(Region) and Zoom Out become a canvas-editing module.
Two operations:

  • inpaint — replace a user-drawn masked region with newly-generated
    content driven by a prompt. Mask is **always** user-drawn; we never
    auto-generate masks.
  • outpaint — extend the canvas in one or more directions. The
    extended region becomes a mask and is filled by inpainting.
  • smart_outpaint — same as outpaint, but the prompt is inferred from
    the original image's analysis (category, palette, background type).

The CanvasEngine sits on top of :class:`RedresserPipeline` for img2img
fall-back when no dedicated inpaint pipeline is available; when a real
inpainting pipeline is loadable from diffusers, that path is used
preferentially for sharper masked boundaries.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent,
              Path.home() / "deepseek-coder" / "PROMPT_GENERATION_101"):
    if _cand.exists() and (_cand / "grokkie_engine.py").exists() \
       and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))


# --------------------------------------------------------------------------- #
# Request / result dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class InpaintRequest:
    """Inpaint a user-masked region."""
    image: Any                              # PIL.Image
    mask: Any                               # PIL.Image (white = edit, black = preserve)
    prompt: str
    negative_prompt: str = ""
    denoising_strength: float = 0.8
    steps: int = 30
    cfg: float = 7.5
    seed: int = -1
    lora_names: list[str] = field(default_factory=list)
    preserve_surrounding: bool = True


@dataclass
class OutpaintRequest:
    """Extend the canvas in a direction."""
    image: Any                              # PIL.Image
    direction: str = "all"                  # left | right | up | down | all
    extend_pixels: int = 256
    prompt: str = ""
    negative_prompt: str = ""
    denoising_strength: float = 0.7
    steps: int = 30
    cfg: float = 7.5
    seed: int = -1
    feather_pixels: int = 24


@dataclass
class CanvasResult:
    image: Any = None                       # PIL.Image (the edited output)
    original_size: tuple[int, int] = (0, 0)
    new_size: tuple[int, int] = (0, 0)
    operation: str = "inpaint"
    generation_time_seconds: float = 0.0
    success: bool = True
    error: str = ""
    notes: list[str] = field(default_factory=list)
    seed_used: int = -1

    def to_dict(self, with_image: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "original_size": list(self.original_size),
            "new_size": list(self.new_size),
            "operation": self.operation,
            "generation_time_seconds": round(self.generation_time_seconds, 2),
            "success": self.success,
            "error": self.error,
            "notes": list(self.notes),
            "seed_used": int(self.seed_used),
        }
        if with_image and self.image is not None:
            from grokkie_redresser import _image_to_b64  # type: ignore
            d["image_b64"] = _image_to_b64(self.image)
        return d


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

class CanvasEngine:
    """Inpaint + outpaint operations.

    Falls back gracefully when neither a diffusers inpaint pipeline nor the
    base img2img RedresserPipeline are loadable — the dry-run path returns
    the original image with ``success=False`` and an explanatory error.
    """

    def __init__(self, pipeline: Any = None) -> None:
        self._pipeline = pipeline

    @property
    def pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from grokkie_redresser import get_redresser
            self._pipeline = get_redresser()
            return self._pipeline
        except Exception:
            return None

    # ---------- inpaint ---------- #

    def inpaint(self, req: InpaintRequest) -> CanvasResult:
        """Replace the masked region with newly-generated content."""
        try:
            from PIL import Image
        except ImportError:
            return CanvasResult(success=False, error="PIL required")

        if not hasattr(req.image, "size"):
            return CanvasResult(success=False, error="image must be a PIL Image")
        if not hasattr(req.mask, "size"):
            return CanvasResult(success=False, error="mask must be a PIL Image")

        orig = req.image.convert("RGB")
        mask = req.mask.convert("L")
        # Mask must match image dims.
        if mask.size != orig.size:
            mask = mask.resize(orig.size, Image.NEAREST)

        t_start = time.time()
        rp = self.pipeline
        output_img: Any = orig
        seed_used: int = -1
        notes: list[str] = []
        success = True
        error = ""

        # Try a real inpaint pipeline first.
        used_inpaint_path = False
        try:
            from diffusers import (
                StableDiffusionInpaintPipeline,
                StableDiffusionXLInpaintPipeline,
            )
            # We require diffusers to even attempt this path.
            if rp is not None and rp._pipe is not None and rp.base_model in ("SDXL", "SD15"):
                import torch
                cls = (StableDiffusionXLInpaintPipeline if rp.base_model == "SDXL"
                       else StableDiffusionInpaintPipeline)
                # Build the inpainter from the base checkpoint path possible.
                inpainter = cls.from_single_file(
                    rp.checkpoint_path,
                    torch_dtype=torch.float16 if rp.device != "cpu" else torch.float32,
                )
                inpainter.to(rp.device)
                if int(req.seed) < 0:
                    import random
                    seed_used = random.randint(0, 2**31 - 1)
                else:
                    seed_used = int(req.seed)
                generator = torch.Generator(
                    device=rp.device if rp.device != "mps" else "cpu",
                ).manual_seed(seed_used)
                with torch.inference_mode():
                    out = inpainter(
                        prompt=req.prompt,
                        negative_prompt=req.negative_prompt,
                        image=orig, mask_image=mask,
                        strength=float(req.denoising_strength),
                        num_inference_steps=int(req.steps),
                        guidance_scale=float(req.cfg),
                        generator=generator,
                    )
                if getattr(out, "images", None):
                    output_img = out.images[0]
                    used_inpaint_path = True
        except Exception as exc:
            notes.append(f"native inpainter unavailable ({exc}) — falling back")

        # Fallback: emulate inpaint by running img2img on the cropped region.
        if not used_inpaint_path:
            if rp is None:
                return CanvasResult(
                    operation="inpaint", image=orig,
                    original_size=orig.size, new_size=orig.size,
                    success=False,
                    error="no pipeline available (install torch + diffusers)",
                    generation_time_seconds=time.time() - t_start,
                    notes=notes,
                )
            try:
                rp.load_reference_image(orig)
                r = rp.generate(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    width=orig.size[0], height=orig.size[1],
                    denoising_strength=float(req.denoising_strength),
                    steps=int(req.steps), cfg=float(req.cfg),
                    seed=int(req.seed), num_images=1,
                    preserve_face=False,
                )
                if r.images:
                    generated = r.images[0]
                    if req.preserve_surrounding:
                        output_img = Image.composite(generated, orig, mask)
                    else:
                        output_img = generated
                    seed_used = int(r.seed_used)
                    notes.append("img2img fallback used — masked region composited")
                else:
                    success = False
                    error = r.error or "no image produced by img2img fallback"
            except Exception as exc:
                success = False
                error = f"inpaint exception: {exc}"

        return CanvasResult(
            operation="inpaint",
            image=output_img,
            original_size=orig.size, new_size=output_img.size,
            generation_time_seconds=time.time() - t_start,
            success=success, error=error, notes=notes, seed_used=seed_used,
        )

    # ---------- outpaint ---------- #

    def outpaint(self, req: OutpaintRequest) -> CanvasResult:
        """Extend the canvas in a direction and fill the new region."""
        try:
            from PIL import Image, ImageFilter
        except ImportError:
            return CanvasResult(success=False, error="PIL required")
        if not hasattr(req.image, "size"):
            return CanvasResult(success=False, error="image must be a PIL Image")

        orig = req.image.convert("RGB")
        ow, oh = orig.size
        d = (req.direction or "all").lower()
        ext = int(max(0, req.extend_pixels))
        if ext <= 0:
            return CanvasResult(
                operation="outpaint", image=orig,
                original_size=(ow, oh), new_size=(ow, oh),
                success=False, error="extend_pixels must be > 0",
            )

        # Build the new canvas.
        left_pad = ext if d in ("left", "all") else 0
        right_pad = ext if d in ("right", "all") else 0
        top_pad = ext if d in ("up", "all") else 0
        bottom_pad = ext if d in ("down", "all") else 0
        new_w = ow + left_pad + right_pad
        new_h = oh + top_pad + bottom_pad

        canvas = Image.new("RGB", (new_w, new_h), (16, 24, 32))
        canvas.paste(orig, (left_pad, top_pad))

        # Mask: white where we want to generate, black where to preserve.
        mask = Image.new("L", (new_w, new_h), 255)
        # Paste a black box for the preserved region.
        pres = Image.new("L", (ow, oh), 0)
        mask.paste(pres, (left_pad, top_pad))
        # Feather the seam.
        if req.feather_pixels > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=req.feather_pixels))

        # Run as an inpaint request.
        ip_req = InpaintRequest(
            image=canvas, mask=mask, prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            denoising_strength=float(req.denoising_strength),
            steps=int(req.steps), cfg=float(req.cfg),
            seed=int(req.seed), preserve_surrounding=True,
        )
        ir = self.inpaint(ip_req)
        return CanvasResult(
            operation="outpaint",
            image=ir.image if ir.success else canvas,
            original_size=(ow, oh), new_size=(new_w, new_h),
            generation_time_seconds=ir.generation_time_seconds,
            success=ir.success, error=ir.error,
            notes=list(ir.notes) + [
                f"extended {ext}px in '{d}' direction"
            ],
            seed_used=ir.seed_used,
        )

    # ---------- smart outpaint ---------- #

    def smart_outpaint(
        self, image: Any, direction: str = "all",
        extend_pixels: int = 256,
        steps: int = 30, cfg: float = 7.5,
        denoising_strength: float = 0.7, seed: int = -1,
    ) -> CanvasResult:
        """Outpaint with an auto-generated prompt derived from image analysis."""
        prompt = ""
        try:
            from grokkie_mastery import LoRAAnalyst
            analyst = LoRAAnalyst("")
            a = analyst.analyze_image_v2(image)
            palette = ", ".join(a.color_palette[:2]) if a.color_palette else ""
            bits = [
                f"continuation of {a.category} scene",
                f"{a.background_type} background" if a.background_type else "",
                f"{palette} tones" if palette else "",
                a.warmth + " color temperature" if a.warmth != "neutral" else "",
            ]
            prompt = ", ".join([b for b in bits if b])
        except Exception:
            prompt = "natural continuation of the scene, seamless"
        req = OutpaintRequest(
            image=image, direction=direction, extend_pixels=int(extend_pixels),
            prompt=prompt, denoising_strength=float(denoising_strength),
            steps=int(steps), cfg=float(cfg), seed=int(seed),
        )
        return self.outpaint(req)


# --------------------------------------------------------------------------- #
# Singleton accessor
# --------------------------------------------------------------------------- #

_singleton: CanvasEngine | None = None


def get_canvas_engine() -> CanvasEngine:
    global _singleton
    if _singleton is None:
        _singleton = CanvasEngine()
    return _singleton


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (256, 256), (200, 100, 100))
    mask = Image.new("L", (256, 256), 0)
    d = ImageDraw.Draw(mask)
    d.rectangle([80, 80, 180, 180], fill=255)
    eng = CanvasEngine()
    r = eng.inpaint(InpaintRequest(image=img, mask=mask, prompt="test"))
    print("inpaint:", r.success, r.error or "OK")
    r = eng.smart_outpaint(img, direction="right", extend_pixels=64)
    print("smart_outpaint:", r.success, r.new_size)
