"""
grokkie_redresser.py — Production Redresser Pipeline (Pony + MPS Optimized)
Borrowed & upgraded from your Dressing Studio codebase.
"""

from __future__ import annotations

import gc
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline

logger = logging.getLogger(__name__)

# =========================================================================== #
# Result
# =========================================================================== #

@dataclass
class RedresserResult:
    images: list[Image.Image] = field(default_factory=list)
    seed_used: int = -1
    generation_time_seconds: float = 0.0
    peak_memory_gb: float = 0.0
    success: bool = True
    error: str = ""
    notes: list[str] = field(default_factory=list)


# =========================================================================== #
# Pipeline
# =========================================================================== #

class RedresserPipeline:
    def __init__(self, device: str = "auto"):
        self.device = device
        self._pipe = None
        self.base_model = "SDXL"
        self.checkpoint_path = ""
        self.adapters: list[tuple[str, float]] = []
        self.ipa_loaded = False

    def load_checkpoint(self, path: str, base_model: str = "auto"):
        """Load Pony Diffusion or any SDXL checkpoint."""
        self.checkpoint_path = path
        self.base_model = base_model.upper() if base_model != "auto" else "SDXL"

        ckpt = Path(path)
        if not ckpt.exists():
            return {"loaded": False, "error": f"Checkpoint not found: {ckpt}"}

        try:
            if ckpt.is_file():
                self._pipe = AutoPipelineForInpainting.from_single_file(
                    str(ckpt),
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    safety_checker=None,
                )
            else:
                self._pipe = AutoPipelineForInpainting.from_pretrained(
                    str(ckpt), torch_dtype=torch.float16, safety_checker=None
                )

            self._pipe.to("mps")
            self._apply_optimizations()
            logger.info(f"✅ Loaded {ckpt.name} on MPS")
            return {"loaded": True, "base_model": self.base_model}

        except Exception as e:
            logger.error(f"Checkpoint load failed: {e}")
            return {"loaded": False, "error": str(e)}

    def _apply_optimizations(self):
        if not self._pipe:
            return
        try:
            self._pipe.enable_model_cpu_offload()
            self._pipe.enable_vae_slicing()
            self._pipe.enable_vae_tiling()
            self._pipe.enable_attention_slicing("auto")
        except Exception as e:
            logger.warning(f"Optimization warning: {e}")

    def load_loras(self, lora_names: list[str], weights: list[float], lora_pool: dict[str, str]):
        if not self._pipe or not lora_names:
            return {"loaded": [], "weights": []}

        loaded = []
        for name, w in zip(lora_names, weights):
            if name not in lora_pool:
                continue
            path = lora_pool[name]
            adapter_name = Path(path).stem.replace(" ", "_").replace("-", "_")

            try:
                self._pipe.load_lora_weights(
                    Path(path).parent, weight_name=Path(path).name, adapter_name=adapter_name
                )
                self.adapters.append((adapter_name, float(w)))
                loaded.append(name)
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")

        # Apply all adapters
        if self.adapters:
            names = [n for n, _ in self.adapters]
            ws = [w for _, w in self.adapters]
            self._pipe.set_adapters(names, adapter_weights=ws)

        logger.info(f"✅ Loaded {len(loaded)} LoRAs")
        return {"loaded": loaded, "weights": [w for _, w in self.adapters]}

    def generate(self, **kwargs) -> RedresserResult:
        start = time.time()
        monitor = MPSMonitor()

        if not self._pipe:
            return RedresserResult(success=False, error="Pipeline not loaded")

        try:
            # Default Pony-friendly prompt
            prompt = kwargs.get("prompt", "")
            if not prompt or "score_" not in prompt:
                prompt = "score_9, score_8_up, score_7_up, " + prompt

            output = self._pipe(
                prompt=prompt,
                negative_prompt=kwargs.get("negative_prompt", ""),
                width=int(kwargs.get("width", 1024)),
                height=int(kwargs.get("height", 1024)),
                num_inference_steps=int(kwargs.get("steps", 35)),
                guidance_scale=float(kwargs.get("cfg", 7.0)),
                strength=float(kwargs.get("denoising_strength", 0.88)),
                generator=torch.Generator("cpu").manual_seed(int(kwargs.get("seed", 42))),
            )

            return RedresserResult(
                images=output.images,
                seed_used=kwargs.get("seed", 42),
                generation_time_seconds=time.time() - start,
                peak_memory_gb=monitor.peak_gb(),
                success=True,
                notes=["Pony Diffusion V6 + Multi-LoRA", "MPS optimized"]
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return RedresserResult(success=False, error=str(e))


class MPSMonitor:
    def peak_gb(self) -> float:
        try:
            if torch.backends.mps.is_available():
                return torch.mps.current_allocated_memory() / (1024**3)
        except Exception:
            pass
        return 0.0


def get_redresser(device: str = "auto") -> RedresserPipeline:
    return RedresserPipeline(device)


if __name__ == "__main__":
    rp = get_redresser()
    print("✅ RedresserPipeline ready (Pony + Multi-LoRA)")
