"""Forge/A1111/reForge API client (stdlib only). Start reForge with --api.

LoRAs are injected the webui way: <lora:name:weight> in the prompt
(negative weights work for subtractive effects). Your reForge install at
forge_root serves this on http://127.0.0.1:7860 by default.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Optional


class ForgeClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7860", timeout: int = 600):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, route: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base}{route}",
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def alive(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base}/sdapi/v1/options")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False

    @staticmethod
    def lora_prompt(prompt: str, loras: list[tuple[str, float]]) -> str:
        """Append <lora:stem:weight> tags (webui convention; negatives ok)."""
        from pathlib import Path
        tags = "".join(
            f" <lora:{Path(p).stem}:{w:g}>" for p, w in loras
        )
        return prompt + tags

    @staticmethod
    def build_txt2img_payload(prompt: str, negative: str = "", steps: int = 28,
                              cfg: float = 6.0, width: int = 1024,
                              height: int = 1024, seed: int = 42,
                              sampler: str = "DPM++ 2M Karras",
                              loras: Optional[list[tuple[str, float]]] = None,
                              hires: bool = False, clip_skip: int = 0) -> dict:
        payload = {
            "prompt": ForgeClient.lora_prompt(prompt, loras or []),
            "negative_prompt": negative,
            "steps": int(steps), "cfg_scale": float(cfg),
            "width": int(width), "height": int(height),
            "seed": int(seed), "sampler_name": sampler,
        }
        if int(clip_skip) > 1:
            payload["override_settings"] = {
                "CLIP_stop_at_last_layers": int(clip_skip)}
            payload["override_settings_restore_afterwards"] = True
        if hires:
            payload.update(enable_hr=True, hr_scale=1.5,
                           hr_upscaler="Latent", denoising_strength=0.45)
        return payload

    def txt2img(self, **kw) -> bytes:
        data = self._post("/sdapi/v1/txt2img", self.build_txt2img_payload(**kw))
        return base64.b64decode(data["images"][0])

    def img2img(self, init_png: bytes, prompt: str, negative: str = "",
                strength: float = 0.6, steps: int = 28, cfg: float = 6.0,
                seed: int = 42, sampler: str = "DPM++ 2M Karras",
                loras: Optional[list[tuple[str, float]]] = None) -> bytes:
        payload = {
            "init_images": [base64.b64encode(init_png).decode()],
            "prompt": self.lora_prompt(prompt, loras or []),
            "negative_prompt": negative,
            "denoising_strength": float(strength),
            "steps": int(steps), "cfg_scale": float(cfg),
            "seed": int(seed), "sampler_name": sampler,
        }
        data = self._post("/sdapi/v1/img2img", payload)
        return base64.b64decode(data["images"][0])
