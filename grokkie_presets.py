"""
grokkie_presets.py — Style Preset System
==========================================
Codename: Z. Phase 7A of Grokkie ULTIMA.

A curated, shareable style preset library. Each preset bundles a prompt
template, LoRA selections, IP-Adapter config, generation params, and
negative complements into a single named recipe the user can activate
with one click — the experience parity with Midjourney's `--style raw`,
`--v 6.1`, `--style cinematic` keyword system.

Built-in presets ship with the project; user presets live in
``presets/user/<id>.json``. Both formats are interchangeable — exporting
a preset gives a portable JSON file.

Public API:
  - :class:`StylePreset` — the recipe dataclass
  - :class:`PresetLibrary` — full CRUD + search + apply
  - ``BUILTIN_PRESETS`` — 20+ shipped recipes
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent,
              Path.home() / "deepseek-coder" / "PROMPT_GENERATION_101"):
    if _cand.exists() and (_cand / "grokkie_engine.py").exists() \
       and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))


# --------------------------------------------------------------------------- #
# Dataclass
# --------------------------------------------------------------------------- #

@dataclass
class StylePreset:
    """A complete style recipe — one click activates the full stack.

    ``prompt_template`` supports a ``{subject}`` placeholder. ``apply_preset``
    fills it with the user-provided subject text. Templates without
    ``{subject}`` get the subject prepended.

    ``negative_template`` supports a ``{style}`` placeholder which is filled
    with the detected style from PromptArchitect.analyze_prompt().
    """
    id: str
    name: str
    category: str
    description: str
    prompt_template: str
    negative_template: str = ""
    prompt_suffix: str = ""
    preview_image: str | None = None
    lora_names: list[str] = field(default_factory=list)
    lora_weights: list[float] = field(default_factory=list)
    ip_adapter_type: str = "none"
    ip_adapter_scale: float = 0.0
    recommended_base_model: str = "SDXL"
    generation_params: dict[str, Any] = field(default_factory=dict)
    denoising_strength: float = 0.5
    preserve_face: bool = True
    preserve_pose: bool = False
    preserve_background: bool = False
    tags: list[str] = field(default_factory=list)
    author: str = "built-in"
    created_at: str = ""
    is_builtin: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StylePreset":
        """Tolerantly build a preset from a dict — unknown keys are dropped."""
        keep = {k for k in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in keep}
        return cls(**clean)


# --------------------------------------------------------------------------- #
# Built-in presets (20+) — each one a complete recipe.
# --------------------------------------------------------------------------- #

BUILTIN_PRESETS: list[dict[str, Any]] = [
    # ---------- Portraits ---------- #
    {"id": "cinematic_golden_hour", "name": "Cinematic Golden Hour", "category": "portrait",
     "description": "Warm backlit portrait with anamorphic flares and film grain — the signature cinematic look.",
     "prompt_template": "{subject}, golden hour backlight, anamorphic lens flare, film grain, volumetric haze, shallow depth of field",
     "negative_template": "{style}, cartoon, anime, flat, overexposed, harsh flash",
     "prompt_suffix": "cinematic lighting, dramatic atmosphere, 35mm film",
     "lora_names": ["cinematic_lighting_xl"], "lora_weights": [0.8],
     "ip_adapter_type": "face_id_plus", "ip_adapter_scale": 0.7,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 35, "cfg": 8.0},
     "denoising_strength": 0.5, "preserve_face": True, "preserve_pose": False, "preserve_background": False,
     "tags": ["dramatic", "warm", "film", "portrait", "cinematic"]},

    {"id": "editorial_clean", "name": "Editorial Clean", "category": "portrait",
     "description": "High-fashion editorial look — clean skin, studio lighting, magazine-ready.",
     "prompt_template": "{subject}, clean editorial lighting, flawless skin, studio backdrop, Vogue quality",
     "negative_template": "{style}, casual, snapshot, low quality, grain, noise",
     "prompt_suffix": "professional photography, retouched, magazine quality",
     "lora_names": ["editorial_composition_xl"], "lora_weights": [0.7],
     "ip_adapter_type": "face_id_plus", "ip_adapter_scale": 0.8,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 30, "cfg": 7.5},
     "denoising_strength": 0.45, "preserve_face": True, "preserve_pose": False, "preserve_background": True,
     "tags": ["clean", "fashion", "studio", "editorial", "professional"]},

    {"id": "film_noir", "name": "Film Noir", "category": "portrait",
     "description": "High-contrast black and white with dramatic shadows and a mysterious atmosphere.",
     "prompt_template": "{subject}, film noir, high contrast monochrome, dramatic shadows, venetian blind light, cigarette smoke",
     "negative_template": "{style}, color, bright, cheerful, flat lighting",
     "prompt_suffix": "black and white, noir, moody, 1940s cinema",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "face_id", "ip_adapter_scale": 0.6,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 30, "cfg": 9.0},
     "denoising_strength": 0.6, "preserve_face": True, "preserve_pose": False, "preserve_background": False,
     "tags": ["noir", "monochrome", "dramatic", "moody", "vintage"]},

    {"id": "oil_painting_classical", "name": "Classical Oil Painting", "category": "portrait",
     "description": "Renaissance-style oil painting with visible brushstrokes, chiaroscuro, and rich pigments.",
     "prompt_template": "{subject}, oil painting, classical renaissance, chiaroscuro, visible brushstrokes, rich pigments, gold leaf frame",
     "negative_template": "{style}, photo, photograph, modern, digital, 3d render",
     "prompt_suffix": "fine art, masterpiece, oil on canvas, gallery quality",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "portrait_style", "ip_adapter_scale": 0.5,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 35, "cfg": 8.5},
     "denoising_strength": 0.65, "preserve_face": True, "preserve_pose": False, "preserve_background": True,
     "tags": ["painting", "classical", "fine art", "renaissance", "traditional"]},

    {"id": "cyberpunk_neon", "name": "Cyberpunk Neon", "category": "portrait",
     "description": "Neon-drenched cyberpunk aesthetic with holographic reflections and rain-slicked surfaces.",
     "prompt_template": "{subject}, cyberpunk, neon lights, holographic reflections, rain-slicked streets, LED advertisements, futuristic cityscape",
     "negative_template": "{style}, natural, daylight, pastoral, vintage, sepia",
     "prompt_suffix": "neon glow, blade runner, sci-fi, cybernetic",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "face_id", "ip_adapter_scale": 0.65,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 35, "cfg": 8.0},
     "denoising_strength": 0.55, "preserve_face": True, "preserve_pose": False, "preserve_background": False,
     "tags": ["cyberpunk", "neon", "sci-fi", "futuristic", "dark"]},

    {"id": "watercolor_soft", "name": "Soft Watercolor", "category": "portrait",
     "description": "Delicate watercolor wash with bleeding pigments and soft paper texture.",
     "prompt_template": "{subject}, watercolor painting, soft pigment wash, bleeding colors, paper texture, delicate, ethereal",
     "negative_template": "{style}, photorealistic, sharp edges, digital, harsh contrast",
     "prompt_suffix": "watercolor art, hand painted, soft edges, traditional media",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "portrait_style", "ip_adapter_scale": 0.4,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 28, "cfg": 7.0},
     "denoising_strength": 0.7, "preserve_face": True, "preserve_pose": False, "preserve_background": True,
     "tags": ["watercolor", "painting", "soft", "delicate", "traditional"]},

    {"id": "dark_fantasy", "name": "Dark Fantasy", "category": "portrait",
     "description": "Brooding dark fantasy portrait — gothic atmosphere, mystical tones, dramatic shadows.",
     "prompt_template": "{subject}, dark fantasy, gothic atmosphere, mystical, dramatic shadow, ornate detail, brooding mood",
     "negative_template": "{style}, bright, cheerful, modern, cartoon",
     "prompt_suffix": "fantasy art, dark fairy tale, baroque ornament, mystical",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "face_id", "ip_adapter_scale": 0.65,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 35, "cfg": 8.5},
     "denoising_strength": 0.6, "preserve_face": True, "preserve_pose": False, "preserve_background": False,
     "tags": ["fantasy", "dark", "gothic", "mystical", "dramatic"]},

    {"id": "anime_cinematic", "name": "Anime Cinematic", "category": "portrait",
     "description": "Anime/manga style with cinematic framing and dramatic lighting — Makoto Shinkai energy.",
     "prompt_template": "{subject}, anime style, cel shading, cinematic framing, dramatic lighting, vibrant colors, Shinkai-inspired",
     "negative_template": "{style}, photorealistic, realistic skin, mundane",
     "prompt_suffix": "anime art, manga, vibrant, detailed background",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "portrait_style", "ip_adapter_scale": 0.5,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 30, "cfg": 8.0},
     "denoising_strength": 0.7, "preserve_face": False, "preserve_pose": False, "preserve_background": True,
     "tags": ["anime", "cinematic", "vibrant", "stylized", "manga"]},

    {"id": "polaroid_vintage", "name": "Polaroid Vintage", "category": "portrait",
     "description": "Instant-film aesthetic — faded colors, light leaks, square aspect, retro 70s feel.",
     "prompt_template": "{subject}, polaroid photo, vintage instant film, faded colors, light leaks, square format, 1970s nostalgia",
     "negative_template": "{style}, sharp, modern, digital, high resolution",
     "prompt_suffix": "polaroid, retro, analog, 70s aesthetic",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "face_id", "ip_adapter_scale": 0.55,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 25, "cfg": 6.5},
     "denoising_strength": 0.5, "preserve_face": True, "preserve_pose": False, "preserve_background": False,
     "tags": ["polaroid", "vintage", "retro", "analog", "nostalgic"]},

    {"id": "underwater_dream", "name": "Underwater Dream", "category": "portrait",
     "description": "Submerged portrait with flowing fabric, caustic light, and ethereal underwater atmosphere.",
     "prompt_template": "{subject}, underwater, submerged, flowing fabric, caustic light patterns, bubbles, ethereal blue tones",
     "negative_template": "{style}, dry, harsh, urban, industrial",
     "prompt_suffix": "underwater photography, surreal, dreamlike, fluid",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "face_id_plus", "ip_adapter_scale": 0.7,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 35, "cfg": 7.5},
     "denoising_strength": 0.6, "preserve_face": True, "preserve_pose": False, "preserve_background": False,
     "tags": ["underwater", "ethereal", "dreamy", "surreal", "blue"]},

    # ---------- Fashion ---------- #
    {"id": "vogue_editorial", "name": "Vogue Editorial", "category": "fashion",
     "description": "High-fashion editorial with dramatic poses, luxury fabrics, and couture styling.",
     "prompt_template": "{subject}, vogue editorial, high fashion, couture, dramatic pose, luxury fabric, designer clothing",
     "negative_template": "{style}, casual, everyday, cheap, wrinkled, poor fitting",
     "prompt_suffix": "vogue quality, fashion photography, magazine spread",
     "lora_names": ["fabric_texture_master"], "lora_weights": [0.6],
     "ip_adapter_type": "face_id_plus", "ip_adapter_scale": 0.75,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 35, "cfg": 7.5},
     "denoising_strength": 0.5, "preserve_face": True, "preserve_pose": True, "preserve_background": False,
     "tags": ["fashion", "vogue", "couture", "luxury", "editorial"]},

    {"id": "street_style_urban", "name": "Street Style Urban", "category": "fashion",
     "description": "Candid street style with urban grit — graffiti walls, concrete, effortless cool.",
     "prompt_template": "{subject}, street style, urban fashion, candid, graffiti wall, concrete, sneakers, layered outfit",
     "negative_template": "{style}, studio, formal, posed, clean background",
     "prompt_suffix": "street photography, urban, raw, authentic",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "face_id", "ip_adapter_scale": 0.6,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 30, "cfg": 7.0},
     "denoising_strength": 0.5, "preserve_face": True, "preserve_pose": True, "preserve_background": False,
     "tags": ["street", "urban", "candid", "casual", "gritty"]},

    {"id": "red_carpet_glitz", "name": "Red Carpet Glitz", "category": "fashion",
     "description": "Hollywood red-carpet moment — sequins, flash photography, glamour at its peak.",
     "prompt_template": "{subject}, red carpet, glamour pose, sequined gown, flash photography, paparazzi atmosphere, hollywood elegance",
     "negative_template": "{style}, casual, candid, daylight, plain backdrop",
     "prompt_suffix": "red carpet event, glamour photography, elegant",
     "lora_names": ["fabric_texture_master"], "lora_weights": [0.55],
     "ip_adapter_type": "face_id_plus", "ip_adapter_scale": 0.7,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 32, "cfg": 7.5},
     "denoising_strength": 0.5, "preserve_face": True, "preserve_pose": True, "preserve_background": False,
     "tags": ["red carpet", "glamour", "hollywood", "elegant", "sequins"]},

    {"id": "minimalist_scandi", "name": "Minimalist Scandi", "category": "fashion",
     "description": "Scandinavian minimalism — neutral tones, clean lines, natural light, considered restraint.",
     "prompt_template": "{subject}, scandinavian minimalism, neutral palette, clean lines, natural light, hygge",
     "negative_template": "{style}, busy, ornate, maximalist, saturated",
     "prompt_suffix": "minimal aesthetic, scandi design, considered",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "face_id", "ip_adapter_scale": 0.6,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 28, "cfg": 6.5},
     "denoising_strength": 0.45, "preserve_face": True, "preserve_pose": False, "preserve_background": True,
     "tags": ["minimalist", "scandi", "neutral", "clean", "hygge"]},

    # ---------- Landscape ---------- #
    {"id": "golden_hour_landscape", "name": "Golden Hour Landscape", "category": "landscape",
     "description": "Breathtaking landscape bathed in golden hour light with volumetric rays.",
     "prompt_template": "{subject}, golden hour, sweeping landscape, volumetric light rays, rich warm tones, dramatic sky",
     "negative_template": "{style}, dark, night, overcast, flat, dull, indoor",
     "prompt_suffix": "landscape photography, national geographic, 8k",
     "lora_names": ["cinematic_lighting_xl"], "lora_weights": [0.6],
     "ip_adapter_type": "none", "ip_adapter_scale": 0.0,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 30, "cfg": 7.0},
     "denoising_strength": 0.5, "preserve_face": False, "preserve_pose": False, "preserve_background": True,
     "tags": ["landscape", "golden hour", "dramatic", "nature", "warm"]},

    {"id": "misty_forest", "name": "Misty Forest", "category": "landscape",
     "description": "Enchanted forest with atmospheric fog, diffused light, deep emerald tones.",
     "prompt_template": "{subject}, misty forest, atmospheric fog, diffused light, emerald canopy, mossy ground, ethereal",
     "negative_template": "{style}, urban, city, desert, bright, harsh light",
     "prompt_suffix": "fantasy landscape, enchanted, deep forest, magical",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "none", "ip_adapter_scale": 0.0,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 30, "cfg": 7.5},
     "denoising_strength": 0.55, "preserve_face": False, "preserve_pose": False, "preserve_background": True,
     "tags": ["forest", "mist", "ethereal", "nature", "fantasy"]},

    {"id": "autumn_harvest", "name": "Autumn Harvest", "category": "landscape",
     "description": "Rolling autumn fields with warm foliage and rural quietness.",
     "prompt_template": "{subject}, autumn harvest, warm foliage, rural landscape, golden fields, rustic barn, soft afternoon light",
     "negative_template": "{style}, winter, snow, urban, modern",
     "prompt_suffix": "autumn photography, warm color grading, pastoral",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "none", "ip_adapter_scale": 0.0,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 28, "cfg": 7.0},
     "denoising_strength": 0.5, "preserve_face": False, "preserve_pose": False, "preserve_background": True,
     "tags": ["autumn", "rural", "warm", "pastoral", "seasonal"]},

    {"id": "moody_rain", "name": "Moody Rain", "category": "landscape",
     "description": "Rainy cityscape with neon reflections and brooding mood.",
     "prompt_template": "{subject}, rainy cityscape, neon reflections on wet pavement, moody atmosphere, blade-runner-esque",
     "negative_template": "{style}, sunny, dry, cheerful, daylight",
     "prompt_suffix": "moody photography, rainy night, neon noir",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "none", "ip_adapter_scale": 0.0,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 32, "cfg": 7.5},
     "denoising_strength": 0.55, "preserve_face": False, "preserve_pose": False, "preserve_background": True,
     "tags": ["rain", "moody", "urban", "neon", "noir"]},

    {"id": "architectural_dramatic", "name": "Architectural Dramatic", "category": "landscape",
     "description": "Dramatic architectural perspective — strong geometry, deep shadows, monumentality.",
     "prompt_template": "{subject}, dramatic architectural perspective, strong geometry, deep shadows, monumental scale, brutalist or modern",
     "negative_template": "{style}, soft, organic, pastoral, cartoon",
     "prompt_suffix": "architectural photography, geometric, monumental",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "none", "ip_adapter_scale": 0.0,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 30, "cfg": 7.0},
     "denoising_strength": 0.5, "preserve_face": False, "preserve_pose": False, "preserve_background": True,
     "tags": ["architecture", "geometric", "dramatic", "monumental", "modern"]},

    # ---------- Product ---------- #
    {"id": "luxury_product_studio", "name": "Luxury Product Studio", "category": "product",
     "description": "High-end product photography — dramatic studio lighting, reflective surfaces, luxury feel.",
     "prompt_template": "{subject}, luxury product photography, studio lighting, reflective surface, dramatic shadows, premium feel",
     "negative_template": "{style}, cheap, plastic, amateur, low quality, grainy",
     "prompt_suffix": "product shot, commercial photography, premium",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "none", "ip_adapter_scale": 0.0,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 30, "cfg": 7.5},
     "denoising_strength": 0.4, "preserve_face": False, "preserve_pose": False, "preserve_background": True,
     "tags": ["product", "luxury", "studio", "commercial", "premium"]},

    {"id": "product_lifestyle", "name": "Product Lifestyle", "category": "product",
     "description": "Product in a natural lifestyle setting — warm, aspirational, contextual.",
     "prompt_template": "{subject}, lifestyle product photography, natural setting, warm tones, aspirational mood, contextual props",
     "negative_template": "{style}, studio, sterile, cold, isolated",
     "prompt_suffix": "lifestyle shot, natural light, warm",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "none", "ip_adapter_scale": 0.0,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 28, "cfg": 7.0},
     "denoising_strength": 0.45, "preserve_face": False, "preserve_pose": False, "preserve_background": True,
     "tags": ["product", "lifestyle", "natural", "warm", "aspirational"]},

    # ---------- Abstract ---------- #
    {"id": "abstract_fluid", "name": "Abstract Fluid Art", "category": "abstract",
     "description": "Fluid acrylic pour abstraction — vivid color blooms and marbled gradients.",
     "prompt_template": "{subject}, abstract fluid art, acrylic pour, marbled gradients, vivid color blooms, swirling forms",
     "negative_template": "{style}, photorealistic, literal, figurative, mundane",
     "prompt_suffix": "abstract art, fluid painting, vibrant",
     "lora_names": [], "lora_weights": [],
     "ip_adapter_type": "none", "ip_adapter_scale": 0.0,
     "recommended_base_model": "SDXL", "generation_params": {"steps": 30, "cfg": 8.0},
     "denoising_strength": 0.75, "preserve_face": False, "preserve_pose": False, "preserve_background": False,
     "tags": ["abstract", "fluid", "vivid", "colorful", "experimental"]},
]


# --------------------------------------------------------------------------- #
# PresetLibrary — the public API
# --------------------------------------------------------------------------- #

DEFAULT_PRESETS_DIR = Path.home() / "deepseek-coder" / "PROMPT_GENERATION_101" / "presets"


class PresetLibrary:
    """Manages built-in + user-created presets.

    Built-in presets live in code (BUILTIN_PRESETS) and are always loaded.
    User presets live in ``<presets_dir>/user/<id>.json`` and are loaded
    on construction (rescan via :meth:`reload`).

    All public methods are docstring'd; the JSON shape on disk matches
    :class:`StylePreset.from_dict`.
    """

    def __init__(self, presets_dir: str | Path | None = None) -> None:
        self.presets_dir = Path(presets_dir or DEFAULT_PRESETS_DIR).expanduser()
        self.user_dir = self.presets_dir / "user"
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._presets: dict[str, StylePreset] = {}
        self.reload()

    # ---------- bootstrap ---------- #

    def reload(self) -> int:
        """Re-read all built-in + user presets from source. Returns total count."""
        self._presets = {}
        # Built-ins first.
        for spec in BUILTIN_PRESETS:
            p = StylePreset.from_dict({
                **spec,
                "is_builtin": True,
                "author": spec.get("author", "built-in"),
                "created_at": spec.get("created_at",
                                       datetime.utcnow().isoformat() + "Z"),
            })
            self._presets[p.id] = p
        # User overlays.
        for fp in sorted(self.user_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text())
                data["is_builtin"] = False
                p = StylePreset.from_dict(data)
                self._presets[p.id] = p
            except Exception:
                continue
        return len(self._presets)

    # ---------- queries ---------- #

    def list_presets(
        self,
        category: str | None = None,
        base_model: str | None = None,
        tags: list[str] | None = None,
    ) -> list[StylePreset]:
        """Filter the catalogue by category / base_model / tags (AND logic)."""
        out: list[StylePreset] = []
        tag_set = {t.lower() for t in (tags or [])}
        for p in self._presets.values():
            if category and p.category != category:
                continue
            if base_model and p.recommended_base_model not in (base_model, "any"):
                continue
            if tag_set and not tag_set.issubset({t.lower() for t in p.tags}):
                continue
            out.append(p)
        # Built-ins first, then user; alphabetical within each.
        out.sort(key=lambda x: (not x.is_builtin, x.name.lower()))
        return out

    def get_preset(self, preset_id: str) -> StylePreset | None:
        """Return a specific preset by id, or None if missing."""
        return self._presets.get(preset_id)

    # ---------- CRUD ---------- #

    def create_preset(self, preset: StylePreset | dict[str, Any]) -> str:
        """Persist a new user preset. Returns its id."""
        if isinstance(preset, dict):
            preset = StylePreset.from_dict(preset)
        preset.is_builtin = False
        if not preset.id:
            preset.id = self._make_id(preset.name)
        if not preset.created_at:
            preset.created_at = datetime.utcnow().isoformat() + "Z"
        fp = self.user_dir / f"{preset.id}.json"
        fp.write_text(json.dumps(preset.to_dict(), indent=2))
        self._presets[preset.id] = preset
        return preset.id

    def update_preset(self, preset_id: str, updates: dict[str, Any]) -> bool:
        """Merge ``updates`` into a user preset. Built-ins are immutable."""
        p = self._presets.get(preset_id)
        if p is None or p.is_builtin:
            return False
        for k, v in updates.items():
            if k in StylePreset.__dataclass_fields__:
                setattr(p, k, v)
        fp = self.user_dir / f"{preset_id}.json"
        fp.write_text(json.dumps(p.to_dict(), indent=2))
        return True

    def delete_preset(self, preset_id: str) -> bool:
        """Remove a user preset. Built-ins cannot be deleted."""
        p = self._presets.get(preset_id)
        if p is None or p.is_builtin:
            return False
        fp = self.user_dir / f"{preset_id}.json"
        try:
            fp.unlink()
        except FileNotFoundError:
            pass
        del self._presets[preset_id]
        return True

    def import_preset(self, json_path: str | Path) -> str:
        """Read a preset JSON from disk and persist as a user preset."""
        data = json.loads(Path(json_path).expanduser().read_text())
        return self.create_preset(data)

    def export_preset(self, preset_id: str, output_path: str | Path) -> str:
        """Write the preset JSON to ``output_path``. Returns the path."""
        p = self._presets.get(preset_id)
        if p is None:
            raise KeyError(f"unknown preset: {preset_id}")
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(p.to_dict(), indent=2))
        return str(out)

    # ---------- search ---------- #

    def search_presets(self, query: str, top_k: int = 10) -> list[StylePreset]:
        """Fuzzy-rank presets by relevance to ``query``.

        Combines token overlap on name + description + tags + category, with
        bonus weight for category direct match. No external NLP — pure
        token-set scoring.
        """
        if not query:
            return self.list_presets()[:top_k]
        q_tokens = _tokenize(query)
        if not q_tokens:
            return self.list_presets()[:top_k]
        scored: list[tuple[float, StylePreset]] = []
        for p in self._presets.values():
            hay = " ".join([p.name, p.description, " ".join(p.tags), p.category])
            h_tokens = _tokenize(hay)
            if not h_tokens:
                continue
            overlap = len(q_tokens & h_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(q_tokens), 1)
            # Category direct hit bonus.
            if p.category.lower() in q_tokens:
                score += 0.5
            scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:top_k]]

    # ---------- generation source ---------- #

    def preset_from_generation(
        self, generation_data: dict[str, Any],
        name: str, category: str = "portrait",
    ) -> StylePreset:
        """Build a preset from a stored generation row.

        ``generation_data`` is the dict returned by
        ``GenerationMemory.get_recent()`` (or any equivalent shape with
        ``raw_prompt`` / ``enhanced_prompt`` / ``negative_prompt`` /
        ``loras_used`` / ``seed`` / ``steps`` / ``cfg``).
        """
        prompt = (generation_data.get("enhanced_prompt")
                  or generation_data.get("raw_prompt") or "")
        # if the user wants a subject placeholder, leave one if the original
        # prompt didn't have a clear "of X" pattern.
        template = prompt
        if "{subject}" not in template:
            template = "{subject}, " + template
        loras = generation_data.get("loras_used") or []
        weights = [0.7] * len(loras) if isinstance(loras, list) else []
        gen_params = {
            "steps": int(generation_data.get("steps") or 30),
            "cfg": float(generation_data.get("cfg") or 7.5),
        }
        return StylePreset(
            id=self._make_id(name),
            name=name,
            category=category,
            description=f"User preset from generation #{generation_data.get('id', '?')}",
            prompt_template=template,
            negative_template=generation_data.get("negative_prompt") or "",
            prompt_suffix="",
            lora_names=list(loras) if isinstance(loras, list) else [],
            lora_weights=weights,
            ip_adapter_type="auto",
            ip_adapter_scale=0.7,
            recommended_base_model="SDXL",
            generation_params=gen_params,
            denoising_strength=0.5,
            preserve_face=True, preserve_pose=False, preserve_background=False,
            tags=[],
            author="user",
            created_at=datetime.utcnow().isoformat() + "Z",
            is_builtin=False,
        )

    # ---------- apply ---------- #

    def apply_preset(
        self, preset_id: str, subject: str = "",
        available_loras: set[str] | None = None,
    ) -> dict[str, Any]:
        """Resolve a preset against ``subject`` + the user's installed LoRAs.

        Returns a dict with fully-resolved generation params plus a
        ``missing_loras`` field listing referenced LoRAs that aren't present
        in ``available_loras``. Missing LoRAs are silently filtered out —
        the call never crashes on an unavailable reference.
        """
        p = self._presets.get(preset_id)
        if p is None:
            return {"ok": False, "error": f"unknown preset: {preset_id}"}

        # Fill {subject} in the prompt template.
        prompt = p.prompt_template
        if "{subject}" in prompt:
            prompt = prompt.replace("{subject}", subject.strip() or "subject")
        elif subject.strip():
            prompt = f"{subject.strip()}, {prompt}"
        if p.prompt_suffix:
            prompt = f"{prompt}, {p.prompt_suffix}"

        # Fill {style} in the negative template (lazy — try to import architect).
        detected_style = ""
        try:
            from grokkie_prompt import PromptIntelligence
            arch = PromptIntelligence()
            ar = arch.analyze_prompt(prompt)
            detected_style = ar.get("detected_style", "") if isinstance(ar, dict) else ""
        except Exception:
            pass
        neg = p.negative_template.replace("{style}", detected_style) \
              if "{style}" in p.negative_template else p.negative_template

        # Filter LoRA list against installed pool.
        missing_loras: list[str] = []
        if available_loras is not None:
            pool = {n.lower() for n in available_loras}
            chosen_names = []
            chosen_weights = []
            for n, w in zip(p.lora_names, p.lora_weights or [0.7] * len(p.lora_names)):
                if n.lower() in pool:
                    chosen_names.append(n)
                    chosen_weights.append(float(w))
                else:
                    missing_loras.append(n)
        else:
            chosen_names = list(p.lora_names)
            chosen_weights = list(p.lora_weights) if p.lora_weights \
                             else [0.7] * len(p.lora_names)

        return {
            "ok": True,
            "preset_id": p.id,
            "preset_name": p.name,
            "category": p.category,
            "prompt": prompt,
            "negative_prompt": neg,
            "lora_names": chosen_names,
            "lora_weights": chosen_weights,
            "missing_loras": missing_loras,
            "ip_adapter_type": p.ip_adapter_type,
            "ip_adapter_scale": p.ip_adapter_scale,
            "recommended_base_model": p.recommended_base_model,
            "generation_params": dict(p.generation_params),
            "denoising_strength": p.denoising_strength,
            "preserve_face": p.preserve_face,
            "preserve_pose": p.preserve_pose,
            "preserve_background": p.preserve_background,
            "tags": list(p.tags),
        }

    # ---------- internals ---------- #

    def _make_id(self, name: str) -> str:
        """Generate a unique, slug-safe id from a name."""
        base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "preset"
        cand = base
        i = 2
        while cand in self._presets:
            cand = f"{base}_{i}"
            i += 1
        return cand


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "with", "and", "or",
              "for", "to", "from", "is", "are", "was", "were", "be"}


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if t not in _STOPWORDS}


# --------------------------------------------------------------------------- #
# Module-level singleton accessor
# --------------------------------------------------------------------------- #

_singleton: PresetLibrary | None = None


def get_library(presets_dir: str | Path | None = None) -> PresetLibrary:
    """Lazy module-level PresetLibrary accessor."""
    global _singleton
    if _singleton is None:
        _singleton = PresetLibrary(presets_dir)
    return _singleton


# --------------------------------------------------------------------------- #
# CLI self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    lib = PresetLibrary()
    print(f"{len(BUILTIN_PRESETS)} built-in presets shipped.")
    for p in lib.list_presets()[:5]:
        print(f"- {p.id:<32} [{p.category}]  {p.name}")
    print("\nSearch 'dramatic dark':")
    for p in lib.search_presets("dramatic dark", top_k=3):
        print(f"  -> {p.id}  ({p.category})  {p.name}")
    print("\nApply 'cinematic_golden_hour' with subject='a woman in red':")
    print(json.dumps(lib.apply_preset("cinematic_golden_hour",
                                       subject="a woman in red"), indent=2))
