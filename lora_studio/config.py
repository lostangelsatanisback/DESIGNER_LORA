"""Project configuration: constants, stage configs, project-file loader.

A *project file* (TOML on py3.11+, JSON everywhere) replaces editing
constants in code. Generate a starter with:  lora-studio init
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib  # py3.11+
    HAVE_TOML = True
except Exception:
    HAVE_TOML = False

# -----------------------------
# Constants
# -----------------------------

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff"}

DB_NAME = "extractor_manifest.sqlite3"
LOG_NAME = "extractor.log"
UI_PORT = 7861  # 7860 is usually taken by Forge/A1111

DEFAULT_VIDEO_DIRS = [
    "/Volumes/CURATOR_SSD/SPOOKUMS_PROJECT/FINAL MEDIA/VIDEOS/complete",
    "/Volumes/CURATOR_SSD/SPOOKUMS_PROJECT/FINAL MEDIA/VIDEOS/ExtraLarge",
    "/Volumes/CURATOR_SSD/SPOOKUMS_PROJECT/FINAL MEDIA/VIDEOS/JAN",
    "/Volumes/CURATOR_SSD/SPOOKUMS_PROJECT/FINAL MEDIA/VIDEOS/JAN_FEB",
    "/Volumes/CURATOR_SSD/SPOOKUMS_PROJECT/FINAL MEDIA/VIDEOS/Large",
    "/Volumes/CURATOR_SSD/SPOOKUMS_PROJECT/FINAL MEDIA/VIDEOS/MDMA",
    "/Volumes/CURATOR_SSD/SPOOKUMS_PROJECT/FINAL MEDIA/VIDEOS/Medium",
    "/Volumes/CURATOR_SSD/SPOOKUMS_PROJECT/FINAL MEDIA/VIDEOS/Small",
]
DEFAULT_PHOTOS_DIR = "/Volumes/CURATOR_SSD/SPOOKUMS_PROJECT/FINAL MEDIA/PHOTOS"
DEFAULT_OUTPUT_BASE = "/Volumes/CURATOR_SSD/SPOOKUMS_PROJECT/EXTRACTED_FRAMES"

# Forge/reForge install whose already-downloaded models we can reuse
# (BLIP captioner, DeepBooru tagger, GFPGAN, YuNet) in later phases.
DEFAULT_FORGE_ROOT = "/Users/snoozeybnuiadmin/FORGE_REPO/stable-diffusion-webui-reForge"


# -----------------------------
# Stage configs
# -----------------------------

@dataclass
class ExtractConfig:
    output_base: Path
    fps: float = 0.25
    jpeg_quality: int = 2
    segment_seconds: int = 300
    use_personfromvid: bool = False
    import_photos: bool = True
    photo_import_mode: str = "hardlink"
    resume: bool = True
    dry_run: bool = False
    limit_videos: int = 0
    overwrite: bool = False
    max_side: int = 0


@dataclass
class CurateConfig:
    output_base: Path
    hamming_threshold: int = 4
    min_sharpness: float = 35.0
    min_brightness: float = 18.0
    max_brightness: float = 242.0
    workers: int = 8
    rescore: bool = False


@dataclass
class SmartCurateConfig:
    output_base: Path
    anchor_dir: Optional[Path] = None      # reference images of the subject
    identity_threshold: float = 0.35       # cosine sim vs anchor (ArcFace space)
    min_face_area: float = 0.015           # face bbox area / image area floor
    det_size: int = 640
    batch_log_every: int = 100
    rescan: bool = False                   # re-run detection on already-scanned frames


@dataclass
class ClusterConfig:
    output_base: Path
    k: int = 0                 # 0 = auto: sqrt(n/2), clamped 2..40
    batch_size: int = 16
    max_iter: int = 30
    reembed: bool = False      # recompute existing CLIP embeddings


@dataclass
class CaptionConfig:
    output_base: Path
    trigger: str = "ohwx"
    class_word: str = "person"
    threshold: float = 0.35           # general-tag confidence floor
    char_threshold: float = 0.85      # character-tag confidence floor
    pony_prefix: bool = False         # prepend score_9, score_8_up, ...
    blacklist: str = ""               # comma-separated tags to drop
    remap: str = ""                   # "old:new,old2:new2"
    prune: str = ""                   # permanent-trait tags to absorb into trigger
    max_tags: int = 30
    force: bool = False               # re-caption frames with edited=0
    batch_log_every: int = 50
    repo_id: str = "SmilingWolf/wd-swinv2-tagger-v3"


DEFAULT_QUOTA = "closeup=0.30,portrait=0.30,upper_body=0.25,full_body=0.15"


@dataclass
class PackageConfig:
    output_base: Path
    token: str = "ohwx"
    class_word: str = "person"
    repeats: int = 10
    max_per_video: int = 40
    max_total: int = 0
    link_mode: str = "hardlink"
    write_captions: bool = True
    caption_text: str = ""
    dataset_name: str = "DATASET"
    quota: str = DEFAULT_QUOTA        # framing quotas; "" disables
    use_caption_table: bool = True    # per-frame WD14 captions when present


# -----------------------------
# Project file
# -----------------------------

@dataclass
class Project:
    name: str = "SPOOKUMS_STUDIO"
    video_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_VIDEO_DIRS))
    photos_dir: str = DEFAULT_PHOTOS_DIR
    output_base: str = DEFAULT_OUTPUT_BASE
    trigger_token: str = "ohwx"
    class_word: str = "person"
    anchor_dir: str = ""                   # subject reference images for identity curation
    forge_root: str = DEFAULT_FORGE_ROOT
    ui_port: int = UI_PORT
    recipes: dict = field(default_factory=dict)   # [recipes.NAME] sections
    # Phase 5 training
    sd_scripts_dir: str = ""               # kohya sd-scripts checkout
    base_model: str = ""                   # Pony Diffusion V6 XL .safetensors path
    lora_output_dir: str = ""              # default: {output_base}/LORA_OUTPUT

    @property
    def output_path(self) -> Path:
        return Path(self.output_base).expanduser()


PROJECT_TEMPLATE = """# LoRA Designer Studio project file
name = "SPOOKUMS_STUDIO"

# Source media (never modified)
video_dirs = [
{video_dirs}
]
photos_dir = "{photos_dir}"

# All derived data (frames, manifest, datasets) goes here
output_base = "{output_base}"

# Training identity
trigger_token = "ohwx"
class_word = "person"

# 5-15 clear reference images of the subject, for identity curation
anchor_dir = ""

# Existing Forge/reForge install (models reused in later phases)
forge_root = "{forge_root}"

ui_port = 7861

# Phase 5 training (kohya sd-scripts)
sd_scripts_dir = ""        # e.g. /Users/you/sd-scripts
base_model = ""            # e.g. /Volumes/.../ponyDiffusionV6XL.safetensors
lora_output_dir = ""       # default: {output_base}/LORA_OUTPUT

# ---------------------------------------------------------------
# Dataset recipes (Phase 4): build with  lora-studio build --recipe NAME
# Filters: framing="closeup,portrait" | include_clusters="2,5" |
#          exclude_clusters="3" | min_identity=0.45 | min_sharpness=50
# ---------------------------------------------------------------

[recipes.character_v1]
type = "character"
repeats = 10
max_total = 400
max_per_video = 40
quota = "closeup=0.30,portrait=0.30,upper_body=0.25,full_body=0.15"
val_fraction = 0.05

[recipes.style_v1]
type = "style"
repeats = 4
max_total = 600
max_per_video = 20
quota = ""

[recipes.outfit_v1]
type = "outfit"
repeats = 8
max_total = 150
framing = "upper_body,full_body"
include_clusters = ""        # set after clustering, e.g. "2,5"
token = "outfitx"
class_word = "outfit"

[recipes.pose_v1]
type = "pose"
repeats = 6
max_total = 200
framing = "full_body"
quota = ""

[recipes.detail_v1]
type = "detail"
repeats = 12
max_total = 120
framing = "closeup"
min_sharpness = 80

# multi-concept build: each listed recipe becomes its own folder
[recipes.combo_v1]
concepts = "character_v1,outfit_v1"
val_fraction = 0.05
"""


def write_template(path: Path) -> None:
    body = PROJECT_TEMPLATE.format(
        video_dirs="\n".join(f'    "{d}",' for d in DEFAULT_VIDEO_DIRS),
        photos_dir=DEFAULT_PHOTOS_DIR,
        output_base=DEFAULT_OUTPUT_BASE,
        forge_root=DEFAULT_FORGE_ROOT,
    )
    path.write_text(body, encoding="utf-8")


def _parse_flat_toml(text: str) -> dict:
    """Minimal TOML subset parser (Python 3.10, no tomllib): scalar keys,
    multi-line string arrays, and [table.subtable] section headers. Only
    intended for files written by `lora-studio init` / the Settings tab."""
    data: dict = {}
    current: dict = data
    key: Optional[str] = None
    in_array = False
    items: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith('"') else raw.strip()
        if not line:
            continue
        if not in_array and line.startswith("[") and line.endswith("]"):
            current = data
            for part in line[1:-1].strip().split("."):
                part = part.strip()
                if part:
                    current = current.setdefault(part, {})
            continue
        if in_array:
            if line.startswith("]"):
                current[key] = items
                in_array = False
                key = None
                items = []
            else:
                item = line.rstrip(",").strip()
                if item.startswith('"') and item.endswith('"'):
                    items.append(item[1:-1])
            continue
        if "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        if v == "[":
            key = k
            in_array = True
            items = []
        elif v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            current[k] = [
                s.strip().strip('"') for s in inner.split(",") if s.strip()
            ] if inner else []
        elif v.startswith('"') and v.endswith('"'):
            current[k] = v[1:-1]
        elif v in ("true", "false"):
            current[k] = v == "true"
        else:
            try:
                current[k] = int(v)
            except ValueError:
                try:
                    current[k] = float(v)
                except ValueError:
                    current[k] = v
    return data


def dumps_toml(prj: Project) -> str:
    """Serialize a Project to TOML (flat schema, no external deps)."""
    lines = [f'name = "{prj.name}"', "", "video_dirs = ["]
    lines += [f'    "{d}",' for d in prj.video_dirs]
    lines += [
        "]",
        f'photos_dir = "{prj.photos_dir}"',
        f'output_base = "{prj.output_base}"',
        f'trigger_token = "{prj.trigger_token}"',
        f'class_word = "{prj.class_word}"',
        f'anchor_dir = "{prj.anchor_dir}"',
        f'forge_root = "{prj.forge_root}"',
        f"ui_port = {prj.ui_port}",
        f'sd_scripts_dir = "{prj.sd_scripts_dir}"',
        f'base_model = "{prj.base_model}"',
        f'lora_output_dir = "{prj.lora_output_dir}"',
        "",
    ]
    for rname in sorted(prj.recipes or {}):
        lines.append(f"[recipes.{rname}]")
        for k, v in (prj.recipes[rname] or {}).items():
            if isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            else:
                lines.append(f'{k} = "{v}"')
        lines.append("")
    return "\n".join(lines)


def save_project(prj: Project, path: Path) -> None:
    if path.suffix.lower() == ".json":
        from dataclasses import asdict
        path.write_text(json.dumps(asdict(prj), indent=2), encoding="utf-8")
    else:
        path.write_text(dumps_toml(prj), encoding="utf-8")


def load_project(path: Optional[str]) -> Project:
    """Load project file (TOML or JSON). Missing path -> defaults."""
    if not path:
        return Project()
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Project file not found: {p}")
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() in (".toml", ".tml"):
        text = p.read_text(encoding="utf-8")
        if HAVE_TOML:
            data = tomllib.loads(text)
        else:
            # py3.10 fallback: our project schema is flat (str/int/list[str])
            data = _parse_flat_toml(text)
    else:
        raise ValueError(f"Unsupported project file type: {p.suffix}")
    known = {f for f in Project.__dataclass_fields__}
    return Project(**{k: v for k, v in data.items() if k in known})
