from pathlib import Path
import os

PROJECT_ROOT = Path.home() / "deepseek-coder" / "PROMPT_GENERATION_101"
LORA_ROOT = PROJECT_ROOT / "loras"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

MERGED_LORAS_DIR = OUTPUT_DIR / "merged_loras"
MERGED_LORAS_DIR.mkdir(parents=True, exist_ok=True)