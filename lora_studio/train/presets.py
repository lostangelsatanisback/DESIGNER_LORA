"""M2-Max-32GB preset library for SDXL/Pony LoRA training (Phase 5.2).

Values follow the roadmap table. All presets share the low-VRAM common args:
fp16, gradient checkpointing, memory-efficient attention, cached latents,
cosine schedule. te_lr = 0 means the text encoder is frozen
(--network_train_unet_only).
"""

from __future__ import annotations

# Per-LoRA-type presets (roadmap Phase 5.2 table)
PRESETS: dict[str, dict] = {
    "character": {
        "network_dim": 32, "network_alpha": 16,
        "unet_lr": 1e-4, "te_lr": 5e-5,
        "batch_size": 1, "epochs": 10,
        "min_snr_gamma": 5,
        "notes": "cosine, ~2500 steps target on a 250-400 image dataset",
    },
    "style": {
        "network_dim": 16, "network_alpha": 8,
        "unet_lr": 8e-5, "te_lr": 0.0,
        "batch_size": 2, "epochs": 14,
        "min_snr_gamma": 5,
        "notes": "TE frozen, more epochs",
    },
    "outfit": {
        "network_dim": 16, "network_alpha": 8,
        "unet_lr": 1e-4, "te_lr": 4e-5,
        "batch_size": 1, "epochs": 10,
        "min_snr_gamma": 5,
        "notes": "tight captions matter more than params here",
    },
    "pose": {
        "network_dim": 8, "network_alpha": 4,
        "unet_lr": 1e-4, "te_lr": 0.0,
        "batch_size": 2, "epochs": 8,
        "min_snr_gamma": 5,
        "notes": "low dim; consider regularization images",
    },
    "detail": {
        "network_dim": 64, "network_alpha": 32,
        "unet_lr": 5e-5, "te_lr": 2e-5,
        "batch_size": 1, "epochs": 6,
        "min_snr_gamma": 5,
        "notes": "short runs, high-res subset (use a detail_v1 recipe)",
    },
}

# Shared low-VRAM / MPS-safe arguments
COMMON_ARGS: dict[str, object] = {
    "mixed_precision": "fp16",
    "save_precision": "fp16",
    "lr_scheduler": "cosine",
    "optimizer_type": "AdamW",
    "resolution": "1024,1024",
    "save_every_n_epochs": 1,
    "seed": 42,
    "caption_extension": ".txt",
    "max_data_loader_n_workers": 2,
}

COMMON_FLAGS = [
    "cache_latents",
    "gradient_checkpointing",
    "mem_eff_attn",
    "enable_bucket",
]

MPS_ENV = {
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
}
