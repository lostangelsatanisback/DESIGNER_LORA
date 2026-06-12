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
    # ---- Study Intelligence Layer presets ------------------------------
    "figure_study": {
        "network_dim": 32, "network_alpha": 16,
        "unet_lr": 1.5e-4, "te_lr": 1e-5,
        "batch_size": 1, "epochs": 10,
        "min_snr_gamma": 5,
        "caption_dropout_rate": 0.02,
        "notes": "Identity-locked figure study: preserves face/character "
                 "identity while improving full-body, pose, form and "
                 "proportion consistency. Low TE LR + low caption dropout "
                 "keep the identity token strong. Overfit risk: pose "
                 "collapse - validate with full-body AND face close-up "
                 "prompts every epoch (best-epoch sweep covers both).",
    },
    "fashion_editorial": {
        "network_dim": 32, "network_alpha": 16,
        "unet_lr": 1e-4, "te_lr": 5e-6,
        "batch_size": 1, "epochs": 12,
        "min_snr_gamma": 5,
        "caption_dropout_rate": 0.05,
        "notes": "Wardrobe styling, garment structure, textile detail, "
                 "editorial pose language. Moderate caption dropout so "
                 "garments bind to descriptors, not the identity token. "
                 "Validate: full-body fashion, upper-body styling, textile "
                 "detail, studio/editorial lighting.",
    },
    "balanced_study": {
        "network_dim": 48, "network_alpha": 24,
        "unet_lr": 1e-4, "te_lr": 8e-6,
        "batch_size": 1, "epochs": 10,
        "min_snr_gamma": 5,
        "caption_dropout_rate": 0.03,
        "notes": "Character + figure/fashion balance on a mixed dataset "
                 "(balanced_character_study_v1). Watch the overfit guard "
                 "in the sweep: likeness-flexibility gap > 0.15 means the "
                 "study frames are overpowering identity - stop earlier.",
    },
    "intimate_figure": {
    "network_dim": 40,
    "network_alpha": 20,
    "unet_lr": 1.2e-4,
    "te_lr": 8e-6,
    "batch_size": 1,
    "epochs": 12,
    "min_snr_gamma": 5,
    "caption_dropout_rate": 0.015,
    "notes": "Specialized for intimate figure studies. Higher emphasis on "
             "body proportion, form, skin detail and natural posing while "
             "maintaining very strong identity lock. Lower caption dropout "
             "than balanced_study. Best used with figure_study / "
             "lingerie_fashion_study recipes.",
    },
}

# Validation prompt set for study LoRAs (used by the eval matrix and
# documented in model cards): face close-up, upper-body, full-body,
# editorial pose, neutral lighting.
STUDY_VALIDATION_PROMPTS: list[str] = [
    "face close-up, natural expression, neutral lighting",
    "upper body framing, relaxed pose, soft studio lighting",
    "full body framing, balanced standing pose, form and proportion clarity",
    "full body framing, expressive editorial pose, garment structure visible",
    "neutral portrait, clean background, natural lighting",
]

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
