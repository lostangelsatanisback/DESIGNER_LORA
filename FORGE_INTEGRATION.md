# Forge/reForge Integration Notes & Model Recommendations

Analysis of `~/FORGE_REPO/stable-diffusion-webui-reForge` and the model strategy
for LoRA Designer Studio's curation stack.

---

## 1. What reForge contains that we can use

| Capability | Where in reForge | Our integration plan |
|---|---|---|
| **Focal-point cropping** (face + entropy + corner-feature weighted POI) | `modules/textual_inversion/autocrop.py`, `scripts/postprocessing_focal_crop.py` | Pattern adopted for Phase 4 subject-centered smart-crop. We upgrade its YuNet/Haar detector to SCRFD (already in our smart curation) and reuse its POI-blending weights idea (face 0.6 / entropy 0.3 / edges 0.1). |
| **DeepBooru tagger** (booru tags, thresholded) | `modules/deepbooru.py`, model at `models/torch_deepdanbooru/model-resnet_custom_v3.pt` | Phase 3 option B. Its threshold + underscore/space formatting logic is the reference; we prefer WD14 v3 (below) but can load this exact model file from the Forge install — zero new downloads. |
| **BLIP captioner** | `modules/interrogate.py`, model at `models/BLIP/model_base_caption_capfilt_large.pth` | Phase 3 natural-language caption option (booru tags suit Pony better; BLIP is secondary). Reuse from disk. |
| **Face restoration stack** (facexlib RetinaFace + GFPGAN v1.4 / CodeFormer) | `modules/face_restoration_utils.py`, `models/GFPGAN/` | Phase 4 optional "rescue pass": restore slightly-soft faces in otherwise-perfect frames instead of rejecting them. Conservative use only — restoration artifacts can poison training. |
| **Upscalers** (ESRGAN/RealESRGAN/SwinIR via spandrel) | `modules/postprocessing.py`, auto-discovered in `models/` | Phase 4 option for upscaling sub-1024px keeper frames before packaging. |
| **Batch postprocessing pipeline pattern** | `modules/postprocessing.py`, `scripts/postprocessing_*.py` | Architectural reference for our stage-plugin design. |
| **API for generation** (txt2img/img2img endpoints) | webui `--api` mode | Phase 6 eval loop: generate test grids per epoch through the running reForge instance. |

**Model reuse paths** (set `forge_root` in your project file — already defaulted):
```
{forge_root}/models/BLIP/model_base_caption_capfilt_large.pth
{forge_root}/models/torch_deepdanbooru/model-resnet_custom_v3.pt
{forge_root}/models/GFPGAN/GFPGANv1.4.pth
{forge_root}/models/opencv/  (YuNet ONNX)
```

**Environment note:** reForge pins `transformers==4.44.0`, torch, facexlib, opencv on Python 3.10. Our core deliberately imports none of these — the studio stays stdlib-only, and AI extras run fine inside the Forge conda env *or* a separate venv. No conflict either way.

---

## 2. Recommended models — face detection & identity curation

### Tier 1 (implemented now: `pip install 'lora-studio[ai]'`)

| Role | Model | Why |
|---|---|---|
| Face detection | **SCRFD-10G** (InsightFace `buffalo_l` bundle) | Best speed/accuracy on hard angles, partial faces, low light — exactly what 2TB of candid personal video produces. Substantially better recall than reForge's YuNet/Haar. |
| Identity embedding | **ArcFace R50 w600k** (same `buffalo_l` bundle) | Industry-standard 512-d face embeddings; cosine similarity vs your anchor set cleanly separates subject from non-subject. One bundle, one download (~280MB, auto to `~/.insightface`). |
| Runtime | **onnxruntime CoreML EP** | Apple Neural Engine / GPU acceleration on M2 Max, CPU fallback automatic. |

Default identity threshold 0.35 cosine — validate against the anchor self-similarity printout from `lora-studio anchor` (your threshold should sit below the anchor's min self-sim).

### Tier 2 (upgrade options, Phase 2 continuation)

| Role | HuggingFace model | When to use |
|---|---|---|
| Higher-accuracy identity | `minchul/cvlface_adaface_ir101_webface12m` (AdaFace IR-101) | If ArcFace borderline-cases (heavy makeup, extreme angles) misclassify; ~2× slower, measurably better hard-pair separation. |
| Detection-only alt | `deepghs/real_face_detection` (YOLOv8-face) | If SCRFD misses in extreme motion blur; good ensemble second opinion. |
| Body/pose keypoints | `Ultralytics YOLOv8n-pose` | Phase 2.3 framing classification beyond face-ratio heuristic (full-body/back/POV poses where no face is visible). |
| Person segmentation | `briaai/RMBG-2.0` or `ZhengPeng7/BiRefNet` | Phase 4 background-variety analysis + optional background-augmentation experiments. |

### Tier 3 (Phase 3 captioning, for reference now)

| Role | HuggingFace model | Notes |
|---|---|---|
| Booru tagger (primary) | `SmilingWolf/wd-eva02-large-tagger-v3` | Current best WD14-family tagger; ONNX; ideal for Pony (booru-tag native). |
| Booru tagger (faster) | `SmilingWolf/wd-swinv2-tagger-v3` | ~2× faster, slightly lower mAP; good for first full-library pass. |
| Aesthetic scoring | `shadowlilac/aesthetic-shadow-v2` | Anime/illustration-trained quality scorer; complements our Laplacian metric with semantic "is this a *good* image" signal. |
| CLIP embeddings | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (via open-clip) | Phase 2.4 diversity clustering; reForge already ships open-clip-torch. |

---

## 3. How smart curation plugs into the pipeline (implemented)

```
extract ──► curate (basic: dedup/blur/exposure) ──► curate --smart ──► package
                                                        │
                       anchor (5-15 reference images) ──┘
```

1. `lora-studio anchor --anchor-dir ~/refs_of_me` — builds mean ArcFace embedding, reports self-similarity tightness.
2. `lora-studio curate --smart` — scans every `selected` frame: rejects no-face (`rejected_noface`), tiny-face (`rejected_smallface`), wrong-person (`rejected_identity`); stores per-frame `face_count, face_area, det_conf, identity_sim, framing` + raw embedding in the `detections` table (embeddings are reused later for clustering — scan once, use forever).
3. Framing classification (`closeup/portrait/upper_body/full_body`) is recorded now, enabling Phase 4 composition quotas with zero re-scanning.

Statuses are explicit and reversible — nothing is deleted, `--rescan` re-evaluates with new thresholds without re-running detection-free stages.
