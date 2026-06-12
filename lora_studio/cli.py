"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from . import manifest
from .config import (
    DB_NAME, DEFAULT_QUOTA, UI_PORT, CaptionConfig, ClusterConfig, CurateConfig,
    ExtractConfig, PackageConfig, SmartCurateConfig, load_project, write_template,
)
from .curate import build_anchor, curate_generator, smart_curate_generator
from .curate.diversity import cluster_generator
from .caption import caption_generator
from .extract import pipeline_generator
from .packager import package_generator
from .util import check_dependencies


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lora-studio",
        description="LoRA Designer Studio - extraction, curation, dataset packaging",
    )
    parser.add_argument("--project", "-p", default=None,
                        help="Project file (.toml or .json). Default: built-in paths")
    sub = parser.add_subparsers(dest="command")

    p_ui = sub.add_parser("ui", help="Launch web dashboard")
    p_ui.add_argument("--port", type=int, default=None)

    sub.add_parser("doctor", help="Check dependencies")
    sub.add_parser("stats", help="Show manifest statistics")

    p_init = sub.add_parser("init", help="Write a starter project file")
    p_init.add_argument("path", nargs="?", default="project.toml")

    p_ex = sub.add_parser("extract", help="Extract frames from videos")
    p_ex.add_argument("--video-dir", action="append", default=[])
    p_ex.add_argument("--photos-dir", default=None)
    p_ex.add_argument("--output", default=None)
    p_ex.add_argument("--fps", type=float, default=0.25)
    p_ex.add_argument("--jpeg-quality", type=int, default=2)
    p_ex.add_argument("--segment-seconds", type=int, default=300)
    p_ex.add_argument("--max-side", type=int, default=0)
    p_ex.add_argument("--personfromvid", action="store_true")
    p_ex.add_argument("--no-photos", action="store_true")
    p_ex.add_argument("--photo-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    p_ex.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p_ex.add_argument("--dry-run", action="store_true")
    p_ex.add_argument("--overwrite", action="store_true")
    p_ex.add_argument("--limit-videos", type=int, default=0)

    p_ph = sub.add_parser("photos", help="Import photos only")
    p_ph.add_argument("--photos-dir", default=None)
    p_ph.add_argument("--output", default=None)
    p_ph.add_argument("--photo-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    p_ph.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p_ph.add_argument("--dry-run", action="store_true")

    p_an = sub.add_parser("anchor", help="Build identity anchor from reference images")
    p_an.add_argument("--anchor-dir", default=None,
                      help="Folder with 5-15 clear reference images of the subject")
    p_an.add_argument("--output", default=None)
    p_an.add_argument("--det-size", type=int, default=640)

    p_cu = sub.add_parser("curate", help="Score, filter and dedup frames")
    p_cu.add_argument("--output", default=None)
    p_cu.add_argument("--hamming", type=int, default=4)
    p_cu.add_argument("--min-sharpness", type=float, default=35.0)
    p_cu.add_argument("--min-brightness", type=float, default=18.0)
    p_cu.add_argument("--max-brightness", type=float, default=242.0)
    p_cu.add_argument("--workers", type=int, default=8)
    p_cu.add_argument("--rescore", action="store_true")
    p_cu.add_argument("--smart", action="store_true",
                      help="Run face+identity filtering after the basic pass")
    p_cu.add_argument("--smart-only", action="store_true",
                      help="Skip basic pass; only run face+identity filtering")
    p_cu.add_argument("--identity-threshold", type=float, default=0.35)
    p_cu.add_argument("--min-face-area", type=float, default=0.015)
    p_cu.add_argument("--rescan", action="store_true",
                      help="Re-run smart scan over previously scanned frames")

    p_cl = sub.add_parser("cluster", help="CLIP diversity clustering ([cluster] extras)")
    p_cl.add_argument("--output", default=None)
    p_cl.add_argument("--k", type=int, default=0, help="0 = auto")
    p_cl.add_argument("--batch-size", type=int, default=16)
    p_cl.add_argument("--reembed", action="store_true")

    p_ca = sub.add_parser("caption", help="WD14 auto-captioning ([caption] extras)")
    p_ca.add_argument("--output", default=None)
    p_ca.add_argument("--trigger", default=None)
    p_ca.add_argument("--class-word", default=None)
    p_ca.add_argument("--threshold", type=float, default=0.35)
    p_ca.add_argument("--char-threshold", type=float, default=0.85)
    p_ca.add_argument("--pony-prefix", action="store_true",
                      help="Prepend score_9, score_8_up, score_7_up")
    p_ca.add_argument("--blacklist", default="", help="comma-separated tags to drop")
    p_ca.add_argument("--remap", default="", help="old:new,old2:new2")
    p_ca.add_argument("--prune", default="",
                      help="permanent-trait tags to absorb into the trigger")
    p_ca.add_argument("--max-tags", type=int, default=30)
    p_ca.add_argument("--force", action="store_true")
    p_ca.add_argument("--repo-id", default="SmilingWolf/wd-swinv2-tagger-v3")

    p_bd = sub.add_parser("build", help="Recipe-driven versioned dataset build (Phase 4)")
    p_bd.add_argument("--recipe", required=True)
    p_bd.add_argument("--output", default=None)
    p_bd.add_argument("--note", default="")
    p_bd.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")

    sub.add_parser("recipes", help="List recipes defined in the project file")
    sub.add_parser("datasets", help="List versioned dataset builds")

    p_df = sub.add_parser("diff", help="Diff two dataset versions")
    p_df.add_argument("a", type=int)
    p_df.add_argument("b", type=int)
    p_df.add_argument("--output", default=None)
    p_df.add_argument("--full", action="store_true", help="list every changed frame")

    p_tr = sub.add_parser("train", help="Train a LoRA from a dataset build (Phase 5)")
    p_tr.add_argument("--dataset", type=int, required=True, help="dataset version")
    p_tr.add_argument("--preset", default="character",
                      choices=["character", "style", "outfit", "pose", "detail"])
    p_tr.add_argument("--name", default="")
    p_tr.add_argument("--output", default=None)
    p_tr.add_argument("--dry-run", action="store_true",
                      help="print the sd-scripts command without running")
    p_tr.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                      help="override any sd-scripts arg, e.g. --set max_train_epochs=12")

    sub.add_parser("runs", help="List training runs")
    sub.add_parser("presets", help="Show training preset parameters")

    p_tg = sub.add_parser("testgen", help="Single test generation (Phase 6)")
    p_tg.add_argument("--prompt", required=True)
    p_tg.add_argument("--negative", default="")
    p_tg.add_argument("--lora", action="append", default=[],
                      metavar="PATH:WEIGHT", help="repeatable; negative weights ok")
    p_tg.add_argument("--backend", choices=["forge", "diffusers"], default="forge")
    p_tg.add_argument("--forge-url", default="http://127.0.0.1:7860")
    p_tg.add_argument("--checkpoint", default="", help="diffusers backend base model")
    p_tg.add_argument("--init-image", default="", help="img2img source")
    p_tg.add_argument("--strength", type=float, default=0.6)
    p_tg.add_argument("--steps", type=int, default=28)
    p_tg.add_argument("--cfg", type=float, default=6.0)
    p_tg.add_argument("--seed", type=int, default=42)
    p_tg.add_argument("--width", type=int, default=1024)
    p_tg.add_argument("--height", type=int, default=1024)
    p_tg.add_argument("--out", default="testgen.png")
    p_tg.add_argument("--output", default=None)

    p_mx = sub.add_parser("matrix", help="Likeness/flexibility/pose/outfit/style test grids")
    p_mx.add_argument("--lora", required=True, help="LoRA path (or model stem for forge)")
    p_mx.add_argument("--label", default="")
    p_mx.add_argument("--weight", type=float, default=0.85)
    p_mx.add_argument("--categories", default="",
                      help="comma list; default all (likeness,flexibility,pose,outfit,style)")
    p_mx.add_argument("--backend", choices=["forge", "diffusers"], default="forge")
    p_mx.add_argument("--forge-url", default="http://127.0.0.1:7860")
    p_mx.add_argument("--checkpoint", default="")
    p_mx.add_argument("--steps", type=int, default=0,
                      help="0 = base-model profile default")
    p_mx.add_argument("--cfg", type=float, default=0.0,
                      help="0 = base-model profile default")
    p_mx.add_argument("--sampler", default="",
                      help="default: base-model profile sampler")
    p_mx.add_argument("--clip-skip", type=int, default=0,
                      help="0 = base-model profile default")
    p_mx.add_argument("--no-pony-prefix", dest="pony", action="store_false", default=True)
    p_mx.add_argument("--output", default=None)

    sub.add_parser("evals", help="Eval summary (avg likeness per LoRA/category)")

    p_sw = sub.add_parser("sweep", help="Best-Epoch Sweep: eval every epoch, recommend best")
    p_sw.add_argument("--run", required=True, help="run name (lora-studio runs)")
    p_sw.add_argument("--backend", default="forge", choices=["forge", "diffusers"])
    p_sw.add_argument("--forge-url", default="http://127.0.0.1:7860")
    p_sw.add_argument("--checkpoint", default="")
    p_sw.add_argument("--weight", type=float, default=0.85)
    p_sw.add_argument("--max-gap", type=float, default=0.15)
    p_sw.add_argument("--limit-epochs", type=int, default=0)
    p_sw.add_argument("--output", default=None)

    # ----- Phase 7 -----
    p_pl = sub.add_parser("pipeline", help="Full DAG: extract->curate->caption->build->train->matrix")
    p_pl.add_argument("action", choices=["run", "resume", "status"])
    p_pl.add_argument("what", nargs="?", default="full")
    p_pl.add_argument("--recipe", default="character_v1")
    p_pl.add_argument("--preset", default="character")
    p_pl.add_argument("--gate", action="append", default=[],
                      help="pause after this stage (repeatable)")
    p_pl.add_argument("--no-smart", dest="smart", action="store_false", default=True)
    p_pl.add_argument("--no-cluster", dest="cluster", action="store_false", default=True)
    p_pl.add_argument("--backend", default="forge", choices=["forge", "diffusers"])
    p_pl.add_argument("--output", default=None)

    p_wa = sub.add_parser("watch", help="Watch source folders; auto-ingest new media")
    p_wa.add_argument("--interval", type=int, default=300)
    p_wa.add_argument("--auto-curate", action="store_true")
    p_wa.add_argument("--once", action="store_true")
    p_wa.add_argument("--output", default=None)

    p_gc = sub.add_parser("gc", help="Disk cleanup (dry-run by default)")
    p_gc.add_argument("--apply", action="store_true")
    p_gc.add_argument("--keep-builds", type=int, default=2)
    p_gc.add_argument("--output", default=None)

    sub.add_parser("space", help="Disk usage dashboard")

    sub.add_parser("registry", help="List all known LoRAs (trained/merged/external)")
    p_cd = sub.add_parser("card", help="Export a model card for a LoRA")
    p_cd.add_argument("name")

    p_mg = sub.add_parser("merge", help="Merge LoRAs (weighted + block weights)")
    p_mg.add_argument("--lora", action="append", required=True,
                      metavar="PATH:WEIGHT", help="repeatable; negative weights ok")
    p_mg.add_argument("--name", default="merged")
    p_mg.add_argument("--blocks", default="",
                      help="default block multipliers, e.g. 'te=0,down=1,mid=1,up=0.5'")
    p_mg.add_argument("--preview", action="store_true",
                      help="run a likeness matrix on the merged LoRA afterwards")
    p_mg.add_argument("--backend", default="forge", choices=["forge", "diffusers"])
    p_mg.add_argument("--output", default=None)

    sub.add_parser("analyze", help="Dataset analysis + LoRA type detection")

    # Concept Control Layer
    p_cc = sub.add_parser(
        "concept", help="Concept Control Layer: explorer scan / stack / batch")
    cc_sub = p_cc.add_subparsers(dest="concept_cmd", required=True)
    cc_s = cc_sub.add_parser("scan", help="Scan LoRA folders -> influence "
                                          "profiles (manifest + sidecars)")
    cc_s.add_argument("--write-sidecars", action="store_true",
                      help="write .concept.json sidecars for new LoRAs")
    cc_k = cc_sub.add_parser("stack", help="Resolve an explained LoRA stack")
    cc_k.add_argument("--lora", action="append", default=[],
                      metavar="ID[:WEIGHT]", help="repeatable")
    cc_b = cc_sub.add_parser("batch", help="Expand a controlled variation "
                                           "grid (manifest-tracked)")
    cc_b.add_argument("--spec", required=True,
                      help="JSON grid spec file (see docs)")
    cc_b.add_argument("--mode", default="low_risk",
                      choices=["low_risk", "balanced", "creative"],
                      help="smart variation mode (caps + value ceilings)")
    cc_b.add_argument("--run", action="store_true",
                      help="generate via Forge if reachable")

    # Study Intelligence Layer
    p_st = sub.add_parser(
        "study", help="Study Intelligence Layer: classify / report / "
                      "recipes / stack / presets")
    st_sub = p_st.add_subparsers(dest="study_cmd", required=True)
    st_c = st_sub.add_parser("classify",
                             help="Classify frames into study categories")
    st_c.add_argument("--rescan", action="store_true",
                      help="re-classify frames that already have labels")
    st_sub.add_parser("report", help="Study classification statistics")
    st_r = st_sub.add_parser("recipes",
                             help="Register the 4 study dataset recipes")
    st_r.add_argument("--apply", action="store_true",
                      help="persist the recipes into the project file")
    st_k = st_sub.add_parser("stack",
                             help="Recommend a study production stack")
    st_k.add_argument("--mode", default="character_figure_study")
    st_p = st_sub.add_parser("presets",
                             help="Write the study Playground preset pack")
    st_p.add_argument("--path", default="",
                      help="presets file (default: outputs/playground_presets.json)")

    p_wz = sub.add_parser("wizard", help="Creator wizard: typed build->train->eval->card")
    p_wz.add_argument("--type", dest="lora_type", default="auto",
                      choices=["auto", "character", "style", "outfit", "pose",
                               "detail", "explicit"])
    p_wz.add_argument("--trigger", default="")
    p_wz.add_argument("--class-word", default="")
    p_wz.add_argument("--name", default="")
    p_wz.add_argument("--no-train", dest="train", action="store_false", default=True)
    p_wz.add_argument("--no-matrix", dest="matrix", action="store_false", default=True)
    p_wz.add_argument("--backend", default="forge", choices=["forge", "diffusers"])
    p_wz.add_argument("--output", default=None)

    p_cb = sub.add_parser("combo", help="One-click character+style+outfit merge")
    p_cb.add_argument("--character", required=True)
    p_cb.add_argument("--style", default="")
    p_cb.add_argument("--outfit", default="")
    p_cb.add_argument("--name", default="combo")
    p_cb.add_argument("--char-weight", type=float, default=1.0)
    p_cb.add_argument("--style-weight", type=float, default=0.4)
    p_cb.add_argument("--outfit-weight", type=float, default=0.6)
    p_cb.add_argument("--output", default=None)

    p_pk = sub.add_parser("package", help="Build kohya-style training dataset (legacy/manual)")
    p_pk.add_argument("--output", default=None)
    p_pk.add_argument("--token", default=None)
    p_pk.add_argument("--class-word", default=None)
    p_pk.add_argument("--repeats", type=int, default=10)
    p_pk.add_argument("--max-per-video", type=int, default=40)
    p_pk.add_argument("--max-total", type=int, default=0)
    p_pk.add_argument("--caption", default="")
    p_pk.add_argument("--no-captions", dest="captions", action="store_false", default=True)
    p_pk.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    p_pk.add_argument("--quota", default=DEFAULT_QUOTA,
                      help="framing quotas, e.g. 'closeup=0.3,portrait=0.3,"
                           "upper_body=0.25,full_body=0.15' (needs --max-total)")
    p_pk.add_argument("--no-quota", dest="quota", action="store_const", const="")
    p_pk.add_argument("--static-captions-only", dest="use_caption_table",
                      action="store_false", default=True)

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    prj = load_project(args.project)
    cmd = args.command or "ui"

    def out_base(flag: Optional[str]) -> Path:
        return Path(flag or prj.output_base).expanduser()

    if cmd == "init":
        path = Path(args.path)
        if path.exists():
            print(f"Refusing to overwrite existing {path}")
            return
        write_template(path)
        print(f"Wrote starter project file: {path}\nEdit it, then run: "
              f"lora-studio -p {path} ui")
    elif cmd == "ui":
        from .ui import main_ui
        main_ui(prj, getattr(args, "port", None), project_path=args.project)
    elif cmd == "doctor":
        print(json.dumps(check_dependencies(), indent=2))
        from .curate import smart_available
        ok, reason = smart_available()
        print(f"smart curation: {'available' if ok else reason}")
    elif cmd == "stats":
        base = prj.output_path
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        conn = manifest.connect(base)
        print(json.dumps(manifest.stats(conn), indent=2))
    elif cmd == "extract":
        cfg = ExtractConfig(
            output_base=out_base(args.output),
            fps=args.fps, jpeg_quality=args.jpeg_quality,
            segment_seconds=args.segment_seconds,
            use_personfromvid=args.personfromvid,
            import_photos=not args.no_photos,
            photo_import_mode=args.photo_mode,
            resume=args.resume, dry_run=args.dry_run,
            limit_videos=args.limit_videos, overwrite=args.overwrite,
            max_side=args.max_side,
        )
        vdirs = args.video_dir or prj.video_dirs
        for update in pipeline_generator(vdirs, args.photos_dir or prj.photos_dir, cfg):
            print(update)
    elif cmd == "photos":
        cfg = ExtractConfig(
            output_base=out_base(args.output),
            import_photos=True, photo_import_mode=args.photo_mode,
            resume=args.resume, dry_run=args.dry_run,
        )
        for update in pipeline_generator([], args.photos_dir or prj.photos_dir, cfg):
            print(update)
    elif cmd == "anchor":
        cfg = SmartCurateConfig(
            output_base=out_base(args.output),
            anchor_dir=Path(args.anchor_dir or prj.anchor_dir).expanduser()
            if (args.anchor_dir or prj.anchor_dir) else None,
            det_size=args.det_size,
        )
        for update in build_anchor(cfg):
            print(update)
    elif cmd == "curate":
        base = out_base(args.output)
        if not args.smart_only:
            cfg = CurateConfig(
                output_base=base,
                hamming_threshold=args.hamming,
                min_sharpness=args.min_sharpness,
                min_brightness=args.min_brightness,
                max_brightness=args.max_brightness,
                workers=args.workers, rescore=args.rescore,
            )
            for update in curate_generator(cfg):
                print(update)
        if args.smart or args.smart_only:
            scfg = SmartCurateConfig(
                output_base=base,
                identity_threshold=args.identity_threshold,
                min_face_area=args.min_face_area,
                rescan=args.rescan,
            )
            for update in smart_curate_generator(scfg):
                print(update)
    elif cmd == "pipeline":
        # NOTE: aliased import - a bare `pipeline_generator` here would shadow
        # the module-level extract.pipeline_generator across ALL branches
        from .pipeline_dag import (PipelineConfig, load_state,
                                   pipeline_generator as dag_generator,
                                   resume_generator)
        base = out_base(getattr(args, "output", None))
        if args.action == "status":
            conn = manifest.connect(base)
            state = load_state(conn)
            print(json.dumps(state, indent=2) if state
                  else "No pipeline in progress.")
        elif args.action == "resume":
            for update in resume_generator(prj, base):
                print(update)
        elif args.what == "mega":
            from .pipeline_dag import MegaConfig, mega_generator
            for update in mega_generator(prj, MegaConfig(
                    output_base=base, backend=args.backend)):
                print(update)
        else:
            cfg = PipelineConfig(
                output_base=base, recipe=args.recipe, preset=args.preset,
                gates=args.gate, smart=args.smart, cluster=args.cluster,
                matrix_backend=args.backend,
            )
            for update in dag_generator(prj, cfg):
                print(update)
    elif cmd == "watch":
        from .maintenance import WatchConfig, watch_generator
        cfg = WatchConfig(output_base=out_base(args.output),
                          interval=args.interval,
                          auto_curate=args.auto_curate, once=args.once)
        try:
            for update in watch_generator(prj, cfg):
                print(update)
        except KeyboardInterrupt:
            print("\nwatch stopped.")
    elif cmd == "gc":
        from .maintenance import GcConfig, gc_generator
        for update in gc_generator(prj, GcConfig(
            output_base=out_base(args.output), apply=args.apply,
            keep_builds=args.keep_builds,
        )):
            print(update)
    elif cmd == "space":
        from .maintenance import space_report
        for row in space_report(prj, prj.output_path):
            print(f"  {row['area']:<16} {str(row['files']):>8} files  {row['human']:>10}")
    elif cmd == "concept":
        from . import lora_explorer as lx
        cards = lx.scan_loras(prj)
        if args.concept_cmd == "scan":
            try:
                conn = manifest.connect(prj.output_path)
                n = lx.sync_profiles_to_manifest(conn, cards)
            except Exception as exc:                             # noqa: BLE001
                n = len(cards)
                print(f"(manifest unavailable - listing only: {exc})")
            print(f"Indexed {n} LoRAs from "
                  f"{', '.join(str(d) for d in lx.lora_dirs(prj)) or '(none)'}")
            for c in lx.filter_cards(cards, sort="family"):
                print(f"  {c.lora_id:<36} {c.profile.family:<12} "
                      f"risk={c.profile.identity_risk:<6} "
                      f"w={c.profile.weight_default}")
            if args.write_sidecars:
                from pathlib import Path as _P
                for c in cards:
                    sp = lx.sidecar_path(_P(c.path))
                    if not sp.exists():
                        lx.save_sidecar(_P(c.path), c.profile)
                        print(f"  sidecar -> {sp.name}")
        elif args.concept_cmd == "stack":
            from .stack_intelligence import resolve_stack
            sel, weights = [], {}
            by_id = {c.lora_id: c for c in cards}
            for spec in args.lora:
                lid, _, w = spec.partition(":")
                if lid not in by_id:
                    print(f"  unknown LoRA id: {lid}")
                    continue
                sel.append(by_id[lid])
                if w:
                    weights[lid] = float(w)
            st = resolve_stack(sel, weights, prj.base_model)
            print(f"Base model: {st.base_model}")
            if st.identity_anchor:
                a = st.identity_anchor
                print(f"  ANCHOR {a.lora_id:<30} {a.weight:<5} {a.reason}")
            for i in st.concept_loras:
                print(f"  {i.family:<12} {i.lora_id:<24} {i.weight:<5} "
                      f"blocks {i.blocks_cli}")
            print(f"Concept strength {st.total_concept_strength} | "
                  f"identity preservation {st.identity_preservation_score}")
            for w in st.warnings:
                print(f"  [{w.severity}] {w.message}")
        elif args.concept_cmd == "batch":
            import json as _json
            from pathlib import Path as _P
            from .batch_variations import (VariationGrid, expand_grid,
                                           run_batch_generator, save_batch)
            spec = _json.loads(_P(args.spec).expanduser().read_text())
            sel_ids = set(spec.pop("loras", []))
            sel = [c for c in cards if not sel_ids or c.lora_id in sel_ids]
            known = VariationGrid.__dataclass_fields__
            spec.setdefault("mode", args.mode)
            grid = VariationGrid(**{k: v for k, v in spec.items()
                                    if k in known})
            jobs = expand_grid(prj, sel, grid)
            conn = manifest.connect(prj.output_path)
            bid = save_batch(conn, prj, grid, jobs)
            print(f"Batch {bid}: {len(jobs)} variation jobs saved.")
            for j in jobs[:5]:
                print(f"  {j.variation_id} seed={j.seed} "
                      f"loras={j.loras} sliders={j.slider_state}")
            if len(jobs) > 5:
                print(f"  ... +{len(jobs) - 5} more")
            if args.run:
                for u in run_batch_generator(prj, conn, jobs):
                    print(u)
    elif cmd == "study":
        from . import study as study_mod
        if args.study_cmd == "classify":
            from .study import StudyConfig, classify_generator
            for update in classify_generator(
                    prj, StudyConfig(output_base=prj.output_path,
                                     rescan=args.rescan)):
                print(update)
        elif args.study_cmd == "report":
            conn = manifest.connect(prj.output_path)
            rep = study_mod.study_report(conn)
            print("Study Intelligence Layer report")
            for cat, n in sorted(rep["categories"].items()):
                print(f"  {cat:<32} {n}")
            print(f"  export_eligible {rep['export_eligible']} | "
                  f"needs_review {rep['needs_review']}")
            for k, v in rep.items():
                if k.endswith("_mean"):
                    print(f"  {k:<32} {v}")
        elif args.study_cmd == "recipes":
            added = study_mod.register_study_recipes(prj)
            for n in study_mod.STUDY_RECIPES:
                mark = "added" if n in added else "exists"
                print(f"  {n:<32} [{mark}]")
            if args.apply and added:
                if not args.project:
                    print("No project file (-p) - cannot persist.")
                else:
                    from pathlib import Path as _P
                    from .config import save_project
                    save_project(prj, _P(args.project))
                    print(f"Saved to {args.project}")
            elif added:
                print("(dry run - use --apply to persist into the project file)")
        elif args.study_cmd == "stack":
            rec = study_mod.suggest_study_stack(args.mode, prj.base_model)
            print(f"Mode: {rec['mode']}   profile: {rec['profile']}")
            for s in rec["stack"]:
                print(f"  {s['type']:<18} {s['role']:<9} w={s['weight']:<5} "
                      f"blocks {s['blocks_cli']}")
            print("Merge order:", " -> ".join(rec["merge_order"]))
            for w in rec["warnings"]:
                print(f"  ! {w}")
        elif args.study_cmd == "presets":
            from pathlib import Path as _P
            target = study_mod.write_study_presets(
                prj, _P(args.path) if args.path else None)
            print(f"Study preset pack written -> {target}")
    elif cmd == "registry":
        from .registry import build_registry, best_likeness
        conn = manifest.connect(prj.output_path)
        entries = build_registry(prj, conn)
        if not entries:
            print("No LoRAs known yet. Train one or drop .safetensors in lora_output_dir.")
        for e in entries:
            lk = best_likeness(e)
            print(f"  {e['name']:<32} [{e['kind']:<8}] "
                  f"likeness={lk if lk is not None else '-':<8} "
                  f"files={len(e.get('files', []))}")
    elif cmd == "card":
        from .registry import model_card
        conn = manifest.connect(prj.output_path)
        path = model_card(prj, conn, args.name)
        print(f"Model card: {path}" if path else f"Unknown LoRA: {args.name}")
    elif cmd == "sweep":
        from .sweep import SweepConfig, sweep_generator
        cfg = SweepConfig(
            output_base=out_base(args.output), run=args.run,
            backend=args.backend, forge_url=args.forge_url,
            checkpoint=args.checkpoint, weight=args.weight,
            max_gap=args.max_gap, limit_epochs=args.limit_epochs,
        )
        for update in sweep_generator(prj, cfg):
            print(update)
    elif cmd == "analyze":
        from .wizard import analyze, detect_type
        conn = manifest.connect(prj.output_path)
        a = analyze(conn)
        print(json.dumps(a, indent=2))
        print("\nType detection:")
        for r in detect_type(a):
            print(f"  {r['type']:<10} {r['score']:.2f}  ({r['reason']})")
    elif cmd == "wizard":
        from .wizard import WizardConfig, wizard_generator
        cfg = WizardConfig(
            output_base=out_base(args.output), lora_type=args.lora_type,
            trigger=args.trigger, class_word=args.class_word, name=args.name,
            train=args.train, matrix=args.matrix, matrix_backend=args.backend,
        )
        for update in wizard_generator(prj, cfg):
            print(update)
    elif cmd == "merge":
        from .eval.matrix import parse_lora_specs
        from .merge import MergeConfig, merge_generator, parse_block_string
        loras = []
        for spec in args.lora:
            loras.extend(parse_lora_specs(spec))
        cfg = MergeConfig(
            output_base=out_base(args.output), loras=loras,
            output_name=args.name,
            default_blocks=parse_block_string(args.blocks),
        )
        for update in merge_generator(prj, cfg):
            print(update)
        if args.preview:
            from .merge import safe_slug as _slug
            from .eval.matrix import MatrixConfig, matrix_generator
            lora_dir = Path(prj.lora_output_dir
                            or (prj.output_path / "LORA_OUTPUT")).expanduser()
            merged = lora_dir / f"{_slug(args.name)}.safetensors"
            if merged.exists():
                for update in matrix_generator(prj, MatrixConfig(
                    output_base=out_base(args.output), lora=str(merged),
                    label=f"{_slug(args.name)}_preview",
                    categories=["likeness"], backend=args.backend,
                    checkpoint=prj.base_model,
                )):
                    print(update)
    elif cmd == "combo":
        from .merge import combo_generator
        for update in combo_generator(
            prj, out_base(args.output), character=args.character,
            style=args.style, outfit=args.outfit, name=args.name,
            char_w=args.char_weight, style_w=args.style_weight,
            outfit_w=args.outfit_weight,
        ):
            print(update)
    elif cmd == "testgen":
        from .eval.matrix import parse_lora_specs
        loras = []
        for spec in args.lora:
            loras.extend(parse_lora_specs(spec))
        if args.backend == "forge":
            from .eval.forge_api import ForgeClient
            client = ForgeClient(args.forge_url)
            if not client.alive():
                print(f"Forge API not reachable at {args.forge_url} (start with --api)")
                return
            if args.init_image:
                png = client.img2img(
                    Path(args.init_image).expanduser().read_bytes(),
                    prompt=args.prompt, negative=args.negative,
                    strength=args.strength, steps=args.steps, cfg=args.cfg,
                    seed=args.seed, loras=loras,
                )
            else:
                png = client.txt2img(
                    prompt=args.prompt, negative=args.negative, steps=args.steps,
                    cfg=args.cfg, width=args.width, height=args.height,
                    seed=args.seed, loras=loras,
                )
            Path(args.out).write_bytes(png)
        else:
            from .eval.pipeline import diffusers_available, get_pipeline
            ok, reason = diffusers_available()
            if not ok:
                print(reason)
                return
            ckpt = args.checkpoint or prj.base_model
            tp = get_pipeline(ckpt, "", loras)
            if args.init_image:
                from PIL import Image
                init = Image.open(Path(args.init_image).expanduser()).convert("RGB")
                img = tp.img2img(init, args.prompt, args.negative,
                                 args.strength, args.steps, args.cfg, args.seed)
            else:
                img = tp.txt2img(args.prompt, args.negative, args.steps,
                                 args.cfg, args.width, args.height, args.seed)
            img.save(args.out)
        print(f"Saved: {args.out}")
    elif cmd == "matrix":
        from .eval.matrix import CATEGORIES, MatrixConfig, matrix_generator
        cats = ([c.strip() for c in args.categories.split(",") if c.strip()]
                or list(CATEGORIES))
        cfg = MatrixConfig(
            output_base=out_base(args.output), lora=args.lora, label=args.label,
            categories=cats, backend=args.backend, forge_url=args.forge_url,
            checkpoint=args.checkpoint, lora_weight=args.weight,
            steps=args.steps, cfg=args.cfg, sampler=args.sampler,
            clip_skip=args.clip_skip, pony_prefix=args.pony,
        )
        for update in matrix_generator(prj, cfg):
            print(update)
    elif cmd == "evals":
        from .eval.matrix import eval_summary
        conn = manifest.connect(prj.output_path)
        rows = eval_summary(conn)
        if not rows:
            print("No evals yet. Run: lora-studio matrix --lora <path>")
        cur = None
        for r in rows:
            if r["label"] != cur:
                cur = r["label"]
                print(f"  {cur}")
            print(f"    {r['category']:<12} n={r['n']:<4} avg_likeness={r['avg_likeness']}")
    elif cmd == "presets":
        from .train import PRESETS
        for name, p in PRESETS.items():
            print(f"  {name:<10} dim {p['network_dim']}/{p['network_alpha']:<3} "
                  f"lr {p['unet_lr']}/{p['te_lr']} batch {p['batch_size']} "
                  f"epochs {p['epochs']}  - {p['notes']}")
    elif cmd == "train":
        from .train.kohya import TrainConfig, train_generator
        overrides = {}
        for kv in args.set:
            if "=" in kv:
                k, v = kv.split("=", 1)
                overrides[k.strip()] = v.strip()
        cfg = TrainConfig(
            output_base=out_base(args.output),
            dataset_version=args.dataset, preset=args.preset,
            name=args.name, overrides=overrides, dry_run=args.dry_run,
        )
        for update in train_generator(prj, cfg):
            print(update)
    elif cmd == "runs":
        from .train.kohya import list_runs
        conn = manifest.connect(prj.output_path)
        rows = list_runs(conn)
        if not rows:
            print("No training runs yet. Run: lora-studio train --dataset N --preset character")
        for r in rows:
            print(f"  #{r['run_id']:<3} {r['name']:<28} v{r['dataset_version']:03d} "
                  f"{r['preset']:<9} {r['status']:<9} "
                  f"step {r['step']}/{r['total_steps']} loss {r['last_loss']}")
    elif cmd == "recipes":
        if not prj.recipes:
            print("No recipes defined. Add [recipes.NAME] sections to the project file\n"
                  "or regenerate a template with: lora-studio init")
        for name, r in sorted(prj.recipes.items()):
            kind = r.get("type", "custom")
            extra = (f"concepts={r['concepts']}" if r.get("concepts")
                     else f"repeats={r.get('repeats', 10)} max_total={r.get('max_total', 0)}")
            print(f"  {name:<18} [{kind}] {extra}")
    elif cmd == "build":
        from .builder import BuildConfig, build_generator
        cfg = BuildConfig(
            output_base=out_base(args.output), recipe=args.recipe,
            note=args.note, link_mode=args.link_mode,
        )
        for update in build_generator(prj, cfg):
            print(update)
    elif cmd == "datasets":
        from .builder import list_datasets
        conn = manifest.connect(prj.output_path)
        rows = list_datasets(conn)
        if not rows:
            print("No dataset builds yet. Run: lora-studio build --recipe <name>")
        for d in rows:
            print(f"  v{d['version']:03d}  {d['recipe']:<16} train={d['train']:<5} "
                  f"val={d['val']:<4} hash={d['hash']}  {d['built_at']}  {d['note']}")
    elif cmd == "diff":
        from .builder import diff_datasets
        conn = manifest.connect(out_base(args.output))
        try:
            d = diff_datasets(conn, args.a, args.b)
        except KeyError as exc:
            print(f"Error: {exc}")
            return
        print(d["summary"])
        print(f"concept mix: {d['concept_mix_a']} -> {d['concept_mix_b']}")
        if args.full:
            for label in ("added", "removed", "caption_changed", "split_changed"):
                for fid, concept in d[label]:
                    print(f"  {label:<16} {fid} ({concept})")
    elif cmd == "cluster":
        cfg = ClusterConfig(
            output_base=out_base(args.output), k=args.k,
            batch_size=args.batch_size, reembed=args.reembed,
        )
        for update in cluster_generator(cfg):
            print(update)
    elif cmd == "caption":
        cfg = CaptionConfig(
            output_base=out_base(args.output),
            trigger=args.trigger or prj.trigger_token,
            class_word=args.class_word or prj.class_word,
            threshold=args.threshold, char_threshold=args.char_threshold,
            pony_prefix=args.pony_prefix, blacklist=args.blacklist,
            remap=args.remap, prune=args.prune, max_tags=args.max_tags,
            force=args.force, repo_id=args.repo_id,
        )
        for update in caption_generator(cfg):
            print(update)
    elif cmd == "package":
        cfg = PackageConfig(
            output_base=out_base(args.output),
            token=args.token or prj.trigger_token,
            class_word=args.class_word or prj.class_word,
            repeats=args.repeats, max_per_video=args.max_per_video,
            max_total=args.max_total, caption_text=args.caption,
            write_captions=args.captions, link_mode=args.link_mode,
            quota=args.quota, use_caption_table=args.use_caption_table,
        )
        for update in package_generator(cfg):
            print(update)
