"""Zero-dependency web dashboard (stdlib http.server). No Gradio, no conflicts.

v2.1: job queue, frame review grid with thumbnails, project editor,
full-pipeline runner.
"""

from __future__ import annotations

import itertools
import json
import logging
import socket
import subprocess
import sys
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

from .. import manifest
from ..config import (
    DB_NAME, DEFAULT_QUOTA, CaptionConfig, ClusterConfig, CurateConfig,
    ExtractConfig, PackageConfig, Project, SmartCurateConfig, save_project,
)
from ..caption import caption_generator
from ..curate import build_anchor, curate_generator, smart_curate_generator
from ..curate.diversity import cluster_generator
from ..extract import pipeline_generator
from ..packager import package_generator
from ..util import HAVE_PIL, LOG_BUFFER, now_iso

if HAVE_PIL:
    from PIL import Image

THUMB_SIZE = 256


# -----------------------------
# Job queue
# -----------------------------

class Job:
    _ids = itertools.count(1)

    def __init__(self, stage: str, factory: Callable) -> None:
        self.id = next(Job._ids)
        self.stage = stage
        self.factory = factory
        self.status = "queued"          # queued|running|done|error|stopped|cancelled
        self.output = ""
        self.created_at = now_iso()
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None

    def brief(self) -> dict:
        return {
            "id": self.id, "stage": self.stage, "status": self.status,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobQueue:
    """Sequential background worker. One job runs at a time; the rest wait.
    Stop is cooperative: honoured at the running job's next progress yield."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.jobs: list[Job] = []
        self.current: Optional[Job] = None
        self.stop_requested = False
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def submit(self, stage: str, factory: Callable) -> int:
        with self.cond:
            job = Job(stage, factory)
            self.jobs.append(job)
            self.cond.notify()
            return job.id

    def cancel(self, job_id: int) -> bool:
        with self.lock:
            for j in self.jobs:
                if j.id == job_id and j.status == "queued":
                    j.status = "cancelled"
                    return True
        return False

    def stop_current(self) -> None:
        with self.lock:
            self.stop_requested = True

    def _next(self) -> Optional[Job]:
        for j in self.jobs:
            if j.status == "queued":
                return j
        return None

    def _loop(self) -> None:
        while True:
            with self.cond:
                job = self._next()
                while job is None:
                    self.cond.wait(timeout=1.0)
                    job = self._next()
                self.current = job
                self.stop_requested = False
                job.status = "running"
                job.started_at = now_iso()
            try:
                for update in job.factory():
                    with self.lock:
                        job.output = update
                        if self.stop_requested:
                            job.output += "\n\n[STOPPED by user]"
                            job.status = "stopped"
                            break
                else:
                    with self.lock:
                        job.status = "done"
            except Exception as exc:
                with self.lock:
                    job.output += f"\n\n[ERROR] {exc}"
                    job.status = "error"
                logging.exception("Job %s failed", job.stage)
            finally:
                with self.lock:
                    job.finished_at = now_iso()
                    self.current = None
                    # if stopped, drain remaining queued jobs of a pipeline
                    if job.status == "stopped":
                        for j in self.jobs:
                            if j.status == "queued":
                                j.status = "cancelled"
                    # keep history bounded
                    finished = [j for j in self.jobs
                                if j.status in ("done", "error", "stopped", "cancelled")]
                    for old in finished[:-20]:
                        self.jobs.remove(old)

    def state(self) -> dict:
        with self.lock:
            cur = self.current
            return {
                "running": cur is not None,
                "stage": cur.stage if cur else None,
                "output": cur.output if cur else
                (self.jobs[-1].output if self.jobs else ""),
                "queue": [j.brief() for j in self.jobs[-30:]],
            }


QUEUE = JobQueue()


# -----------------------------
# Child app manager (Playground / Factory run as managed Gradio subprocesses
# embedded via iframes; deps live in the user's Forge env, not in core)
# -----------------------------

APPS: dict[str, dict] = {
    "playground": {"script": "grokkie_playground.py", "port": 7870, "proc": None,
                   "needs_project": False, "args": []},
    "factory": {"script": "grokkie_dataset_factory.py", "port": 7875, "proc": None,
                "needs_project": True, "args": []},
    # reForge with API enabled - the eval/matrix/sweep generation backend
    "forge": {"script": "launch.py", "port": 7860, "proc": None,
              "needs_project": False, "args": ["--api"]},
}

FORGE_ROOT: Optional[str] = None
# Visual LoRA Explorer preview index: (lora_id, level) -> verified path.
# Preview serving goes ONLY through this index - no path traversal.
CC_PREVIEW_INDEX: dict[tuple, str] = {}     # set by main_ui from the project file


def _app_script(name: str) -> Optional[Path]:
    fname = APPS[name]["script"]
    if name == "forge":
        if FORGE_ROOT:
            cand = Path(FORGE_ROOT).expanduser() / fname
            if cand.exists():
                return cand
        return None
    for root in (Path(__file__).resolve().parents[2], Path.cwd()):
        cand = root / fname
        if cand.exists():
            return cand
    return None


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _app_log_path(name: str) -> Optional[Path]:
    script = _app_script(name)
    if script is None:
        return None
    return script.parent / "outputs" / f"{name}_launch.log"


def app_status() -> dict:
    out = {}
    for name, info in APPS.items():
        script = _app_script(name)
        running = _port_open(info["port"])
        proc = info["proc"]
        crashed = (proc is not None and proc.poll() is not None and not running)
        entry = {
            "script_found": script is not None,
            "port": info["port"],
            "running": running,
            "managed": proc is not None and proc.poll() is None,
            "crashed": crashed,
            "log_tail": "",
        }
        if crashed:
            log = _app_log_path(name)
            if log and log.exists():
                try:
                    entry["log_tail"] = "\n".join(
                        log.read_text(errors="replace").splitlines()[-12:])
                except Exception:
                    pass
        out[name] = entry
    return out


def start_app(name: str, project_path: Optional[str]) -> dict:
    info = APPS.get(name)
    if info is None:
        return {"ok": False, "error": f"unknown app {name}"}
    if _port_open(info["port"]):
        return {"ok": True, "note": "already running"}
    script = _app_script(name)
    if script is None:
        return {"ok": False,
                "error": f"{info['script']} not found next to the package"}
    if name == "forge":
        cmd = [sys.executable, str(script), *info["args"]]
    else:
        cmd = [sys.executable, str(script), "--port", str(info["port"]),
               *info["args"]]
        if info["needs_project"] and project_path:
            cmd += ["-p", project_path]
    try:
        log = _app_log_path(name)
        log.parent.mkdir(parents=True, exist_ok=True)
        logf = open(log, "w", encoding="utf-8")          # noqa: SIM115
        info["proc"] = subprocess.Popen(
            cmd, cwd=str(script.parent), stdout=logf, stderr=subprocess.STDOUT,
        )
        return {"ok": True, "note": f"starting on :{info['port']} "
                                    "(first start loads heavy deps - give it ~20s)"}
    except Exception as exc:                                    # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def stop_app(name: str) -> dict:
    info = APPS.get(name)
    if info and info["proc"] is not None and info["proc"].poll() is None:
        info["proc"].terminate()
        info["proc"] = None
        return {"ok": True}
    return {"ok": True, "note": "not managed by hub (or already stopped)"}


def _chain(*factories):
    def gen():
        for f in factories:
            yield from f()
    return gen


# -----------------------------
# Thumbnails
# -----------------------------

def ensure_thumb(output_base: Path, frame_id: str, src: Path) -> Optional[Path]:
    tdir = output_base / ".thumbs"
    tdir.mkdir(parents=True, exist_ok=True)
    dest = tdir / f"{frame_id}.jpg"
    if dest.exists():
        return dest
    if not src.exists():
        return None
    try:
        if HAVE_PIL:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB_SIZE, THUMB_SIZE))
                im.save(dest, "JPEG", quality=82)
        else:
            subprocess.run(
                ["ffmpeg", "-v", "quiet", "-y", "-i", str(src),
                 "-vf", f"scale='min({THUMB_SIZE},iw)':-2", str(dest)],
                check=True, capture_output=True,
            )
        return dest
    except Exception:
        return None


# -----------------------------
# HTML
# -----------------------------

UI_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LoRA Designer Studio</title>
<style>
:root{--bg:#0d1017;--panel:#161b26;--panel2:#1d2433;--text:#dde3ee;--dim:#8a93a6;
--accent:#7c5cff;--accent2:#00d4aa;--err:#ff5c7a;--warn:#ffb454;--border:#262e40}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Helvetica,Arial,sans-serif;padding:20px}
h1{font-size:20px;letter-spacing:.5px;display:inline-block}
h1 span{color:var(--accent)}
.sub{color:var(--dim);font-size:12px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:340px 1fr;gap:16px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:16px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin-bottom:12px}
label{display:block;font-size:11px;color:var(--dim);margin:8px 0 3px}
input,select,textarea{width:100%;background:var(--panel2);border:1px solid var(--border);
border-radius:6px;color:var(--text);padding:7px 9px;font-size:13px;font-family:inherit}
textarea{font-family:ui-monospace,Menlo,monospace;font-size:11px}
.row{display:flex;gap:8px}.row>*{flex:1}
.chk{display:flex;align-items:center;gap:6px;margin-top:8px;font-size:12px;color:var(--text)}
.chk input{width:auto}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:9px 14px;
font-size:13px;font-weight:600;cursor:pointer;margin-top:12px;width:100%}
button:hover{filter:brightness(1.15)}
button.alt{background:var(--accent2);color:#06281f}
button.warn{background:var(--warn);color:#3a2400}
button.stop{background:var(--err);width:auto;padding:6px 14px;margin:0}
button.mini{width:auto;padding:3px 10px;margin:0;font-size:11px}
#out{background:#0a0d13;border:1px solid var(--border);border-radius:8px;padding:14px;
white-space:pre-wrap;font:11.5px/1.5 ui-monospace,Menlo,monospace;height:360px;overflow-y:auto;color:#b9f0d8}
#log{background:#0a0d13;border:1px solid var(--border);border-radius:8px;padding:14px;
white-space:pre-wrap;font:10.5px/1.45 ui-monospace,Menlo,monospace;height:140px;overflow-y:auto;color:var(--dim)}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px}
.stat{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px;text-align:center}
.stat .v{font-size:22px;font-weight:700;color:var(--accent2)}
.stat .k{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600}
.badge.run{background:#173527;color:var(--accent2)}
.badge.idle{background:#252b3b;color:var(--dim)}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.tabs{display:flex;gap:6px;margin-bottom:12px}
.tab{padding:6px 12px;border-radius:6px;font-size:12px;cursor:pointer;color:var(--dim);background:var(--panel2)}
.tab.on{background:var(--accent);color:#fff}
.pane{display:none}.pane.on{display:block}
.page{display:none}.page.on{display:block}
body{padding-left:196px}
.nav{position:fixed;left:0;top:0;bottom:0;width:180px;background:var(--panel);
border-right:1px solid var(--border);padding:16px 10px;display:flex;
flex-direction:column;gap:4px;overflow-y:auto;z-index:10}
.nav .brand{font-size:15px;font-weight:700;margin-bottom:10px;padding:0 6px}
.nav .brand span{color:var(--accent)}
.nav .tab{font-size:13px;padding:9px 12px;border-radius:8px;background:transparent}
.nav .tab.on{background:var(--accent);color:#fff}
.nav .group{font-size:9px;color:var(--dim);text-transform:uppercase;
letter-spacing:1.5px;margin:12px 6px 2px}
.appframe{width:100%;height:calc(100vh - 130px);border:1px solid var(--border);
border-radius:10px;background:#0a0d13}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--border)}
th{color:var(--dim);font-size:10px;text-transform:uppercase;letter-spacing:1px}
.jq-done{color:var(--accent2)}.jq-error{color:var(--err)}.jq-running{color:var(--warn)}
.jq-queued{color:var(--dim)}.jq-stopped,.jq-cancelled{color:var(--dim);text-decoration:line-through}
#frames{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.fcard{background:var(--panel);border:2px solid var(--border);border-radius:8px;overflow:hidden;cursor:pointer;position:relative}
.fcard img{width:100%;height:120px;object-fit:cover;display:block}
.fcard .fi{padding:5px 7px;font-size:10px;color:var(--dim);line-height:1.4}
.fcard.s-selected{border-color:var(--accent2)}
.fcard.s-packaged{border-color:var(--accent)}
.fcard.s-rejected{border-color:var(--err);opacity:.55}
.fcard .tag{position:absolute;top:5px;left:5px;background:#000a;padding:1px 7px;border-radius:10px;font-size:9px}
.pager{display:flex;gap:8px;align-items:center;margin-top:12px;color:var(--dim);font-size:12px}
.filters{display:flex;gap:8px;align-items:flex-end;margin-bottom:12px}
.filters>div{min-width:150px}
.appdot{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--dim);
  padding:3px 10px;cursor:pointer;user-select:none}
.appdot:hover{color:var(--fg)}
.appdot .dot{width:8px;height:8px;border-radius:50%;background:#555;flex:none}
.appdot .dot.run{background:var(--ok,#4caf50)}
.appdot .dot.crash{background:var(--err,#f44336)}
.appdot .dotlabel{margin-left:auto;font-size:10px}
body.compact{font-size:12px}
body.compact .fcard{width:120px}
body.compact .filters>div{min-width:110px}
body.compact h2{margin:4px 0}
</style></head><body>
<div class="nav">
  <div class="brand">LoRA <span>Studio</span>
    <span style="float:right;cursor:pointer;font-size:10px;color:var(--dim)"
      title="toggle compact density" onclick="document.body.classList.toggle('compact')">&#9636;</span></div>
  <div class="group">Generate</div>
  <div class="tab pg on" data-g="play">Playground</div>
  <div class="tab pg" data-g="concept">Concept Control</div>
  <div class="tab pg" data-g="wardrobe">Wardrobe Variation</div>
  <div class="group">Build</div>
  <div class="tab pg" data-g="factory">Dataset Factory</div>
  <div class="tab pg" data-g="wizard">Create Wizard</div>
  <div class="tab pg" data-g="pipeline">Pipeline</div>
  <div class="group">Curate</div>
  <div class="tab pg" data-g="review">Review</div>
  <div class="tab pg" data-g="tags">Tags</div>
  <div class="tab pg" data-g="builds">Builds</div>
  <div class="group">Ship</div>
  <div class="tab pg" data-g="train">Train</div>
  <div class="tab pg" data-g="test">Test</div>
  <div class="tab pg" data-g="lab">Lab</div>
  <div class="tab pg" data-g="mergeforge">MergeForge</div>
  <div class="group">&nbsp;</div>
  <div class="tab pg" data-g="settings">Settings</div>
  <div class="group">Search</div>
  <div style="padding:2px 10px"><input id="gsearch" placeholder="search all..."
    style="width:100%;font-size:11px" onchange="gSearch()"></div>
  <div id="gresults" style="font-size:10px;padding:0 10px;color:var(--dim)"></div>
  <div class="group">Engines</div>
  <div class="appdot" id="dot_forge" onclick="appStart('forge')" title="click to start">
    <span class="dot idle"></span>Forge <span class="dotlabel">...</span></div>
  <div class="appdot" id="dot_play" onclick="appStart('playground')" title="click to start">
    <span class="dot idle"></span>Playground <span class="dotlabel">...</span></div>
  <div class="appdot" id="dot_factory" onclick="appStart('factory')" title="click to start">
    <span class="dot idle"></span>Factory <span class="dotlabel">...</span></div>
</div>
<div style="margin-bottom:6px"><h1>LoRA <span>Designer Studio</span></h1></div>
<div class="sub">__PNAME__ &middot; local &middot; non-destructive &middot; resumable &middot; manifest-tracked</div>

<!-- ============ PLAYGROUND (embedded) ============ -->
<div class="page on" id="page-play">
  <div class="topbar">
    <h2 style="margin:0">Grokkie Playground</h2>
    <div>
      <span id="play_badge" class="badge idle">checking...</span>
      <button class="mini" onclick="appStart('playground')">Start</button>
      <button class="mini stop" onclick="appStop('playground')">Stop</button>
    </div>
  </div>
  <div id="play_hint" class="sub">Unlimited LoRA stack, WeightEngine weighting,
  img2img, batch/vary, canvas. Runs in your Forge env as a managed child app;
  shipped presets from the Factory / mega-pipeline appear under its Presets tab.</div>
  <iframe id="play_frame" class="appframe" src="about:blank"></iframe>
</div>

<!-- ============ FACTORY (embedded) ============ -->
<div class="page" id="page-wardrobe">
<h2>Wardrobe Variation</h2>
<div class="sub">selective region editing &middot; identity-preserving inpainting &middot; garment replacement &middot; visual continuity</div>
<div class="filters">
  <div style="flex:1"><label>Character image path</label>
    <input id="wd_image" placeholder="/path/to/character.png"></div>
  <div><label>Mask image (optional)</label><input id="wd_mask" placeholder="manual mask path"></div>
</div>
<div class="filters">
  <div><label>Region</label><select id="wd_region" onchange="wdRegion()"></select></div>
  <div><label>Edit mode</label>
    <select id="wd_mode">
      <option value="garment_replacement">Garment replacement</option>
      <option value="garment_layering">Garment layering</option>
      <option value="style_variation">Style variation</option>
      <option value="full_wardrobe_variation">Full wardrobe variation</option>
      <option value="background_environment_variation">Background / environment variation</option>
    </select></div>
  <div><label>Denoise (0 = recommended)</label><input id="wd_denoise" value="0" style="width:70px"></div>
  <div><label>Seed</label><input id="wd_seed" value="42" style="width:80px"></div>
  <div><label>FaceID Guidance</label>
    <select id="wd_faceid">
      <option value="off">Off</option>
      <option value="balanced" selected>Balanced</option>
      <option value="strong">Strong</option>
      <option value="maximum">Maximum</option>
    </select></div>
  <div style="align-self:flex-end">
    <label style="font-size:11px"><input type="checkbox" id="wd_facelock"> <b>Strong Face Lock</b></label>
    <label style="font-size:11px"><input type="checkbox" id="wd_pose" checked> pose consistency</label>
    <label style="font-size:11px"><input type="checkbox" id="wd_body"> body structure lock</label>
    <label style="font-size:11px"><input type="checkbox" id="wd_sil"> silhouette guidance</label>
    <label style="font-size:11px"><input type="checkbox" id="wd_bg" checked> background consistency</label>
    <label style="font-size:11px"><input type="checkbox" id="wd_stack" checked> use Concept Control stack</label>
  </div>
</div>
<div id="wd_region_info" style="font-size:11px;color:var(--dim);margin-bottom:6px"></div>
<div class="filters">
  <div style="flex:1"><label>Garment / fashion direction</label>
    <input id="wd_prompt" placeholder="tailored charcoal blazer, satin blouse, editorial styling"></div>
  <div style="align-self:flex-end">
    <button class="mini" onclick="wdReadiness()">Check readiness</button>
    <button onclick="wdGenerate()">Generate edit</button></div>
</div>
<div id="wd_readiness" style="font-size:12px"></div>
<div id="wd_result" style="font-size:12px;margin-top:6px"></div>
</div>

<div class="page" id="page-concept">
<h2>Concept Control</h2>
<div class="sub">Visual LoRA Explorer &middot; attribute controls &middot; stack intelligence &middot; controlled variation</div>
<div style="border:1px solid #2a3142;border-radius:8px;padding:10px;margin-bottom:14px">
  <b>Character + Concept Stack</b>
  <span style="color:var(--dim);font-size:11px"> &mdash; guided flow: identity anchor &rarr; concept layers &rarr; resolve &rarr; Playground</span>
  <div id="cc_wf_status" style="font-size:11px;color:var(--dim);margin:4px 0">Loading library overview...</div>
  <div id="cc_starters" style="display:flex;flex-wrap:wrap;gap:8px"></div>
</div>
<div class="filters">
  <div><label>Search</label><input id="cc_search" oninput="loadConceptCards()"></div>
  <div><label>Concept family</label>
    <select id="cc_family" onchange="loadConceptCards()">
      <option value="">any</option><option>identity</option><option>character</option>
      <option>wardrobe</option><option>fashion</option><option>lighting</option>
      <option>pose</option><option>style</option><option>texture</option>
      <option>detail</option><option>environment</option><option>composition</option>
      <option>camera</option><option>refinement</option>
    </select></div>
  <div><label>Influence tag</label>
    <select id="cc_tag" onchange="loadConceptCards()">
      <option value="">any</option><option>identity_anchor</option><option>silhouette</option>
      <option>garment_style</option><option>fabric_texture</option><option>lighting_mood</option>
      <option>pose_energy</option><option>facial_consistency</option><option>anatomy_balance</option>
      <option>color_palette</option><option>scene_context</option><option>detail_density</option>
      <option>composition_flow</option><option>camera_perspective</option>
    </select></div>
  <div><label>Sort</label>
    <select id="cc_sort" onchange="loadConceptCards()">
      <option value="name">name</option><option value="modified">modified</option>
      <option value="family">family</option><option value="identity_risk">identity risk</option>
    </select></div>
  <div><label>Preview</label>
    <select id="cc_preview" onchange="loadConceptCards()">
      <option value="">any</option><option value="has">has preview</option>
      <option value="missing">missing preview</option>
    </select></div>
  <div style="align-self:flex-end"><button class="mini" onclick="loadConceptCards(true)">Rescan folders</button></div>
</div>
<div id="cc_cards" style="display:flex;flex-wrap:wrap;gap:8px"></div>
<h3 style="margin-top:18px">Attribute Controls</h3>
<div id="cc_sliders" style="max-width:640px"></div>
<h3 style="margin-top:14px">Resolved Stack</h3>
<div style="display:flex;gap:8px;margin-bottom:8px">
  <button onclick="ccResolve()">Resolve stack</button>
  <button class="mini" onclick="ccSend()">Send to Playground</button>
  <button class="mini" onclick="ccSaveWorkflowPreset()">Save stack preset</button>
  <button class="mini" onclick="ccLoadPresets()">Saved presets</button>
  <button class="mini" onclick="ccDiffPresets()">Diff presets</button>
  <button class="mini" onclick="ccHistory()">Stack history</button>
  <button class="mini" id="cc_balance_btn" style="display:none"
    onclick="ccApplyBalance()">Apply Suggested Balance</button>
  <button class="mini" id="cc_clearpins_btn" style="display:none"
    onclick="ccOverrides={};ccResolve()">Clear manual weights</button>
</div>
<div id="cc_stack" style="font-size:12px"></div>
<div id="cc_warnings" style="font-size:12px;margin-top:6px"></div>
<div id="cc_presets" style="font-size:12px;margin-top:6px"></div>
<h3 style="margin-top:18px">Batch Variations</h3>
<div class="filters">
  <div><label>Variation mode</label>
    <select id="cc_bv_mode" onchange="ccPreviewGrid()">
      <option value="low_risk">Low-Risk Studio Sweep (cap 24)</option>
      <option value="balanced">Balanced Exploration (cap 64)</option>
      <option value="creative">Creative Exploration (cap 128)</option>
    </select></div>
  <div><label>Axis 1</label><select id="cc_bv_slider" onchange="ccPreviewGrid()"></select></div>
  <div><label>Values (csv or min:max:step)</label>
    <input id="cc_bv_values" value="0.2,0.4,0.6" oninput="ccPreviewGrid()"></div>
  <div><label>Axis 2 (optional)</label><select id="cc_bv_slider2" onchange="ccPreviewGrid()"></select></div>
  <div><label>Axis 2 values</label><input id="cc_bv_values2" oninput="ccPreviewGrid()"></div>
  <div><label>Seeds (csv)</label><input id="cc_bv_seeds" value="42" oninput="ccPreviewGrid()"></div>
</div>
<div class="filters">
  <div style="flex:1"><label>Composition goal / style descriptors</label>
    <input id="cc_bv_tail" placeholder="full body framing, soft studio lighting"></div>
  <div style="align-self:flex-end"><span id="cc_bv_estimate" style="font-size:11px;color:var(--dim);margin-right:8px"></span>
    <button class="mini" onclick="ccBatch()">Create Variation Grid</button></div>
</div>
<div id="cc_batch" style="font-size:12px"></div>
<h3 style="margin-top:18px">Variation Review Grid</h3>
<div class="filters">
  <div><label>Batch</label><select id="cc_rg_batch"></select></div>
  <div style="align-self:flex-end">
    <button class="mini" onclick="rgLoadBatches()">Refresh batches</button>
    <button class="mini" onclick="rgShow()">Open review grid</button></div>
</div>
<div id="cc_review" style="display:flex;flex-wrap:wrap;gap:8px;font-size:11px"></div>
</div>

<div class="page" id="page-factory">
  <div class="topbar">
    <h2 style="margin:0">Grokkie Dataset Factory</h2>
    <div>
      <span id="factory_badge" class="badge idle">checking...</span>
      <button class="mini" onclick="appStart('factory')">Start</button>
      <button class="mini stop" onclick="appStop('factory')">Stop</button>
    </div>
  </div>
  <div class="sub" id="factory_hint">Guided 7-step flow: ingest &rarr; analyze &rarr;
  stack plan &rarr; curate &rarr; build &rarr; train + best-epoch sweep &rarr;
  merge &amp; ship. Shares this project's manifest live.</div>
  <iframe id="factory_frame" class="appframe" src="about:blank"></iframe>
</div>

<!-- ============ CREATOR WIZARD ============ -->
<div class="page" id="page-wizard">
<div class="grid">
<div>
  <div class="card">
    <h2>Step 1 &middot; Analyze Dataset</h2>
    <div class="sub">Reads the manifest (no re-scanning). Curate + caption first
    for best detection.</div>
    <button onclick="wzAnalyze()">Analyze</button>
    <pre id="wz_stats" style="font-size:11px;color:var(--dim);white-space:pre-wrap;margin-top:8px"></pre>
  </div>
  <div class="card">
    <h2>Step 2 &middot; Type &amp; Identity</h2>
    <div id="wz_types"></div>
    <div class="row">
      <div><label>Trigger word</label><input id="wz_trigger" value="__TOKEN__"></div>
      <div><label>Class word</label><input id="wz_cls" value="__CLS__"></div>
    </div>
    <label>Output name (blank = auto)</label>
    <input id="wz_name" value="">
  </div>
  <div class="card">
    <h2>Step 3 &middot; Create</h2>
    <div class="chk"><input type="checkbox" id="wz_train" checked><span>Train after building (preset auto by type)</span></div>
    <div class="chk"><input type="checkbox" id="wz_matrix" checked><span>Eval matrix + model card after training</span></div>
    <div class="row">
      <div><label>Eval backend</label><select id="wz_backend"><option>forge</option><option>diffusers</option></select></div>
    </div>
    <button class="warn" onclick="runStage('wizard')">Create LoRA</button>
    <div class="sub" style="margin-top:8px">Queues build &rarr; train &rarr; matrix &rarr;
    model card as one job. Watch progress in Pipeline tab's Live Output.</div>
  </div>
</div>
<div class="card">
  <h2>How the wizard decides</h2>
  <div class="sub" style="line-height:1.8">
  <b>character</b> — high face presence, tight identity consistency, balanced framings<br>
  <b>style</b> — low face dependence, high scene/cluster diversity<br>
  <b>outfit</b> — body framings, concentrated clusters (set include_clusters after review)<br>
  <b>pose</b> — full-body dominance<br>
  <b>detail</b> — closeup dominance + high sharpness (smart-crop enabled)<br>
  <b>explicit</b> — explicit caption tags present; trains with character parameters<br><br>
  Each type maps to a tuned recipe (quota, repeats, filters) + an M2-Max training
  preset. You can always override the recommendation by clicking another card.
  </div>
</div>
</div>
</div>

<!-- ============ PIPELINE PAGE ============ -->
<div class="page" id="page-pipeline">
<div class="stats" id="stats"></div>
<div class="grid">
<div>
  <div class="tabs">
    <div class="tab on" data-p="extract">Extract</div>
    <div class="tab" data-p="curate">Curate</div>
    <div class="tab" data-p="caption">Caption</div>
    <div class="tab" data-p="package">Package</div>
  </div>

  <div class="card pane on" id="pane-extract">
    <h2>1 &middot; Frame Extraction</h2>
    <label>Video directories (one per line)</label>
    <textarea id="vdirs" rows="6">__VDIRS__</textarea>
    <label>Photos directory</label>
    <input id="pdir" value="__PDIR__">
    <label>Output base</label>
    <input id="outbase" value="__OUTBASE__">
    <div class="row">
      <div><label>FPS</label><input id="fps" type="number" step="0.05" value="0.25"></div>
      <div><label>JPEG q</label><input id="jq" type="number" value="2" min="1" max="10"></div>
      <div><label>Segment s</label><input id="seg" type="number" value="300" min="30"></div>
    </div>
    <div class="row">
      <div><label>Limit (0=all)</label><input id="limitv" type="number" value="0"></div>
      <div><label>Photo mode</label><select id="pmode"><option>hardlink</option><option>copy</option><option>symlink</option></select></div>
    </div>
    <div class="chk"><input type="checkbox" id="dry" checked><span>Dry run</span></div>
    <div class="chk"><input type="checkbox" id="resume" checked><span>Resume / skip completed</span></div>
    <div class="chk"><input type="checkbox" id="photos" checked><span>Import photos</span></div>
    <div class="chk"><input type="checkbox" id="overwrite"><span>Overwrite outputs</span></div>
    <button onclick="runStage('extract')">Queue Extraction</button>
  </div>

  <div class="card pane" id="pane-curate">
    <h2>2 &middot; Curation</h2>
    <div class="row">
      <div><label>Dedup distance</label><input id="ham" type="number" value="4" min="0" max="16"></div>
      <div><label>Workers</label><input id="workers" type="number" value="8" min="1" max="16"></div>
    </div>
    <div class="row">
      <div><label>Min sharp</label><input id="minsharp" type="number" value="35"></div>
      <div><label>Min bright</label><input id="minb" type="number" value="18"></div>
      <div><label>Max bright</label><input id="maxb" type="number" value="242"></div>
    </div>
    <div class="chk"><input type="checkbox" id="rescore"><span>Re-score everything</span></div>
    <hr style="border-color:var(--border);margin:14px 0">
    <h2>2b &middot; Identity ([ai] extras)</h2>
    <div class="chk"><input type="checkbox" id="smart"><span>Face + identity filtering</span></div>
    <label>Anchor dir (subject references)</label>
    <input id="anchordir" value="__ANCHOR__">
    <div class="row">
      <div><label>Id threshold</label><input id="idthr" type="number" step="0.01" value="0.35"></div>
      <div><label>Min face area</label><input id="minface" type="number" step="0.005" value="0.015"></div>
    </div>
    <div class="chk"><input type="checkbox" id="buildanchor"><span>(Re)build anchor first</span></div>
    <hr style="border-color:var(--border);margin:14px 0">
    <h2>2c &middot; Diversity ([cluster] extras)</h2>
    <div class="chk"><input type="checkbox" id="cluster"><span>CLIP clustering (scene/outfit diversity)</span></div>
    <div class="row">
      <div><label>Clusters (0=auto)</label><input id="kclusters" type="number" value="0"></div>
      <div><label>Batch size</label><input id="cbatch" type="number" value="16"></div>
    </div>
    <button class="alt" onclick="runStage('curate')">Queue Curation</button>
  </div>

  <div class="card pane" id="pane-caption">
    <h2>3 &middot; Auto-Captioning (WD14, [caption] extras)</h2>
    <div class="row">
      <div><label>Trigger token</label><input id="cap_trigger" value="__TOKEN__"></div>
      <div><label>Class word</label><input id="cap_cls" value="__CLS__"></div>
    </div>
    <div class="row">
      <div><label>Tag threshold</label><input id="cap_thr" type="number" step="0.05" value="0.35"></div>
      <div><label>Max tags</label><input id="cap_max" type="number" value="30"></div>
    </div>
    <label>Blacklist (comma-separated tags to drop)</label>
    <input id="cap_black" value="">
    <label>Remap (old:new,old2:new2)</label>
    <input id="cap_remap" value="">
    <label>Prune (permanent traits to absorb into trigger)</label>
    <input id="cap_prune" value="">
    <div class="chk"><input type="checkbox" id="cap_pony"><span>Pony quality prefix (score_9, ...)</span></div>
    <div class="chk"><input type="checkbox" id="cap_force"><span>Re-caption (keeps manual edits)</span></div>
    <button class="alt" onclick="runStage('caption')">Queue Captioning</button>
  </div>

  <div class="card pane" id="pane-package">
    <h2>3 &middot; Dataset Packaging</h2>
    <div class="row">
      <div><label>Trigger token</label><input id="token" value="__TOKEN__"></div>
      <div><label>Class word</label><input id="cls" value="__CLS__"></div>
    </div>
    <div class="row">
      <div><label>Repeats</label><input id="repeats" type="number" value="10"></div>
      <div><label>Max/video</label><input id="maxpv" type="number" value="40"></div>
      <div><label>Max total</label><input id="maxtot" type="number" value="0"></div>
    </div>
    <label>Static caption fallback (blank = "token class")</label>
    <input id="caption" value="">
    <label>Framing quotas (blank = off; needs Max total)</label>
    <input id="quota" value="__QUOTA__">
    <div class="chk"><input type="checkbox" id="captions" checked><span>Write .txt captions (WD14 per-frame when available)</span></div>
    <button class="alt" onclick="runStage('package')">Queue Packaging</button>
  </div>

  <div class="card">
    <h2>Full Pipeline</h2>
    <div style="font-size:11px;color:var(--dim)">extract &rarr; curate__SMARTLBL__ &rarr; caption &rarr; package, queued as separate jobs with current settings. Stages missing optional extras skip themselves gracefully.</div>
    <button class="warn" onclick="runStage('pipeline')">Queue Full Pipeline</button>
    <hr style="border-color:var(--border);margin:14px 0">
    <h2>Full Studio (one click)</h2>
    <div style="font-size:11px;color:var(--dim)">raw media &rarr; curate &rarr; caption
    &rarr; auto-typed build &rarr; train &rarr; <b>best-epoch sweep</b> &rarr;
    Playground preset. One job, fully streamed.</div>
    <button class="warn" onclick="runStage('mega')">Run FULL STUDIO Pipeline</button>
  </div>
</div>

<div>
  <div class="card">
    <div class="topbar">
      <h2 style="margin:0">Live Output</h2>
      <div><span id="badge" class="badge idle">idle</span>
      <button class="stop" onclick="fetch('/api/stop',{method:'POST'})">Stop</button></div>
    </div>
    <div id="out">Ready. Queue a stage on the left.\nTip: always dry run extraction first.</div>
  </div>
  <div class="card">
    <h2>Job Queue</h2>
    <table><thead><tr><th>#</th><th>Stage</th><th>Status</th><th>Started</th><th></th></tr></thead>
    <tbody id="jq"></tbody></table>
  </div>
  <div class="card">
    <h2>Recent Log</h2>
    <div id="log"></div>
  </div>
</div>
</div>
</div>

<!-- ============ REVIEW PAGE ============ -->
<div class="page" id="page-review">
<div class="filters">
  <div><label>Status</label>
    <select id="fstatus" onchange="pg=0;loadFrames()">
      <option value="selected">selected</option>
      <option value="packaged">packaged</option>
      <option value="duplicate">duplicate</option>
      <option value="rejected_blur">rejected_blur</option>
      <option value="rejected_exposure">rejected_exposure</option>
      <option value="rejected_noface">rejected_noface</option>
      <option value="rejected_smallface">rejected_smallface</option>
      <option value="rejected_identity">rejected_identity</option>
      <option value="rejected_manual">rejected_manual</option>
      <option value="scored">scored</option>
      <option value="new">new</option>
    </select></div>
  <div><label>Framing</label>
    <select id="fframing" onchange="pg=0;loadFrames()">
      <option value="">any</option><option>closeup</option><option>portrait</option>
      <option>upper_body</option><option>full_body</option><option>none</option>
    </select></div>
  <div><label>Study</label>
    <select id="fstudy" onchange="pg=0;loadFrames()">
      <option value="">any</option>
      <option value="figure_study_candidate">Figure Study Candidates</option>
      <option value="fashion_study_candidate">Fashion Study Candidates</option>
      <option value="lingerie_fashion_candidate">Lingerie/Fashion Study</option>
      <option value="form_proportion_candidate">Form &amp; Proportion</option>
      <option value="needs_review">Needs Study Review</option>
      <option value="export_eligible">Export Eligible</option>
      <option value="identity_strong">Strong Identity Lock</option>
      <option value="pose_strong">Strong Pose Clarity</option>
    </select></div>
  <div><label>Caption contains</label>
    <input id="fcaption" placeholder="e.g. 1boy" style="width:110px"
           onchange="pg=0;loadFrames()"></div>
  <div><label>Sort</label>
    <select id="fsort" onchange="pg=0;loadFrames()">
      <option value="">path</option>
      <option value="study_confidence">study confidence</option>
      <option value="identity_lock">identity lock</option>
      <option value="figure_study">figure-study score</option>
      <option value="fashion_study">fashion-study score</option>
    </select></div>
  <div style="flex:1;color:var(--dim);font-size:11px;align-self:center">
    Click a card to toggle keep (green) / reject (red). Verdicts are manual overrides, stored in the manifest.</div>
  <div style="display:flex;gap:6px;align-items:flex-end">
    <button class="mini" onclick="bulkVerdict('keep')">Keep all shown</button>
    <button class="mini stop" onclick="bulkVerdict('reject')">Reject all shown</button>
  </div>
</div>
<div id="frames"></div>
<div class="pager">
  <button class="mini" onclick="if(pg>0){pg--;loadFrames()}">&larr; prev</button>
  <span id="pinfo"></span>
  <button class="mini" onclick="pg++;loadFrames()">next &rarr;</button>
</div>
</div>

<!-- ============ TAGS PAGE ============ -->
<div class="page" id="page-tags">
<div class="card" style="max-width:760px">
  <div class="topbar"><h2 style="margin:0">Tag Frequency (captioned frames)</h2>
  <button class="mini" onclick="loadTags()">Refresh</button></div>
  <div class="sub">Tags appearing in &gt;80% of frames describe permanent traits -
  add them to the Caption stage <b>prune</b> list so they bind to your trigger word.</div>
  <table><thead><tr><th>Tag</th><th>Count</th><th>%</th><th></th></tr></thead>
  <tbody id="tagrows"></tbody></table>
</div>
</div>

<!-- ============ BUILDS PAGE ============ -->
<div class="page" id="page-builds">
<div class="grid">
<div>
  <div class="card">
    <h2>New Build (recipe)</h2>
    <label>Recipe</label>
    <select id="b_recipe"></select>
    <label>Note (optional)</label>
    <input id="b_note" value="">
    <button class="warn" onclick="runBuild()">Queue Versioned Build</button>
    <div class="sub" style="margin-top:10px">Recipes are defined in the project file
    ([recipes.NAME]). Identical content produces an identical hash and is not rebuilt.</div>
  </div>
  <div class="card">
    <h2>Diff Versions</h2>
    <div class="row">
      <div><label>From (A)</label><input id="d_a" type="number" value="1"></div>
      <div><label>To (B)</label><input id="d_b" type="number" value="2"></div>
    </div>
    <button class="mini" style="margin-top:10px" onclick="runDiff()">Diff</button>
    <pre id="diffout" style="font-size:11px;color:var(--dim);white-space:pre-wrap;margin-top:10px"></pre>
  </div>
</div>
<div class="card">
  <div class="topbar"><h2 style="margin:0">Dataset Versions</h2>
  <button class="mini" onclick="loadBuilds()">Refresh</button></div>
  <table><thead><tr><th>v</th><th>Recipe</th><th>Train</th><th>Val</th><th>Hash</th><th>Built</th><th>Note</th></tr></thead>
  <tbody id="buildrows"></tbody></table>
</div>
</div>
</div>

<!-- ============ TRAIN PAGE ============ -->
<div class="page" id="page-train">
<div class="grid">
<div>
  <div class="card">
    <h2>New Training Run</h2>
    <label>Dataset version</label>
    <select id="t_dataset"></select>
    <label>Preset</label>
    <select id="t_preset">
      <option>character</option><option>style</option><option>outfit</option>
      <option>pose</option><option>detail</option>
    </select>
    <button class="mini" onclick="suggestPreset()">Suggest Training Preset</button>
    <div id="t_suggest" class="sub" style="margin-top:6px"></div>
    <label>Run name (blank = auto)</label>
    <input id="t_name" value="">
    <div class="chk"><input type="checkbox" id="t_dry" checked><span>Dry run (print command only)</span></div>
    <button class="warn" onclick="runTrain()">Queue Training Run</button>
    <div class="sub" style="margin-top:10px">Requires sd_scripts_dir + base_model
    in Settings. Checkpoints save every epoch; Stop keeps the last epoch.</div>
  </div>
  <div class="card">
    <h2>Best-Epoch Sweep</h2>
    <label>Run name</label>
    <input id="sw_run" value="">
    <div class="row">
      <div><label>Backend</label><select id="sw_backend"><option>forge</option><option>diffusers</option></select></div>
      <div><label>Max gap</label><input id="sw_gap" type="number" step="0.01" value="0.15"></div>
    </div>
    <button class="warn" onclick="runStage('sweep')">Queue Sweep</button>
    <button class="mini" style="margin-top:8px" onclick="loadSweep()">Show stored result</button>
    <pre id="sw_out" style="font-size:10.5px;color:var(--dim);white-space:pre-wrap;margin-top:8px"></pre>
  </div>
  <div class="card">
    <h2>Loss Curve</h2>
    <label>Run</label>
    <select id="m_run" onchange="loadCurve()"></select>
    <canvas id="curve" width="300" height="140" style="width:100%;background:#0a0d13;border:1px solid var(--border);border-radius:8px;margin-top:8px"></canvas>
    <div id="curveinfo" class="sub" style="margin-top:6px"></div>
  </div>
</div>
<div class="card">
  <div class="topbar"><h2 style="margin:0">Training Runs</h2>
  <button class="mini" onclick="loadRuns()">Refresh</button></div>
  <table><thead><tr><th>#</th><th>Name</th><th>Dataset</th><th>Preset</th><th>Status</th><th>Progress</th><th>Loss</th></tr></thead>
  <tbody id="runrows"></tbody></table>
</div>
</div>
</div>

<!-- ============ TEST PAGE (Phase 6) ============ -->
<div class="page" id="page-test">
<div class="grid">
<div>
  <div class="card">
    <div class="topbar"><h2 style="margin:0">Forge Backend</h2>
      <div><span id="forge_badge" class="badge idle">checking...</span>
      <button class="mini" onclick="appStart('forge')">Start</button>
      <button class="mini stop" onclick="appStop('forge')">Stop</button></div>
    </div>
    <div class="sub" id="forge_hint">reForge with --api on :7860 - required for the
    'forge' backend in Test, Matrix and Sweep. Auto-detected if already running.</div>
  </div>
  <div class="card">
    <h2>Test Generation</h2>
    <div class="row">
      <div><label>Backend</label><select id="g_backend"><option>forge</option><option>diffusers</option></select></div>
      <div><label>Forge URL</label><input id="g_url" value="http://127.0.0.1:7860"></div>
    </div>
    <label>LoRA stack (path:weight per line, negative weights ok)</label>
    <textarea id="g_loras" rows="3"></textarea>
    <label>Prompt</label>
    <textarea id="g_prompt" rows="3">score_9, score_8_up, __TOKEN__ __CLS__, studio portrait, photorealistic</textarea>
    <label>Negative</label>
    <input id="g_neg" value="lowres, bad anatomy, blurry, watermark, worst quality">
    <div class="row">
      <div><label>Steps</label><input id="g_steps" type="number" value="28"></div>
      <div><label>CFG</label><input id="g_cfg" type="number" step="0.5" value="6"></div>
      <div><label>Seed</label><input id="g_seed" type="number" value="42"></div>
    </div>
    <div class="row">
      <div><label>Width</label><input id="g_w" type="number" step="64" value="1024"></div>
      <div><label>Height</label><input id="g_h" type="number" step="64" value="1024"></div>
    </div>
    <label>Init image path (blank = txt2img)</label>
    <div class="row">
      <div style="flex:3"><input id="g_init" value=""></div>
      <div><label style="margin:0 0 3px">strength</label><input id="g_strength" type="number" step="0.05" value="0.6"></div>
    </div>
    <button onclick="runStage('testgen')">Queue Generation</button>
  </div>
  <div class="card">
    <h2>Test Matrix (one-click eval)</h2>
    <label>LoRA (path, or model stem for forge)</label>
    <input id="m_lora" value="">
    <div class="row">
      <div><label>Weight</label><input id="m_weight" type="number" step="0.05" value="0.85"></div>
      <div><label>Steps</label><input id="m_steps" type="number" value="28"></div>
    </div>
    <div class="chk"><input type="checkbox" id="mc_likeness" checked><span>likeness</span></div>
    <div class="chk"><input type="checkbox" id="mc_flexibility" checked><span>flexibility (overfit probe)</span></div>
    <div class="chk"><input type="checkbox" id="mc_pose" checked><span>pose</span></div>
    <div class="chk"><input type="checkbox" id="mc_outfit" checked><span>outfit</span></div>
    <div class="chk"><input type="checkbox" id="mc_style"><span>style</span></div>
    <button class="warn" onclick="runStage('matrix')">Queue Test Matrix</button>
    <div class="sub" style="margin-top:8px">Scores likeness vs your identity anchor;
    grids + averages land in evals/. Run per epoch checkpoint to find the best epoch.</div>
  </div>
</div>
<div class="card">
  <div class="topbar"><h2 style="margin:0">Eval Results</h2>
  <button class="mini" onclick="loadEvals()">Refresh</button></div>
  <table><thead><tr><th>LoRA</th><th>Category</th><th>n</th><th>Avg likeness</th></tr></thead>
  <tbody id="evalrows"></tbody></table>
  <div id="evalimgs" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin-top:12px"></div>
</div>
</div>
</div>

<!-- ============ LAB PAGE (Phase 7) ============ -->
<div class="page" id="page-lab">
<div class="grid">
<div>
  <div class="card">
    <h2>Merging Lab</h2>
    <label>LoRAs (path:weight per line, negative ok, mixed ranks ok)</label>
    <textarea id="mg_loras" rows="4"></textarea>
    <label>Block multipliers (te / down / mid / up)</label>
    <div class="row">
      <div><input id="mg_te" type="number" step="0.1" value="1"></div>
      <div><input id="mg_down" type="number" step="0.1" value="1"></div>
      <div><input id="mg_mid" type="number" step="0.1" value="1"></div>
      <div><input id="mg_up" type="number" step="0.1" value="1"></div>
    </div>
    <label>Output name</label>
    <input id="mg_name" value="combo_v1">
    <div class="chk"><input type="checkbox" id="mg_preview"><span>Preview: likeness matrix on result (uses Test tab backend)</span></div>
    <button class="warn" onclick="runStage('merge')">Queue Merge</button>
    <div class="sub" style="margin-top:8px">Concat method: result is a normal
    LoRA. Tip: style at up=1, down=0.3, te=0 layers style onto identity.</div>
  </div>
  <div class="card">
    <h2>A/B Blind Compare</h2>
    <div class="row">
      <div><label>A (eval label)</label><input id="ab_a" value=""></div>
      <div><label>B (eval label)</label><input id="ab_b" value=""></div>
    </div>
    <button class="mini" style="margin-top:10px" onclick="loadAB()">Load pairs</button>
    <div id="ab_tally" class="sub" style="margin-top:8px"></div>
  </div>
</div>
<div class="card">
  <h2>Comparison (labels hidden until you pick)</h2>
  <div id="ab_pairs" style="display:grid;grid-template-columns:1fr;gap:14px"></div>
</div>
</div>
</div>

<!-- ============ MERGEFORGE PAGE ============ -->
<div class="page" id="page-mergeforge">
<div class="grid">
<div>
  <div class="card">
    <h2>Library Scan</h2>
    <div class="sub" id="mf_scan_sum">scanning...</div>
    <div id="mf_library" style="max-height:340px;overflow:auto;margin-top:8px"></div>
    <button class="mini" onclick="mfScan()">Rescan</button>
  </div>
  <div class="card">
    <h2>Guided Merge Wizard</h2>
    <div class="sub">1. tick LoRAs above &middot; 2. Plan &middot; 3. adjust weights &middot; 4. Forge</div>
    <button class="mini" onclick="mfPlan()">Plan selected</button>
    <div id="mf_plan" style="margin-top:8px"></div>
    <label>Output name</label>
    <input id="mf_name" value="mergeforge_v1">
    <label>Target rank (0 = auto: max input rank)</label>
    <input id="mf_rank" type="number" value="0" min="0" max="256">
    <button class="warn" id="mf_go" style="display:none" onclick="mfMerge()">Forge Merge</button>
    <div class="sub" id="mf_msg" style="margin-top:6px"></div>
  </div>
</div>
<div>
  <div class="card">
    <h2>Smart Recommendations</h2>
    <div id="mf_recs" class="sub">open this tab to load...</div>
  </div>
  <div class="card">
    <h2>Recipe History</h2>
    <div id="mf_recipes" class="sub">none yet</div>
  </div>
</div>
</div>
</div>

<!-- ============ SETTINGS PAGE ============ -->
<div class="page" id="page-settings">
<div class="grid">
<div class="card" style="grid-column:1/3;max-width:760px">
  <h2>Project Configuration</h2>
  <div class="sub" id="prjpath"></div>
  <div class="sub" id="runtimeinfo"></div>
  <label>Project name</label><input id="s_name">
  <label>Video directories (one per line)</label><textarea id="s_vdirs" rows="7"></textarea>
  <label>Photos directory</label><input id="s_pdir">
  <label>Output base</label><input id="s_out">
  <div class="row">
    <div><label>Trigger token</label><input id="s_token"></div>
    <div><label>Class word</label><input id="s_cls"></div>
    <div><label>UI port</label><input id="s_port" type="number"></div>
  </div>
  <label>Anchor dir (identity references)</label><input id="s_anchor">
  <label>Forge root (model reuse)</label><input id="s_forge">
  <label>sd-scripts dir (kohya, for training)</label><input id="s_sd">
  <label>Base model (.safetensors, Pony V6 XL)</label><input id="s_base">
  <label>LoRA output dir (blank = output_base/LORA_OUTPUT)</label><input id="s_lout">
  <label>Save to file</label><input id="s_path">
  <button onclick="saveProject()">Save Project File</button>
  <div id="s_msg" style="font-size:12px;color:var(--accent2);margin-top:8px"></div>
</div>
</div>
</div>

<script>
let pg=0;
document.querySelectorAll('.tab.pg').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab.pg').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('on'));
  t.classList.add('on');
  document.getElementById('page-'+t.dataset.g).classList.add('on');
  if(t.dataset.g==='review')loadFrames();
  if(t.dataset.g==='settings')loadProject();
  if(t.dataset.g==='tags')loadTags();
  if(t.dataset.g==='builds'){loadBuilds();loadRecipes();}
  if(t.dataset.g==='train'){loadRuns();loadTrainDatasets();}
  if(t.dataset.g==='test'){loadEvals();appPoll();}
  if(t.dataset.g==='play'||t.dataset.g==='factory')appPoll();
  if(t.dataset.g==='concept'){loadConceptSliders();loadConceptCards();loadStarters();rgLoadBatches();}
  if(t.dataset.g==='wardrobe')loadWardrobe();
  if(t.dataset.g==='mergeforge'){mfScan();mfLoadRecipes();}
});

// ============ Concept Control ============
let ccSel={};        // lora_id -> weight (selected stack)
let ccOverrides={};  // lora_id -> pinned manual weight (Apply Balance / numeric edit)
let ccLastStack=null;
let ccCards=[];
async function loadConceptCards(rescan){
  const q=`family=${v('cc_family')||''}&tag=${v('cc_tag')||''}&search=${encodeURIComponent(v('cc_search')||'')}`+
    `&sort=${v('cc_sort')||'name'}&preview=${v('cc_preview')||''}${rescan?'&rescan=1':''}`+
    `&output=${encodeURIComponent(v('outbase')||'')}`;
  const r=await(await fetch('/api/concept/cards?'+q)).json();
  ccCards=r.items||r.cards||[];
  document.getElementById('cc_cards').innerHTML=ccCards.map(c=>{
    const sel=ccSel[c.lora_id]!=null;
    const risk=c.profile.identity_risk;
    const rcol=risk==='high'?'var(--err)':risk==='medium'?'var(--warn)':'var(--ok,#4caf50)';
    const purl=(c.preview_urls||{}).default;
    const img=purl?`<img id="ccimg_${c.lora_id}" src="${purl}" style="width:100%;height:90px;object-fit:cover;border-radius:4px">`:
      `<div style="height:90px;background:#11161f;border-radius:4px;display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--dim)">`+
      `<div>${c.profile.family}</div><div style="font-size:9px">no preview</div></div>`;
    const levels=(c.preview_levels_available||[]).length>1?
      `<div style="font-size:9px;margin-top:2px">`+(c.preview_levels_available||[]).map(lv=>
        `<a style="cursor:pointer;color:var(--acc,#4af);margin-right:5px" `+
        `onclick="event.stopPropagation();document.getElementById('ccimg_${c.lora_id}').src='${c.preview_urls[lv]}'">${lv}</a>`).join('')+`</div>`:'';
    return `<div style="width:170px;border:1px solid ${sel?'var(--acc,#4af)':'#2a3142'};border-radius:6px;padding:6px;cursor:pointer" `+
      `onclick="ccToggle('${c.lora_id}')" title="${(c.description||c.profile.notes||'').replace(/"/g,'&quot;')}">${img}${levels}`+
      `<div style="font-size:11px;margin-top:4px;word-break:break-all"><b>${c.display_name||c.lora_id}</b></div>`+
      `<div style="font-size:10px;color:var(--dim)">${c.profile.family} &middot; <span style="color:${rcol}">${risk} risk</span>`+
      ` &middot; w ${c.profile.weight_default}</div>`+
      `<div style="font-size:9px;color:var(--dim)">${(c.profile.influence_tags||[]).join(' ')}</div>`+
      ((c.concept_meta&&((c.concept_meta.control_axes||[]).length||(c.concept_meta.known_conflicts||[]).length||c.concept_meta.priority_hint!=='normal'))?
        `<div style="font-size:9px;color:var(--dim)">`+
        ((c.concept_meta.control_axes||[]).length?`${c.concept_meta.control_axes.length} control axe(s) `:'')+
        (c.concept_meta.priority_hint!=='normal'?`&middot; ${c.concept_meta.priority_hint} `:'')+
        ((c.concept_meta.known_conflicts||[]).length?`&middot; ${c.concept_meta.known_conflicts.length} known conflict(s)`:'')+`</div>`:'')+
      `<button class="mini" style="margin-top:4px;width:49%" `+
      `onclick="event.stopPropagation();ccAddToStack('${c.lora_id}')">${sel?'In stack &#10003;':'Add'}</button>`+
      `<button class="mini" style="margin-top:4px;width:49%;float:right" title="edit sidecar metadata" `+
      `onclick="event.stopPropagation();ccEditSidecar('${c.lora_id}','${c.profile.family}','${c.profile.identity_risk}',${c.profile.weight_default})">Edit</button></div>`;
  }).join('')||'<span style="color:var(--dim)">No LoRAs found - check lora_output_dir / forge_root, then Rescan.</span>';
}
async function ccAddToStack(id){
  if(ccSel[id]==null){const c=ccCards.find(x=>x.lora_id===id);ccSel[id]=c?c.profile.weight_default:0.3;}
  await fetch('/api/playground/stack/add-lora',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({lora_id:id})});
  loadConceptCards();ccResolve();
}
function ccToggle(id){
  if(ccSel[id]!=null)delete ccSel[id];
  else{const c=ccCards.find(x=>x.lora_id===id);ccSel[id]=c?c.profile.weight_default:0.3;}
  loadConceptCards();ccResolve();
}
async function loadConceptSliders(){
  const r=await(await fetch('/api/concept/sliders')).json();
  try{
    const ax=await(await fetch('/api/concept/variation/axes')).json();
    const opts=ax.axes.map(a=>`<option value="${a.slider}">${a.label} (${a.identity_impact})</option>`).join('');
    document.getElementById('cc_bv_slider').innerHTML=opts;
    document.getElementById('cc_bv_slider2').innerHTML='<option value="">none</option>'+opts;
  }catch(e){}
  document.getElementById('cc_sliders').innerHTML=r.sliders.map(s=>
    `<div style="display:flex;gap:8px;align-items:center;margin:3px 0" title="${s.explanation}${s.identity_note?' | '+s.identity_note:''}">`+
    `<span style="width:200px;font-size:12px">${s.label}</span>`+
    `<input type="range" min="${s.minimum}" max="${s.maximum}" step="${s.step}" value="${s.default}" `+
    `id="ccs_${s.slider_id}" style="flex:1" oninput="document.getElementById('ccn_${s.slider_id}').value=this.value">`+
    `<input id="ccn_${s.slider_id}" value="${s.default}" style="width:54px" `+
    `onchange="document.getElementById('ccs_${s.slider_id}').value=this.value"></div>`).join('');
}
function ccState(){
  const st={};
  document.querySelectorAll('[id^=ccs_]').forEach(e=>{st[e.id.slice(4)]=parseFloat(e.value);});
  return st;
}
function ccRiskColor(l){return l==='high'?'var(--err)':l==='elevated'?'var(--warn)':l==='watch'?'#d4b106':'var(--ok,#4caf50)';}
function ccSevColor(s){return s==='critical'?'var(--err)':s==='caution'?'var(--warn)':s==='advisory'?'#d4b106':'var(--dim)';}
function ccEditWeight(id,cur){
  const w=prompt('Manual weight for '+id+' (pinned - the resolver will warn but not adjust it):',cur);
  if(w===null)return;
  const f=parseFloat(w);
  if(isNaN(f)){alert('Enter a number.');return;}
  ccOverrides[id]=f;ccResolve();
}
async function ccResolve(){
  if(!Object.keys(ccSel).length){document.getElementById('cc_stack').innerHTML='<span style="color:var(--dim)">Select LoRAs above.</span>';
    document.getElementById('cc_warnings').innerHTML='';return null;}
  const r=await(await fetch('/api/concept/resolve',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({loras:Object.keys(ccSel),slider_state:ccState(),weights:ccOverrides,output:v('outbase')||''})})).json();
  if(r.error){alert(r.error);return null;}
  ccLastStack=r;
  const row=i=>{
    const adj=i.adjusted?` <span title="auto-adjusted for stack balance" style="color:var(--warn)">&#9881;</span>`:'';
    const pin=i.pinned?` <span title="pinned manual weight" style="color:var(--acc,#4af)">&#128204;</span>`:'';
    const req=(i.requested_weight!=null&&i.requested_weight!==i.weight)?
      `<span style="color:var(--dim);text-decoration:line-through">${i.requested_weight}</span> &rarr; `:'';
    return `<tr><td>${i.family}</td><td>${i.lora_id}${pin}</td>`+
      `<td style="cursor:pointer" title="click to set a manual weight" `+
      `onclick="ccEditWeight('${i.lora_id}',${i.weight})">${req}<b>${i.weight}</b>${adj}</td>`+
      `<td style="color:var(--dim)">${i.reason}</td></tr>`;
  };
  const rows=[];
  if(r.identity_anchor)rows.push(row(r.identity_anchor));
  r.concept_loras.forEach(i=>rows.push(row(i)));
  const badge=`<span style="background:${ccRiskColor(r.risk_level)};color:#000;border-radius:10px;`+
    `padding:1px 10px;font-size:11px;font-weight:bold">${r.risk_level.toUpperCase()}</span>`;
  document.getElementById('cc_stack').innerHTML=
    `<div style="margin-bottom:6px">${badge} <b>${r.summary||''}</b></div>`+
    `<table><tr><th>family</th><th>LoRA</th><th>requested &rarr; resolved</th><th>reason</th></tr>${rows.join('')}</table>`+
    `<div style="margin-top:4px">concept strength <b>${r.total_concept_strength}</b> &middot; `+
    `identity preservation <b style="color:${ccRiskColor(r.risk_level)}">${r.identity_preservation_score}</b>`+
    ((r.influence_pressure||[]).length?` &middot; top pressure: <b>${r.influence_pressure[0].lora_id}</b> (${r.influence_pressure[0].pressure})`:'')+`</div>`;
  document.getElementById('cc_warnings').innerHTML=
    (r.warnings||[]).map(w=>`<div style="color:${ccSevColor(w.severity)}">[${w.severity}] ${w.message}</div>`).join('')+
    ((r.recommendations||[]).length?`<div style="margin-top:6px;border-top:1px solid #2a3142;padding-top:4px">`+
      `<b>Recommendations</b>`+r.recommendations.map(rc=>`<div style="color:var(--dim)">&bull; ${rc.message}</div>`).join('')+`</div>`:'');
  const hasRec=(r.recommendations||[]).some(rc=>Object.keys(rc.proposed_weights||{}).length);
  document.getElementById('cc_balance_btn').style.display=hasRec?'':'none';
  document.getElementById('cc_clearpins_btn').style.display=Object.keys(ccOverrides).length?'':'none';
  return r;
}
let wdPresets=[];
async function loadWardrobe(){
  if(wdPresets.length)return;
  const r=await(await fetch('/api/wardrobe/presets')).json();
  wdPresets=r.presets||[];
  document.getElementById('wd_region').innerHTML=wdPresets.map(p=>
    `<option value="${p.region_id}">${p.label}</option>`).join('');
  wdRegion();
}
function wdRegion(){
  const p=wdPresets.find(x=>x.region_id===v('wd_region'));if(!p)return;
  document.getElementById('wd_region_info').innerHTML=
    `${p.description} &middot; recommended denoise ${p.recommended_denoise[0]}-${p.recommended_denoise[1]} `+
    `&middot; identity priority <b>${p.identity_priority}</b> &middot; guidance: ${p.recommended_controlnets.join(', ')}`+
    `<br><span style="color:var(--dim)">${p.notes}</span>`;
}
function wdRequest(){
  const loras=[];
  if(document.getElementById('wd_stack').checked&&typeof ccSel!=='undefined')
    for(const k in ccSel)loras.push([k,ccSel[k]]);
  return {image_path:v('wd_image')||'',mask_path:v('wd_mask')||'',
    region_id:v('wd_region'),edit_mode:v('wd_mode'),
    garment_direction_prompt:v('wd_prompt')||'',
    selected_loras:loras,
    preserve_pose:document.getElementById('wd_pose').checked,
    strong_face_lock:document.getElementById('wd_facelock').checked,
    faceid_preset:v('wd_faceid')||'balanced',
    body_structure_lock:document.getElementById('wd_body').checked,
    silhouette_guidance:document.getElementById('wd_sil').checked,
    preserve_background:document.getElementById('wd_bg').checked,
    denoise:parseFloat(v('wd_denoise'))||0,
    seed:parseInt(v('wd_seed'))||42,output:v('outbase')||''};
}
async function wdReadiness(){
  const r=await(await fetch('/api/wardrobe/readiness',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(wdRequest())})).json();
  if(r.error){alert(r.error);return;}
  const reqs=(r.model_requirements||[]).map(m=>{
    const col=m.detected==='found'?'var(--ok,#4caf50)':m.detected==='missing'?'var(--err)':'var(--dim)';
    return `<tr><td>${m.category}</td><td style="color:${col}">${m.detected}</td>`+
      `<td style="color:var(--dim)">${m.guidance}<br><span style="font-size:10px">${(m.expected_paths||[]).join(', ')}</span></td></tr>`;
  }).join('');
  document.getElementById('wd_readiness').innerHTML=
    `<div><b>${r.region}</b> &middot; denoise ${r.denoise} &middot; identity preservation `+
    `<b>${r.identity_preservation_score}</b> (${r.identity_risk_level} risk)</div>`+
    (r.identity_tools?`<div>identity tools: <b>${r.identity_tools.join(', ')||'none'}</b>`+
      ((r.degraded_features||[]).length?` &middot; <span style="color:var(--warn)">degraded: ${r.degraded_features.join(', ')}</span>`:'')+`</div>`:'')+
    (r.auto_adjustment_suggestions||[]).map(s=>`<div style="color:var(--warn)">! ${s}</div>`).join('')+
    (r.consistency_notes||[]).map(s=>`<div style="color:var(--dim)">${s}</div>`).join('')+
    `<table><tr><th>component</th><th>status</th><th>guidance</th></tr>${reqs}</table>`;
}
async function wdGenerate(){
  if(!v('wd_image')){alert('Set the character image path first.');return;}
  const r=await(await fetch('/api/wardrobe/generate',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(wdRequest())})).json();
  document.getElementById('wd_result').innerHTML=r.ok?
    `Queued as hub job #${r.job_id} - watch the Pipeline page; results land in wardrobe_edits/.`:
    `<span style="color:var(--err)">${r.error||'failed'}</span>`;
}
async function gSearch(){
  const q=v('gsearch');if(!q)return;
  const r=await(await fetch(`/api/search?q=${encodeURIComponent(q)}&output=${encodeURIComponent(v('outbase')||'')}`)).json();
  const sec=(t,items,fmt)=>items&&items.length?`<div><b>${t}</b>${items.map(fmt).join('')}</div>`:'';
  document.getElementById('gresults').innerHTML=
    sec('LoRAs',r.loras,x=>`<div>&bull; ${x}</div>`)+
    sec('Presets',r.presets,x=>`<div>&bull; ${x}</div>`)+
    sec('Batches',r.batches,x=>`<div>&bull; ${x}</div>`)+
    sec('Frames',r.frames,x=>`<div>&bull; ${x.frame_id.slice(0,14)}: ${x.caption}</div>`)||
    '<div>no matches</div>';
}
async function ccEditSidecar(id,fam,risk,w){
  const family=prompt('Concept family (identity/character/wardrobe/fashion/lighting/pose/style/texture/detail/environment/composition/camera/refinement):',fam);
  if(family===null)return;
  const irisk=prompt('Identity risk (none/low/medium/high):',risk);if(irisk===null)return;
  const wd=prompt('Default weight:',w);if(wd===null)return;
  const conf=prompt('Known conflicts (comma-separated LoRA ids, blank=none):','');
  const notes=prompt('Notes (blank=keep):','');
  const r=await(await fetch('/api/concept/sidecar_save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({lora_id:id,family:family,identity_risk:irisk,weight_default:wd,
      known_conflicts:conf,notes:notes})})).json();
  if(r.ok){loadConceptCards();}else alert(r.error||'failed');
}
async function ccHistory(){
  const r=await(await fetch(`/api/concept/history?output=${encodeURIComponent(v('outbase')||'')}`)).json();
  document.getElementById('cc_presets').innerHTML='<b>Stack timeline (latest first)</b>'+
    (r.history||[]).map(h=>{
      const pl=h.payload||{};
      return `<div>${h.name.replace('hist_','')} &middot; score <b>${pl.score??'-'}</b> [${pl.risk||''}] `+
      `${(pl.loras||[]).map(l=>l[0]+':'+l[1]).join(' ')} `+
      `<button class="mini" onclick='ccApplyPreset({preset_version:2,handoff:{loras:${JSON.stringify(pl.loras||[])}},slider_state:{}})'>Load</button></div>`;
    }).join('')||'<div style="color:var(--dim)">No history yet - resolve a stack first.</div>';
}
async function loadStarters(){
  try{
    const ov=await(await fetch('/api/concept/workflow/overview')).json();
    const st=document.getElementById('cc_wf_status');
    if(!ov.ready){st.innerHTML='No identity anchor in the library yet - train or import a character identity LoRA, then Rescan.';
      document.getElementById('cc_starters').innerHTML='';return;}
    st.innerHTML=`${ov.library_size} LoRA(s) &middot; identity: <b>${ov.identity_candidates.join(', ')}</b> &middot; `+
      `concept families: ${Object.keys(ov.concept_families).join(', ')||'none yet'}`;
    const r=await(await fetch('/api/concept/starters')).json();
    document.getElementById('cc_starters').innerHTML=(r.starters||[]).map(s=>{
      if(!s.available)return `<div style="color:var(--dim);font-size:11px">${s.reasons[s.reasons.length-1]}</div>`;
      return `<div style="border:1px solid #2a3142;border-radius:6px;padding:8px;width:230px">`+
        `<b style="font-size:12px">${s.name}</b>`+
        `<div style="font-size:10px;color:var(--dim)">${s.stack.map(l=>l[0]+':'+l[1]).join('<br>')}</div>`+
        `<div style="font-size:10px">preservation <b style="color:${ccRiskColor(s.risk_level)}">${s.preservation_score}</b></div>`+
        (s.warnings.length?`<div style="font-size:9px;color:var(--warn)">${s.warnings[0]}</div>`:'')+
        `<button class="mini" style="margin-top:4px" title="${(s.reasons||[]).join('&#10;')}" `+
        `onclick='ccApplyStarter(${JSON.stringify(s.weights)})'>Apply stack</button></div>`;
    }).join('')||'<span style="color:var(--dim);font-size:11px">No starter stacks available yet.</span>';
  }catch(e){}
}
function ccApplyStarter(weights){
  ccSel={};ccOverrides={};
  for(const k in weights){ccSel[k]=weights[k];ccOverrides[k]=weights[k];}
  loadConceptCards();ccResolve();
  window.scrollTo({top:document.getElementById('cc_stack').offsetTop-80,behavior:'smooth'});
}
async function ccSaveWorkflowPreset(){
  const r=await ccResolve();if(!r)return;
  const name=prompt('Stack preset name:','character_stack_v1');if(!name)return;
  const weights={};
  if(r.identity_anchor)weights[r.identity_anchor.lora_id]=r.identity_anchor.weight;
  r.concept_loras.forEach(i=>{weights[i.lora_id]=i.weight;});
  const res=await(await fetch('/api/concept/workflow/save_preset',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:name,weights:weights,slider_state:ccState(),
      loras:Object.keys(ccSel),output:v('outbase')||''})})).json();
  alert(res.ok?(`Preset '${name}' saved (preservation ${res.preservation_score}).`):(res.error||'failed'));
  ccLoadPresets();
}
function ccApplyBalance(){
  if(!ccLastStack||!ccLastStack.recommended_weights)return;
  ccOverrides={...ccLastStack.recommended_weights};
  ccResolve();
}
async function ccSend(){
  const r=await ccResolve();if(!r)return;
  const name=prompt('Playground preset name:','concept_stack');if(!name)return;
  const res=await(await fetch('/api/concept/send_playground',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:name,stack:r,output:v('outbase')||''})})).json();
  alert(res.ok?('Preset written: '+res.path+' - Playground > Presets > Refresh > Load'):(res.error||'failed'));
}
async function ccSavePreset(){
  const r=await ccResolve();if(!r)return;
  const name=prompt('Stack preset name:','my_stack');if(!name)return;
  const res=await(await fetch('/api/concept/preset_save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:name,kind:'lora_stack',payload:{sel:ccSel,slider_state:ccState(),stack:r},output:v('outbase')||''})})).json();
  if(!res.ok)alert(res.error||'failed');else ccLoadPresets();
}
async function rgLoadBatches(){
  const r=await(await fetch(`/api/concept/variation/batches?output=${encodeURIComponent(v('outbase')||'')}`)).json();
  document.getElementById('cc_rg_batch').innerHTML=(r.batches||[]).map(b=>
    `<option value="${b.batch_id}">${b.batch_id} [${b.mode||''}] ${b.generated}/${b.jobs} generated</option>`).join('')||
    '<option value="">no batches yet</option>';
}
async function rgShow(){
  const bid=v('cc_rg_batch');if(!bid)return;
  const r=await(await fetch(`/api/concept/variation/batch/${bid}?output=${encodeURIComponent(v('outbase')||'')}`)).json();
  if(r.error){alert(r.error);return;}
  document.getElementById('cc_review').innerHTML=(r.jobs||[]).map(j=>{
    const img=j.output_path?`<img loading="lazy" src="/api/concept/variation/result/${bid}/${j.variation_id}" style="width:100%;border-radius:4px">`:
      `<div style="height:120px;background:#11161f;border-radius:4px;display:flex;align-items:center;justify-content:center;color:var(--dim)">${j.status}</div>`;
    return `<div style="width:200px;border:1px solid #2a3142;border-radius:6px;padding:6px">${img}`+
      `<div><b>${j.variation_id}</b> seed ${j.seed} &middot; ${ccRiskBadge(j.risk_level||'info')}</div>`+
      `<div style="color:var(--dim)">${JSON.stringify(j.slider_state)}</div>`+
      `<div style="color:var(--dim)">${(j.loras||[]).map(l=>l[0]+':'+l[1]).join(' ')}</div>`+
      `<div>preservation <b>${j.preservation_score!=null?j.preservation_score:'-'}</b>`+
      (j.measured_face_sim!=null?` &middot; measured face <b style="color:${j.measured_face_sim>=0.5?'var(--ok,#4caf50)':'var(--warn)'}">${j.measured_face_sim}</b>`:'')+`</div>`+
      `<button class="mini" style="margin-top:4px;width:100%" onclick="rgPromote('${bid}','${j.variation_id}')">Promote to preset</button></div>`;
  }).join('')||'<span style="color:var(--dim)">No jobs in this batch.</span>';
}
async function rgPromote(bid,vid){
  const name=prompt('Playground preset name for this winner:',`winner_${vid}`);if(!name)return;
  const r=await(await fetch('/api/concept/variation/promote',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({batch_id:bid,variation_id:vid,name:name})})).json();
  alert(r.ok?`Promoted -> Playground preset '${r.name}' (Presets > Refresh > Load).`:(r.error||'failed'));
}
async function ccDiffPresets(){
  const a=prompt('First preset name:');if(!a)return;
  const b=prompt('Second preset name:');if(!b)return;
  const r=await(await fetch('/api/concept/preset_diff',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({a:a,b:b})})).json();
  if(r.error){alert(r.error);return;}
  document.getElementById('cc_presets').innerHTML=
    `<div><b>Diff:</b> ${r.a_name} (score ${r.score_a??'-'}) vs ${r.b_name} (score ${r.score_b??'-'})</div>`+
    `<table><tr><th>LoRA</th><th>A</th><th>B</th><th>delta</th></tr>`+
    r.rows.map(x=>`<tr><td>${x.lora_id}</td><td>${x.a??'-'}</td><td>${x.b??'-'}</td>`+
      `<td style="color:${x.delta>0?'var(--warn)':'var(--dim)'}">${x.delta??''}</td></tr>`).join('')+`</table>`;
}
async function ccLoadPresets(){
  const r=await(await fetch(`/api/concept/presets?output=${encodeURIComponent(v('outbase')||'')}`)).json();
  document.getElementById('cc_presets').innerHTML=(r.presets||[]).map(p=>
    `<div><b>${p.name}</b> <span style="color:var(--dim)">[${p.kind}] ${p.updated_at||''}</span> `+
    `<button class="mini" onclick='ccApplyPreset(${JSON.stringify(p.payload)})'>Load</button></div>`).join('')||
    '<span style="color:var(--dim)">No saved presets.</span>';
}
function ccApplyPreset(payload){
  if(payload.preset_version===2){
    ccSel={};ccOverrides={};
    (payload.handoff&&payload.handoff.loras||[]).forEach(l=>{ccSel[l[0]]=l[1];ccOverrides[l[0]]=l[1];});
    const st2=payload.slider_state||{};
    for(const k in st2){const s=document.getElementById('ccs_'+k),n=document.getElementById('ccn_'+k);
      if(s)s.value=st2[k];if(n)n.value=st2[k];}
    loadConceptCards();ccResolve();return;
  }
  ccSel=payload.sel||{};
  const st=payload.slider_state||{};
  for(const k in st){const s=document.getElementById('ccs_'+k),n=document.getElementById('ccn_'+k);
    if(s){s.value=st[k];}if(n){n.value=st[k];}}
  loadConceptCards();ccResolve();
}
function ccAxisSpec(slId,valId){
  const sl=v(slId);if(!sl)return null;
  const raw=(v(valId)||'').trim();if(!raw)return null;
  if(raw.includes(':')){
    const p=raw.split(':').map(parseFloat);
    if(p.length>=2&&!isNaN(p[0])&&!isNaN(p[1]))
      return {slider:sl,minimum:p[0],maximum:p[1],step:(p.length>2&&!isNaN(p[2]))?p[2]:0.1};
    return null;
  }
  const vals=raw.split(',').map(parseFloat).filter(x=>!isNaN(x));
  return vals.length?{slider:sl,values:vals}:null;
}
function ccGridSpec(){
  const axes=[ccAxisSpec('cc_bv_slider','cc_bv_values'),
              ccAxisSpec('cc_bv_slider2','cc_bv_values2')].filter(a=>a);
  const seeds=(v('cc_bv_seeds')||'42').split(',').map(x=>parseInt(x)).filter(x=>!isNaN(x));
  return {slider_axes:axes,seeds:seeds.length?seeds:[42],mode:v('cc_bv_mode')||'low_risk'};
}
async function ccPreviewGrid(){
  const g=ccGridSpec();
  if(!g.slider_axes.length){document.getElementById('cc_bv_estimate').textContent='';return;}
  const r=await(await fetch('/api/concept/variation/preview',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(g)})).json();
  if(r.error)return;
  document.getElementById('cc_bv_estimate').innerHTML=
    `${r.estimated} planned job(s), cap ${r.cap}`+
    (r.within_cap?'':` <span style="color:var(--err)">- ${r.guidance}</span>`);
}
function ccRiskBadge(l){
  const col=l==='blocked_or_needs_review'?'var(--err)':l==='high_risk'?'var(--err)':
    l==='caution'?'var(--warn)':'var(--ok,#4caf50)';
  return `<span style="color:${col}">${l}</span>`;
}
async function ccBatch(){
  if(!Object.keys(ccSel).length){alert('Select LoRAs first.');return;}
  const g=ccGridSpec();
  if(!g.slider_axes.length){alert('Define at least one variation axis.');return;}
  const r=await(await fetch('/api/concept/batch_expand',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({loras:Object.keys(ccSel),base_state:ccState(),weights:ccOverrides,
      slider_axes:g.slider_axes,seeds:g.seeds,mode:g.mode,
      prompt_tail:v('cc_bv_tail')||'',output:v('outbase')||''})})).json();
  if(r.error){alert(r.error);return;}
  document.getElementById('cc_batch').innerHTML=
    `<div>Batch <b>${r.batch_id}</b> [${g.mode}]: ${r.jobs.length} variation job(s) saved to the manifest. `+
    `<button class="mini" onclick="ccGenerate('${r.batch_id}')">Generate Variations</button></div>`+
    `<table><tr><th>id</th><th>seed</th><th>slider state</th><th>stack</th><th>preservation</th><th>risk</th><th>warnings</th></tr>`+
    r.jobs.map(j=>`<tr><td>${j.variation_id}</td><td>${j.seed}</td>`+
      `<td>${JSON.stringify(j.slider_state)}</td><td>${j.loras.map(l=>l[0]+':'+l[1]).join(' ')}</td>`+
      `<td>${j.preservation_score!=null?j.preservation_score:'-'}</td>`+
      `<td>${ccRiskBadge(j.risk_level||'info')}</td>`+
      `<td style="color:var(--warn)">${(j.warnings||[]).join('; ')}</td></tr>`).join('')+'</table>'+
    `<div style="color:var(--dim)">Generation needs Forge running (Engines strip). CLI: lora-studio concept batch --spec ... --mode ${g.mode} --run</div>`;
}
async function ccGenerate(bid){
  const r=await(await fetch('/api/concept/variation/generate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({batch_id:bid,output:v('outbase')||''})})).json();
  alert(r.ok?('Queued as hub job #'+r.job_id+' - watch the Pipeline page; results land in variation_batches/.'):(r.error||'failed'));
}

async function appPoll(){
  try{
    const s=await(await fetch('/api/apps')).json();
    for(const name of ['playground','factory','forge']){
      const st=s[name];if(!st)continue;
      const pre=name==='playground'?'play':name;
      const b=document.getElementById(pre+'_badge');
      if(!b)continue;
      if(!st.script_found){b.className='badge idle';
        b.textContent=name==='forge'?'forge_root not set':'script not found';continue;}
      b.className='badge '+(st.running?'run':'idle');
      b.textContent=st.running?('running :'+st.port):(st.crashed?'CRASHED':'stopped');
      const hint=document.getElementById(pre+'_hint');
      if(st.crashed&&st.log_tail&&hint){
        hint.innerHTML='<b style="color:var(--err)">App crashed - last log lines:</b>'+
          '<pre style="font-size:10px;white-space:pre-wrap;color:var(--dim)">'+
          st.log_tail.replace(/</g,'&lt;')+'</pre>';
      }
      const fr=document.getElementById(pre+'_frame');
      if(fr){
        if(st.running&&fr.src==='about:blank')fr.src='http://127.0.0.1:'+st.port;
        if(!st.running&&fr.src!=='about:blank')fr.src='about:blank';
      }
    }
    for(const name of ['playground','factory','forge']){
      const st=s[name];
      const d=document.getElementById('dot_'+(name==='playground'?'play':name));
      if(!d)continue;
      const dot=d.querySelector('.dot'),lbl=d.querySelector('.dotlabel');
      if(!st||!st.script_found){dot.className='dot';lbl.textContent='n/a';continue;}
      dot.className='dot '+(st.running?'run':(st.crashed?'crash':''));
      lbl.textContent=st.running?(':'+st.port):(st.crashed?'crashed':'off');
    }
  }catch(e){}
  clearTimeout(window._appPollT);
  window._appPollT=setTimeout(appPoll,3000);
}
async function appStart(name){
  const r=await(await fetch('/api/apps/start',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({app:name})})).json();
  if(!r.ok)alert(r.error||'failed'); else appPoll();
}
async function appStop(name){
  await fetch('/api/apps/stop',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({app:name})});
  appPoll();
}
async function loadSweep(){
  const run=v('sw_run');
  if(!run){document.getElementById('sw_out').textContent='Enter a run name (see Runs table).';return;}
  const r=await(await fetch(`/api/sweep?run=${encodeURIComponent(run)}&output=${encodeURIComponent(v('outbase')||'')}`)).json();
  if(!r.sweep){document.getElementById('sw_out').textContent='No stored sweep for this run yet.';return;}
  const rows=(r.sweep.rows||[]).map(x=>
    `epoch ${String(x.epoch).padStart(3)}  likeness ${x.likeness??'-'}  flex ${x.flexibility??'-'}`).join('\\n');
  const b=r.sweep.best;
  document.getElementById('sw_out').textContent=rows+
    (b?`\\n\\nBEST: epoch ${b.epoch} (${b.label})${b.overfit_warning?'  [overfit warning]':''}`:'\\n\\n(unscored)');
}
async function loadEvals(){
  const r=await(await fetch(`/api/evals?output=${encodeURIComponent(v('outbase')||'')}`)).json();
  document.getElementById('evalrows').innerHTML=(r.summary||[]).map(x=>
    `<tr><td>${x.label}</td><td>${x.category}</td><td>${x.n}</td>`+
    `<td>${x.avg_likeness==null?'-':x.avg_likeness.toFixed(3)}</td></tr>`).join('');
  document.getElementById('evalimgs').innerHTML=(r.recent||[]).map(e=>
    `<div style="border:1px solid var(--border);border-radius:8px;overflow:hidden">`+
    `<img loading="lazy" src="/eval_img/${e.eval_id}?output=${encodeURIComponent(v('outbase')||'')}" style="width:100%;display:block">`+
    `<div style="font-size:9px;color:var(--dim);padding:3px 5px">${e.category} s${e.seed} `+
    `${e.likeness==null?'':'lk '+e.likeness.toFixed(2)}</div></div>`).join('');
}
async function loadTrainDatasets(){
  const r=await(await fetch(`/api/datasets?output=${encodeURIComponent(v('outbase')||'')}`)).json();
  document.getElementById('t_dataset').innerHTML=(r.datasets||[]).slice().reverse().map(d=>
    `<option value="${d.version}">v${String(d.version).padStart(3,'0')} ${d.recipe} (${d.train} imgs)</option>`).join('')
    ||'<option value="">(no builds yet)</option>';
  try{
    const p=await(await fetch('/api/train/presets')).json();
    if((p.presets||[]).length){
      const cur=v('t_preset');
      document.getElementById('t_preset').innerHTML=p.presets.map(x=>
        `<option${x===cur?' selected':''}>${x}</option>`).join('');
    }
  }catch(e){}
}
async function suggestPreset(){
  const ds=v('t_dataset');
  const out=document.getElementById('t_suggest');
  if(!ds){out.textContent='Build a dataset first (Builds tab)';return;}
  out.textContent='analyzing dataset...';
  const r=await(await fetch(`/api/train/suggest_preset?dataset=${ds}&output=${encodeURIComponent(v('outbase')||'')}`)).json();
  if(r.error){out.textContent='Error: '+r.error;return;}
  const alts=(r.alternative_presets||[]).map(a=>
    `<div>&middot; ${a.preset} (${a.confidence.toFixed(2)}) - ${a.reason}</div>`).join('');
  const warns=(r.warnings||[]).map(w=>`<div style="color:var(--warn,#d90)">! ${w}</div>`).join('');
  out.innerHTML=`<b>${r.recommended_preset}</b> &middot; confidence ${r.confidence.toFixed(2)}`+
    ` <button class="mini" onclick="usePreset('${r.recommended_preset}')">Use This Preset</button>`+
    `<div style="margin-top:4px">${r.reason}</div>${alts}${warns}`;
}
function usePreset(name){
  const sel=document.getElementById('t_preset');
  if(![...sel.options].some(o=>o.value===name)){
    sel.innerHTML+=`<option>${name}</option>`;
  }
  sel.value=name;
}
function runTrain(){
  const ds=v('t_dataset');
  if(!ds){alert('Build a dataset first (Builds tab)');return;}
  fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stage:'train',params:{dataset:ds,preset:v('t_preset'),
      name:v('t_name'),dry:document.getElementById('t_dry').checked,output:v('outbase')}})})
    .then(r=>r.json()).then(d=>{if(!d.ok)alert(d.error||'failed');});
}
async function loadRuns(){
  const r=await(await fetch(`/api/runs?output=${encodeURIComponent(v('outbase')||'')}`)).json();
  document.getElementById('runrows').innerHTML=(r.runs||[]).slice().reverse().map(x=>
    `<tr><td>${x.run_id}</td><td>${x.name}</td><td>v${String(x.dataset_version).padStart(3,'0')}</td>`+
    `<td>${x.preset}</td><td class="jq-${x.status==='completed'?'done':(x.status==='running'?'running':'error')}">${x.status}</td>`+
    `<td>${x.step}/${x.total_steps} (e${x.epoch}/${x.total_epochs})</td>`+
    `<td>${x.last_loss==null?'-':x.last_loss.toFixed(4)}</td></tr>`).join('');
  document.getElementById('m_run').innerHTML=(r.runs||[]).slice().reverse().map(x=>
    `<option value="${x.run_id}">#${x.run_id} ${x.name}</option>`).join('');
  if((r.runs||[]).length)loadCurve();
}
async function loadCurve(){
  const id=v('m_run'); if(!id)return;
  const r=await(await fetch(`/api/run_metrics?run=${id}&output=${encodeURIComponent(v('outbase')||'')}`)).json();
  const pts=r.points||[]; const cv=document.getElementById('curve'); const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  if(pts.length<2){document.getElementById('curveinfo').textContent='No metrics yet.';return;}
  const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  ctx.strokeStyle='#00d4aa';ctx.lineWidth=1.5;ctx.beginPath();
  pts.forEach((p,i)=>{
    const x=8+(p[0]-x0)/(x1-x0||1)*(cv.width-16);
    const y=cv.height-10-(p[1]-y0)/(y1-y0||1)*(cv.height-20);
    i?ctx.lineTo(x,y):ctx.moveTo(x,y);
  });
  ctx.stroke();
  document.getElementById('curveinfo').textContent=
    `${pts.length} pts | loss ${y1.toFixed(4)} -> ${ys[ys.length-1].toFixed(4)} (min ${y0.toFixed(4)})`;
}
document.querySelectorAll('.tab[data-p]').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab[data-p]').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('.pane').forEach(x=>x.classList.remove('on'));
  t.classList.add('on');
  document.getElementById('pane-'+t.dataset.p).classList.add('on');
});
function v(id){return document.getElementById(id).value}
function c(id){return document.getElementById(id).checked}
function collect(stage){
  if(stage==='extract')return{video_dirs:v('vdirs'),photos_dir:v('pdir'),output:v('outbase'),
    fps:v('fps'),jq:v('jq'),seg:v('seg'),limit:v('limitv'),pmode:v('pmode'),
    dry:c('dry'),resume:c('resume'),photos:c('photos'),overwrite:c('overwrite')};
  if(stage==='curate')return{output:v('outbase'),ham:v('ham'),workers:v('workers'),
    minsharp:v('minsharp'),minb:v('minb'),maxb:v('maxb'),rescore:c('rescore'),
    smart:c('smart'),anchordir:v('anchordir'),idthr:v('idthr'),minface:v('minface'),
    buildanchor:c('buildanchor'),cluster:c('cluster'),k:v('kclusters'),cbatch:v('cbatch')};
  if(stage==='caption')return{output:v('outbase'),trigger:v('cap_trigger'),cls:v('cap_cls'),
    thr:v('cap_thr'),maxtags:v('cap_max'),blacklist:v('cap_black'),remap:v('cap_remap'),
    prune:v('cap_prune'),pony:c('cap_pony'),force:c('cap_force')};
  if(stage==='package')return{output:v('outbase'),token:v('token'),cls:v('cls'),
    repeats:v('repeats'),maxpv:v('maxpv'),maxtot:v('maxtot'),caption:v('caption'),
    captions:c('captions'),quota:v('quota')};
  if(stage==='testgen')return{output:v('outbase'),backend:v('g_backend'),url:v('g_url'),
    loras:v('g_loras'),prompt:v('g_prompt'),neg:v('g_neg'),steps:v('g_steps'),
    cfg:v('g_cfg'),seed:v('g_seed'),w:v('g_w'),h:v('g_h'),
    init:v('g_init'),strength:v('g_strength')};
  if(stage==='matrix'){
    const cats=['likeness','flexibility','pose','outfit','style']
      .filter(x=>document.getElementById('mc_'+x).checked).join(',');
    return{output:v('outbase'),lora:v('m_lora'),weight:v('m_weight'),
      steps:v('m_steps'),backend:v('g_backend'),url:v('g_url'),categories:cats};
  }
  if(stage==='merge')return{output:v('outbase'),loras:v('mg_loras'),name:v('mg_name'),
    te:v('mg_te'),down:v('mg_down'),mid:v('mg_mid'),up:v('mg_up'),
    preview:document.getElementById('mg_preview').checked,backend:v('g_backend')};
  if(stage==='wizard')return{output:v('outbase'),type:wzType,trigger:v('wz_trigger'),
    cls:v('wz_cls'),name:v('wz_name'),train:c('wz_train'),matrix:c('wz_matrix'),
    backend:v('wz_backend')};
  if(stage==='sweep')return{output:v('outbase'),run:v('sw_run'),
    backend:v('sw_backend'),gap:v('sw_gap')};
  if(stage==='mega')return{output:v('outbase')};
  return{};
}
let wzType='auto';
async function wzAnalyze(){
  const r=await(await fetch(`/api/wizard/analyze?output=${encodeURIComponent(v('outbase')||'')}`)).json();
  if(r.error){document.getElementById('wz_stats').textContent='Error: '+r.error;return;}
  const a=r.analysis;
  document.getElementById('wz_stats').textContent=
    `selected ${a.selected} | scanned ${a.scanned} | captioned ${a.captioned}\n`+
    `face rate ${(a.face_rate*100).toFixed(0)}% | clusters ${a.clusters} (entropy ${a.cluster_entropy})\n`+
    `framing ${JSON.stringify(a.framing_mix)}\nexplicit-tag rate ${(a.explicit_rate*100).toFixed(0)}%`;
  document.getElementById('wz_types').innerHTML=r.ranking.map((t,i)=>
    `<div class="fcard ${i===0?'s-selected':''}" style="padding:8px;margin-bottom:6px;cursor:pointer" `+
    `onclick="wzPick(this,'${t.type}')"><b>${t.type}</b> &middot; score ${t.score.toFixed(2)}`+
    `${i===0?' &middot; recommended':''}<div class="sub">${t.reason}</div></div>`).join('');
  wzType=r.ranking[0].type;
}
function wzPick(el,t){
  wzType=t;
  document.querySelectorAll('#wz_types .fcard').forEach(x=>x.classList.remove('s-selected'));
  el.classList.add('s-selected');
}
async function loadAB(){
  const a=v('ab_a'),b=v('ab_b');
  const r=await(await fetch(`/api/ab?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}&output=${encodeURIComponent(v('outbase')||'')}`)).json();
  if(r.error){alert(r.error);return;}
  document.getElementById('ab_tally').textContent=
    `Votes so far - ${a}: ${r.tally[a]||0} | ${b}: ${r.tally[b]||0}`;
  document.getElementById('ab_pairs').innerHTML=r.pairs.map(p=>{
    const flip=Math.random()<0.5;
    const L=flip?p.b:p.a, R=flip?p.a:p.b, Ll=flip?b:a, Rl=flip?a:b;
    return `<div><div class="sub">${p.category} / seed ${p.seed}</div>
      <div style="display:flex;gap:10px">
      <div style="flex:1"><img src="/eval_img/${L}?output=${encodeURIComponent(v('outbase')||'')}" style="width:100%;border-radius:8px">
        <button class="mini" style="margin-top:4px" onclick="abVote('${Ll}','${a}','${b}','${p.category}',${p.seed},this)">Pick left</button></div>
      <div style="flex:1"><img src="/eval_img/${R}?output=${encodeURIComponent(v('outbase')||'')}" style="width:100%;border-radius:8px">
        <button class="mini" style="margin-top:4px" onclick="abVote('${Rl}','${a}','${b}','${p.category}',${p.seed},this)">Pick right</button></div>
      </div></div>`;
  }).join('');
}
async function abVote(winner,a,b,cat,seed,btn){
  await fetch('/api/ab_vote',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({winner:winner,a:a,b:b,category:cat,seed:seed,output:v('outbase')||''})});
  btn.textContent='voted: '+winner; btn.disabled=true;
  loadAB!==undefined; // tally refresh on next load
}
function runStage(stage){
  let params = stage==='pipeline'
    ? {extract:collect('extract'),curate:collect('curate'),
       caption:collect('caption'),package:collect('package')}
    : collect(stage);
  fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stage:stage,params:params})})
    .then(r=>r.json()).then(d=>{if(!d.ok)alert(d.error||'failed');});
}
function cancelJob(id){fetch('/api/cancel',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({id:id})});}
function fmt(n){return n==null?'-':n.toLocaleString()}
async function poll(){
  try{
    const s=await(await fetch('/api/state')).json();
    const out=document.getElementById('out');
    if(s.job.output){out.textContent=s.job.output;out.scrollTop=out.scrollHeight;}
    const b=document.getElementById('badge');
    b.className='badge '+(s.job.running?'run':'idle');
    b.textContent=s.job.running?('running: '+s.job.stage):'idle';
    document.getElementById('log').textContent=s.log.join('\\n');
    const t=s.stats.totals||{};const f=s.stats.frames||{};
    document.getElementById('stats').innerHTML=
      stat(fmt(t.sources),'sources')+stat(fmt(t.frames_extracted),'frames')+
      stat(fmt(f.selected),'selected')+stat(fmt(t.faces_scanned),'faces scanned')+
      stat(fmt(f.packaged),'packaged');
    document.getElementById('jq').innerHTML=(s.job.queue||[]).slice().reverse().map(j=>
      `<tr><td>${j.id}</td><td>${j.stage}</td><td class="jq-${j.status}">${j.status}</td>`+
      `<td>${j.started_at?j.started_at.slice(11):'-'}</td>`+
      `<td>${j.status==='queued'?`<button class="mini stop" onclick="cancelJob(${j.id})">x</button>`:''}</td></tr>`).join('');
  }catch(e){}
  setTimeout(poll,1500);
}
function stat(v,k){return `<div class="stat"><div class="v">${v??'-'}</div><div class="k">${k}</div></div>`}

let shownFrameIds=[];
async function bulkVerdict(action){
  if(!shownFrameIds.length){alert('Nothing loaded.');return;}
  if(!confirm(`${action==='reject'?'Reject':'Keep'} ${shownFrameIds.length} frames shown on this page?`))return;
  await fetch('/api/verdict_bulk',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame_ids:shownFrameIds,action:action,output:v('outbase')||''})});
  loadFrames();
}
async function loadFrames(){
  const st=v('fstatus'), fr=v('fframing'), sd=v('fstudy')||'', so=v('fsort')||'';
  const cp=encodeURIComponent(v('fcaption')||'');
  const r=await(await fetch(`/api/frames?status=${st}&framing=${fr}&study=${sd}&sort=${so}&caption=${cp}&offset=${pg*60}&limit=60&output=${encodeURIComponent(v('outbase')||'')}`)).json();
  shownFrameIds=(r.frames||[]).map(f=>f.id);
  document.getElementById('pinfo').textContent=`${pg*60+1}-${Math.min((pg+1)*60,r.total)} of ${r.total}`;
  document.getElementById('frames').innerHTML=r.frames.map(f=>{
    let cls=f.status==='packaged'?'s-packaged':(f.status==='selected'?'s-selected':(f.status.startsWith('rejected')||f.status==='duplicate'?'s-rejected':''));
    let sim=f.identity_sim!=null?` id ${f.identity_sim.toFixed(2)}`:'';
    let cl=f.cluster_id!=null?` c${f.cluster_id}`:'';
    let fra=f.framing?`<span class="tag">${f.framing}</span>`:'';
    let cap=f.caption?`<br><span style="color:#6f7b92">${f.caption.slice(0,70)}</span>`:'';
    let capesc=(f.caption||'').replace(/"/g,'&quot;');
    let studyShort={figure_study_candidate:'figure study',fashion_study_candidate:'fashion study',
      lingerie_fashion_candidate:'lingerie/fashion',form_proportion_candidate:'form+proportion',
      rejected_study:'study: rejected'}[f.study]||'';
    let studyChip=studyShort?`<span class="tag" style="top:24px;background:#1a3a5caa" `+
      `title="confidence ${f.study_conf!=null?f.study_conf.toFixed(2):'-'} | review ${f.study_review||'-'}`+
      `${f.study_export?' | export eligible':''}\\nreasons: ${(f.study_reasons||'').replace(/"/g,'')}">`+
      `${studyShort}${f.study_review==='needs_review'?' ?':''}</span>`:'';
    return `<div class="fcard ${cls}" title="${capesc}" onclick="verdict('${f.id}',this)">${fra}${studyChip}`+
      `<button class="mini" style="position:absolute;top:5px;right:5px;background:#000a;color:#fff" `+
      `onclick="event.stopPropagation();editCaption('${f.id}',this)" data-cap="${capesc}">&#9998;</button>`+
      `<button class="mini" style="position:absolute;top:28px;right:5px;background:#000a;color:#fff" `+
      `title="set study label" onclick="event.stopPropagation();editStudy('${f.id}')">S</button>`+
      `<img loading="lazy" src="/thumb/${f.id}"><div class="fi">${f.name}<br>`+
      `sharp ${f.sharpness==null?'-':f.sharpness.toFixed(0)}${sim}${cl}${cap}</div></div>`;
  }).join('');
}
let kbIdx=-1;
function kbCards(){return Array.from(document.querySelectorAll('#frames .fcard'));}
function kbFocus(i){
  const cs=kbCards();if(!cs.length)return;
  kbIdx=Math.max(0,Math.min(cs.length-1,i));
  cs.forEach(c=>c.style.outline='');
  cs[kbIdx].style.outline='2px solid var(--acc,#4af)';
  cs[kbIdx].scrollIntoView({block:'nearest'});
}
document.addEventListener('keydown',e=>{
  if(!document.getElementById('page-review').classList.contains('on'))return;
  if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;
  const cs=kbCards();
  if(e.key==='ArrowRight'){kbFocus(kbIdx+1);e.preventDefault();}
  else if(e.key==='ArrowLeft'){kbFocus(kbIdx-1);e.preventDefault();}
  else if(e.key==='ArrowDown'){kbFocus(kbIdx+6);e.preventDefault();}
  else if(e.key==='ArrowUp'){kbFocus(kbIdx-6);e.preventDefault();}
  else if((e.key==='k'||e.key==='K'||e.key==='r'||e.key==='R')&&kbIdx>=0&&cs[kbIdx]){
    cs[kbIdx].click();          // verdict toggle on the focused card
    kbFocus(kbIdx+1);e.preventDefault();
  }
  else if(e.key===']'){pg++;loadFrames();kbIdx=-1;}
  else if(e.key==='['&&pg>0){pg--;loadFrames();kbIdx=-1;}
});
async function editStudy(id){
  const lbl=prompt('Study label:\\n1 figure_study_candidate\\n2 fashion_study_candidate\\n'+
    '3 lingerie_fashion_candidate\\n4 form_proportion_candidate\\n5 rejected_study\\n\\nEnter 1-5:');
  if(lbl===null)return;
  const map={'1':'figure_study_candidate','2':'fashion_study_candidate',
    '3':'lingerie_fashion_candidate','4':'form_proportion_candidate','5':'rejected_study'};
  const label=map[lbl.trim()]||lbl.trim();
  const r=await(await fetch('/api/study_override',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame_id:id,study_primary:label,output:v('outbase')||''})})).json();
  if(r.ok)loadFrames(); else alert(r.error||'invalid label');
}
async function editCaption(id,btn){
  const cur=btn.dataset.cap||'';
  const text=prompt('Caption (manual edits are never overwritten by re-captioning):',cur);
  if(text===null)return;
  const r=await(await fetch('/api/caption',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame_id:id,caption:text,output:v('outbase')||''})})).json();
  if(r.ok)loadFrames(); else alert(r.error||'failed');
}
async function loadTags(){
  const r=await(await fetch(`/api/tags?output=${encodeURIComponent(v('outbase')||'')}`)).json();
  document.getElementById('tagrows').innerHTML=r.tags.map(t=>{
    const hot=t.pct>=80?' style="color:var(--warn)"':'';
    return `<tr${hot}><td>${t.tag}</td><td>${t.count}</td><td>${t.pct.toFixed(0)}%</td>`+
      `<td>${t.pct>=80?'prune candidate':''}</td></tr>`;
  }).join('');
}
async function loadRecipes(){
  const r=await(await fetch('/api/project')).json();
  const names=Object.keys(r.project.recipes||{});
  document.getElementById('b_recipe').innerHTML=
    names.length?names.map(n=>`<option>${n}</option>`).join(''):'<option value="">(no recipes in project file)</option>';
}
async function loadBuilds(){
  const r=await(await fetch(`/api/datasets?output=${encodeURIComponent(v('outbase')||'')}`)).json();
  document.getElementById('buildrows').innerHTML=(r.datasets||[]).slice().reverse().map(d=>
    `<tr><td>v${String(d.version).padStart(3,'0')}</td><td>${d.recipe}</td><td>${d.train}</td>`+
    `<td>${d.val}</td><td style="font-family:monospace;font-size:10px">${d.hash}</td>`+
    `<td>${(d.built_at||'').slice(0,16)}</td><td>${d.note}</td></tr>`).join('');
}
function runBuild(){
  const rec=v('b_recipe');
  if(!rec){alert('No recipes defined in the project file');return;}
  fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stage:'build',params:{recipe:rec,note:v('b_note'),output:v('outbase')}})})
    .then(r=>r.json()).then(d=>{if(!d.ok)alert(d.error||'failed');});
}
async function runDiff(){
  const r=await(await fetch(`/api/diff?a=${v('d_a')}&b=${v('d_b')}&output=${encodeURIComponent(v('outbase')||'')}`)).json();
  document.getElementById('diffout').textContent=r.error?('Error: '+r.error):
    r.summary+'\\nconcepts: '+JSON.stringify(r.concept_mix_a)+' -> '+JSON.stringify(r.concept_mix_b);
}
async function verdict(id,el){
  const reject=!el.classList.contains('s-rejected');
  const r=await(await fetch('/api/verdict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame_id:id,action:reject?'reject':'keep',
      output:v('outbase')||''})})).json();
  if(r.ok){el.classList.toggle('s-rejected',reject);el.classList.toggle('s-selected',!reject);}
}

async function loadProject(){
  const r=await(await fetch('/api/project')).json();
  const p=r.project;
  document.getElementById('prjpath').textContent='Loaded from: '+(r.path||'(defaults - not yet saved)');
  try{
    const rt=await(await fetch('/api/runtime')).json();
    document.getElementById('runtimeinfo').textContent=
      'Runtime: '+(rt.selected_device||'?')+(rt.cuda_device?' ('+rt.cuda_device+', '+rt.cuda_vram_gb+' GB)':'')+
      ' | dtype '+(rt.selected_dtype||'?')+(rt.tf32_enabled?' | tf32':'')+(rt.in_colab?' | colab':'');
  }catch(e){}
  s_name.value=p.name;s_vdirs.value=p.video_dirs.join('\\n');s_pdir.value=p.photos_dir;
  s_out.value=p.output_base;s_token.value=p.trigger_token;s_cls.value=p.class_word;
  s_port.value=p.ui_port;s_anchor.value=p.anchor_dir;s_forge.value=p.forge_root;
  s_sd.value=p.sd_scripts_dir||'';s_base.value=p.base_model||'';s_lout.value=p.lora_output_dir||'';
  s_path.value=r.path||'project.toml';
}
async function saveProject(){
  const data={name:s_name.value,video_dirs:s_vdirs.value.split('\\n').map(x=>x.trim()).filter(x=>x),
    photos_dir:s_pdir.value,output_base:s_out.value,trigger_token:s_token.value,
    class_word:s_cls.value,ui_port:parseInt(s_port.value)||7861,
    anchor_dir:s_anchor.value,forge_root:s_forge.value,
    sd_scripts_dir:s_sd.value,base_model:s_base.value,lora_output_dir:s_lout.value};
  const r=await(await fetch('/api/project',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({project:data,save_path:s_path.value})})).json();
  document.getElementById('s_msg').textContent=r.ok?
    ('Saved to '+r.path+' at '+new Date().toLocaleTimeString()+' (applies to new jobs immediately)'):('Error: '+r.error);
}
// ---------- MergeForge ----------
let mfEntries=[];
function mfSel(){return mfEntries.filter(e=>document.getElementById('mfc_'+e.lora_id)&&document.getElementById('mfc_'+e.lora_id).checked).map(e=>e.lora_id);}
async function mfScan(){
  const r=await(await fetch('/api/mergeforge/scan')).json();
  mfEntries=r.entries||[];
  document.getElementById('mf_scan_sum').textContent=
    r.library_size+' LoRAs | roles: '+Object.entries(r.role_counts||{}).map(([k,v])=>k+':'+v).join(' ')+
    (r.merge_ready?' | merge-ready':' | '+(r.notes||[]).join('; '));
  document.getElementById('mf_library').innerHTML=mfEntries.map(e=>
    '<div style="display:flex;gap:8px;align-items:center;padding:4px 2px;border-bottom:1px solid var(--line)">'+
    '<input type="checkbox" id="mfc_'+e.lora_id+'">'+
    '<div style="flex:1"><b style="font-size:12px">'+e.display_name+'</b>'+
    ' <span class="sub">'+e.role+' &middot; '+e.family+'</span></div>'+
    '<span class="sub" title="'+e.health.reasons.join('; ')+'">health '+e.health.grade+' ('+e.health.score+')</span></div>').join('');
  mfRecs();
}
async function mfRecs(){
  const r=await(await fetch('/api/mergeforge/recommendations')).json();
  const g=r.groups||[];
  document.getElementById('mf_recs').innerHTML=g.length?g.map(x=>
    '<div style="padding:6px 0;border-bottom:1px solid var(--line)">'+
    '<b style="font-size:12px">'+x.title+'</b> <span class="sub">'+x.compatibility.verdict+' ('+x.compatibility.score+')</span><br>'+
    '<span class="sub">'+x.members.map(m=>m.lora_id+' @'+m.weight).join(' + ')+'</span><br>'+
    '<span class="sub">'+x.rationale+'</span> '+
    '<button class="mini" onclick="mfUseGroup(\\''+x.group_id+'\\')">Use</button></div>').join('')
    :'need at least 2 classified LoRAs for recommendations';
  window._mfGroups=g;
}
function mfUseGroup(gid){
  const g=(window._mfGroups||[]).find(x=>x.group_id===gid); if(!g)return;
  mfEntries.forEach(e=>{const c=document.getElementById('mfc_'+e.lora_id); if(c)c.checked=false;});
  g.members.forEach(m=>{const c=document.getElementById('mfc_'+m.lora_id); if(c)c.checked=true;});
  document.getElementById('mf_name').value=gid+'_v1';
  mfPlan();
}
let mfDraft=null;
async function mfPlan(){
  const sel=mfSel();
  if(sel.length<2){document.getElementById('mf_plan').innerHTML='<span class="sub">tick at least 2 LoRAs in the scan list</span>';return;}
  const r=await(await fetch('/api/mergeforge/plan',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({loras:sel})})).json();
  if(r.error){document.getElementById('mf_plan').textContent=r.error;return;}
  mfDraft=r.recipe_draft;
  const c=r.compatibility;
  document.getElementById('mf_plan').innerHTML=
    '<b>Compatibility: '+c.overall_verdict+' ('+c.overall_score+')</b><br>'+
    (c.pairs||[]).filter(p=>p.warnings.length).map(p=>'<span class="sub">'+p.warnings.join('; ')+'</span>').join('<br>')+
    '<div style="margin-top:6px">'+(mfDraft.inputs||[]).map((i,ix)=>
      '<div style="display:flex;gap:6px;align-items:center;margin:2px 0"><span style="flex:1;font-size:12px">'+i.name+'</span>'+
      '<input type="number" step="0.05" style="width:70px" id="mfw_'+ix+'" value="'+i.weight+'"></div>').join('')+'</div>'+
    '<span class="sub">'+(r.weights.notes||[]).join('; ')+'</span>';
  document.getElementById('mf_go').style.display=r.ready?'inline-block':'none';
  document.getElementById('mf_msg').textContent=r.ready?'':'verdict is avoid - adjust selection';
}
async function mfMerge(){
  if(!mfDraft)return;
  mfDraft.inputs.forEach((i,ix)=>{const w=document.getElementById('mfw_'+ix); if(w)i.weight=parseFloat(w.value)||i.weight;});
  const r=await(await fetch('/api/mergeforge/merge',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({recipe:mfDraft,output_name:v('mf_name')||'mergeforge_v1',
      target_rank:parseInt(v('mf_rank'))||0})})).json();
  document.getElementById('mf_msg').textContent=r.ok?('queued as job #'+r.job_id+' - watch the queue below'):('error: '+r.error);
  if(r.ok)mfLoadRecipes();
}
async function mfLoadRecipes(){
  const r=await(await fetch('/api/mergeforge/recipes?output='+encodeURIComponent(v('outbase')||''))).json();
  const recs=r.recipes||[];
  document.getElementById('mf_recipes').innerHTML=recs.length?recs.map(x=>
    '<div style="padding:5px 0;border-bottom:1px solid var(--line)"><b style="font-size:12px">'+x.name+'</b>'+
    ' <span class="sub">'+x.method+' &middot; '+(x.created_at||'').slice(0,16)+'</span><br>'+
    '<span class="sub">'+(x.inputs||[]).map(i=>i.name+' @'+i.weight).join(' + ')+'</span>'+
    (x.output?'<br><span class="sub">-&gt; '+x.output+'</span>':'')+
    ' <button class="mini" onclick="mfRerun(\\''+x.recipe_id+'\\')">Re-run</button></div>').join('')
    :'none yet';
  window._mfRecipes=recs;
}
async function mfRerun(rid){
  const rec=(window._mfRecipes||[]).find(x=>x.recipe_id===rid); if(!rec)return;
  const r=await(await fetch('/api/mergeforge/merge',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({recipe:rec,output_name:rec.name+'_rerun'})})).json();
  document.getElementById('mf_msg').textContent=r.ok?('re-run queued as job #'+r.job_id):('error: '+r.error);
}
poll();
appPoll();   // Playground is the default page - start app status checks now
</script></body></html>
"""


# -----------------------------
# HTTP handler
# -----------------------------

class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        pass

    @property
    def project(self) -> Project:
        return getattr(self.server, "project", Project())  # type: ignore[attr-defined]

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _out_base(self, query: dict) -> Path:
        q = (query.get("output") or [""])[0]
        return Path(q or getattr(self.server, "output_base", self.project.output_base)
                    ).expanduser()

    # ---------- GET ----------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/":
            prj = self.project
            from ..curate import smart_available
            smart_ok, _ = smart_available()
            html = (UI_HTML
                    .replace("__PNAME__", prj.name)
                    .replace("__VDIRS__", "\n".join(prj.video_dirs))
                    .replace("__PDIR__", prj.photos_dir)
                    .replace("__OUTBASE__", prj.output_base)
                    .replace("__ANCHOR__", prj.anchor_dir)
                    .replace("__TOKEN__", prj.trigger_token)
                    .replace("__CLS__", prj.class_word)
                    .replace("__QUOTA__", DEFAULT_QUOTA)
                    .replace("__SMARTLBL__", " (+identity)" if smart_ok else ""))
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/state":
            stats: dict = {}
            try:
                out_base = self._out_base(query)
                if (out_base / DB_NAME).exists():
                    conn = manifest.connect(out_base)
                    stats = manifest.stats(conn)
                    conn.close()
            except Exception:
                stats = {}
            self._json({
                "job": QUEUE.state(),
                "log": list(LOG_BUFFER)[-50:],
                "stats": stats,
            })

        elif path == "/api/frames":
            try:
                out_base = self._out_base(query)
                status = (query.get("status") or ["selected"])[0]
                framing = (query.get("framing") or [""])[0]
                study = (query.get("study") or [""])[0]
                sort = (query.get("sort") or [""])[0]
                cap = (query.get("caption") or [""])[0]
                offset = int((query.get("offset") or ["0"])[0])
                limit = min(200, int((query.get("limit") or ["60"])[0]))
                conn = manifest.connect(out_base)
                where = "f.status = ?"
                params: list = [status]
                if framing:
                    where += " AND d.framing = ?"
                    params.append(framing)
                if cap:
                    where += " AND c.caption_text LIKE ?"
                    params.append(f"%{cap}%")
                # Study Intelligence Layer filters (LEFT JOIN: old manifests
                # simply return no study rows - never crashes)
                if study in ("figure_study_candidate", "fashion_study_candidate",
                             "lingerie_fashion_candidate",
                             "form_proportion_candidate"):
                    where += " AND s.study_primary = ?"
                    params.append(study)
                elif study == "needs_review":
                    where += " AND s.study_review_status = 'needs_review'"
                elif study == "export_eligible":
                    where += " AND s.study_export_eligible = 1"
                elif study == "identity_strong":
                    where += " AND s.identity_lock_score >= 0.7"
                elif study == "pose_strong":
                    where += " AND s.pose_clarity_score >= 0.7"
                order = {"study_confidence": "s.study_confidence DESC",
                         "identity_lock": "s.identity_lock_score DESC",
                         "figure_study": "s.figure_study_score DESC",
                         "fashion_study": "s.fashion_study_score DESC",
                         }.get(sort, "f.path")
                joins = ("LEFT JOIN detections d ON d.frame_id=f.frame_id "
                         "LEFT JOIN study_labels s ON s.frame_id=f.frame_id "
                         "LEFT JOIN captions c ON c.frame_id=f.frame_id ")
                total = conn.execute(
                    f"SELECT COUNT(*) FROM frames f {joins}WHERE {where}",
                    params,
                ).fetchone()[0]
                rows = conn.execute(
                    f"SELECT f.frame_id, f.path, f.sharpness, f.brightness, f.status, "
                    f"d.framing, d.identity_sim, f.cluster_id, c.caption_text, "
                    f"s.study_primary, s.study_confidence, s.study_reason_codes, "
                    f"s.study_review_status, s.study_export_eligible "
                    f"FROM frames f {joins}"
                    f"WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
                    params + [limit, offset],
                ).fetchall()
                conn.close()
                self._json({"total": total, "frames": [
                    {"id": r[0], "path": r[1], "name": Path(r[1]).name,
                     "sharpness": r[2], "brightness": r[3], "status": r[4],
                     "framing": r[5], "identity_sim": r[6], "cluster_id": r[7],
                     "caption": r[8], "study": r[9], "study_conf": r[10],
                     "study_reasons": r[11], "study_review": r[12],
                     "study_export": r[13]} for r in rows
                ]})
            except Exception as exc:
                self._json({"total": 0, "frames": [], "error": str(exc)})

        elif path == "/api/runtime":
            try:
                from ..runtime import configure_from_project
                self._json(configure_from_project(self.project))
            except Exception as exc:
                self._json({"error": str(exc)})

        elif path == "/api/train/presets":
            try:
                from ..train.presets import PRESETS
                self._json({"presets": sorted(PRESETS)})
            except Exception as exc:
                self._json({"presets": [], "error": str(exc)})

        elif path == "/api/train/suggest_preset":
            try:
                from ..train.preset_recommender import suggest_preset
                ds = int((query.get("dataset") or ["0"])[0])
                recipe = (query.get("recipe") or [None])[0]
                conn = manifest.connect(self._out_base(query))
                try:
                    rec = suggest_preset(conn, ds, recipe_name=recipe)
                finally:
                    conn.close()
                self._json(rec)
            except ValueError as exc:
                self._json({"error": str(exc)})
            except Exception as exc:
                self._json({"error": str(exc)})

        elif path == "/api/mergeforge/scan":
            try:
                from ..mergeforge import analyze_library
                self._json(analyze_library(self.project))
            except Exception as exc:
                self._json({"entries": [], "role_counts": {},
                            "library_size": 0, "merge_ready": False,
                            "notes": [str(exc)]})

        elif path == "/api/mergeforge/recommendations":
            try:
                from ..mergeforge import build_recommendations
                self._json({"groups": build_recommendations(self.project)})
            except Exception as exc:
                self._json({"groups": [], "error": str(exc)})

        elif path == "/api/mergeforge/recipes":
            try:
                from ..mergeforge import load_recipes
                self._json({"recipes":
                            load_recipes(self._out_base(query))})
            except Exception as exc:
                self._json({"recipes": [], "error": str(exc)})

        elif path == "/api/concept/cards":
            try:
                from .. import lora_explorer as lx
                cards = lx.scan_loras_cached(
                    self.project,
                    force=bool((query.get('rescan') or [''])[0]))
                CC_PREVIEW_INDEX.clear()
                for c in cards:
                    for lv, fp in c.preview_levels.items():
                        CC_PREVIEW_INDEX[(c.lora_id, lv)] = fp
                if (query.get("rescan") or [""])[0]:
                    conn = manifest.connect(self._out_base(query))
                    lx.sync_profiles_to_manifest(conn, cards)
                    conn.close()
                out = lx.filter_cards(
                    cards,
                    family=(query.get("family") or [""])[0],
                    tag=(query.get("tag") or [""])[0],
                    search=(query.get("search") or [""])[0],
                    sort=(query.get("sort") or ["name"])[0])
                pv = (query.get("preview") or [""])[0]
                if pv == "has":
                    out = [c for c in out if c.has_preview]
                elif pv == "missing":
                    out = [c for c in out if not c.has_preview]
                payload_out = lx.build_explorer_payload(out)
                payload_out["cards"] = payload_out["items"]   # legacy alias
                payload_out["roots"] = [str(d)
                                        for d in lx.lora_dirs(self.project)]
                self._json(payload_out)
            except Exception as exc:
                self._json({"cards": [], "items": [], "error": str(exc)})

        elif path == "/api/wardrobe/presets":
            from ..wardrobe import list_region_presets
            self._json({"presets": list_region_presets()})

        elif path == "/api/concept/starters":
            try:
                from .. import lora_explorer as lx
                from ..starter_stacks import recommend_starter_stacks
                cards = lx.scan_loras_cached(self.project)
                self._json({"starters": recommend_starter_stacks(
                    cards, self.project.base_model)})
            except Exception as exc:
                self._json({"starters": [], "error": str(exc)})

        elif path == "/api/concept/workflow/overview":
            try:
                from .. import lora_explorer as lx
                from ..stack_workflow import workflow_overview
                self._json(workflow_overview(
                    lx.scan_loras_cached(self.project)))
            except Exception as exc:
                self._json({"ready": False, "error": str(exc)})

        elif path == "/api/concept/variation/axes":
            from ..concept_control import variation_axes
            from ..batch_variations import VARIATION_MODES
            self._json({"axes": variation_axes(),
                        "modes": {k: {"label": v["label"], "cap": v["cap"],
                                      "notes": v["notes"]}
                                  for k, v in VARIATION_MODES.items()}})

        elif path == "/api/concept/variation/batches":
            try:
                conn = manifest.connect(self._out_base(query))
                rows = conn.execute(
                    "SELECT batch_id, mode, job_count, created_at, "
                    "(SELECT COUNT(*) FROM variation_jobs j WHERE "
                    "j.batch_id=b.batch_id AND j.status='generated') "
                    "FROM variation_batches b ORDER BY created_at DESC "
                    "LIMIT 25").fetchall()
                conn.close()
                self._json({"batches": [
                    {"batch_id": r[0], "mode": r[1], "jobs": r[2],
                     "created_at": r[3], "generated": r[4]}
                    for r in rows]})
            except Exception as exc:
                self._json({"batches": [], "error": str(exc)})

        elif path.startswith("/api/concept/variation/result/"):
            # generated variation image, served only via its manifest row
            data = b""
            try:
                bid, _, vid = path.rsplit("/", 2)[-2], "", path.rsplit(
                    "/", 1)[-1]
                bid = path.split("/api/concept/variation/result/",
                                 1)[1].split("/")[0]
                conn = manifest.connect(self._out_base(query))
                row = conn.execute(
                    "SELECT output_path FROM variation_jobs WHERE "
                    "batch_id=? AND variation_id=?", (bid, vid)).fetchone()
                conn.close()
                if row and row[0] and Path(row[0]).exists():
                    data = Path(row[0]).read_bytes()
            except Exception:
                data = b""
            if data:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)
            return

        elif path.startswith("/api/concept/variation/batch/"):
            try:
                from ..batch_variations import load_batch
                bid = path.rsplit("/", 1)[1]
                conn = manifest.connect(self._out_base(query))
                data = load_batch(conn, bid)
                conn.close()
                self._json(data or {"error": f"unknown batch {bid}"})
            except Exception as exc:
                self._json({"error": str(exc)})

        elif path == "/api/concept/sliders":
            from ..concept_control import slider_specs
            self._json({"sliders": slider_specs()})

        elif path == "/api/concept/history":
            try:
                from ..concept_control import load_presets
                conn = manifest.connect(self._out_base(query))
                hist = load_presets(conn, "stack_history")[-25:]
                conn.close()
                self._json({"history": list(reversed(hist))})
            except Exception as exc:
                self._json({"history": [], "error": str(exc)})

        elif path == "/api/search":
            try:
                from ..insights import global_search
                conn = manifest.connect(self._out_base(query))
                res = global_search(conn, self.project,
                                    (query.get("q") or [""])[0])
                conn.close()
                self._json(res)
            except Exception as exc:
                self._json({"error": str(exc)})

        elif path == "/api/concept/presets":
            try:
                from ..concept_control import load_presets
                conn = manifest.connect(self._out_base(query))
                presets = load_presets(conn)
                conn.close()
                self._json({"presets": presets})
            except Exception as exc:
                self._json({"presets": [], "error": str(exc)})

        elif path.startswith("/api/concept/lora_preview/"):
            # served ONLY from the explorer index (no path traversal):
            # /api/concept/lora_preview/<lora_id>/<level>
            data = b""
            try:
                parts = path.split("/api/concept/lora_preview/", 1)[1]
                lora_id, _, level = parts.strip("/").rpartition("/")
                if not CC_PREVIEW_INDEX:
                    from .. import lora_explorer as lx
                    for c in lx.scan_loras_cached(self.project):
                        for lv, fp0 in c.preview_levels.items():
                            CC_PREVIEW_INDEX[(c.lora_id, lv)] = fp0
                fp = CC_PREVIEW_INDEX.get((lora_id, level or "default"))
                if fp and Path(fp).exists():
                    data = Path(fp).read_bytes()
            except Exception:
                data = b""
            if data:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        elif path.startswith("/thumb/"):
            frame_id = path.split("/thumb/", 1)[1].strip("/")
            out_base = self._out_base(query)
            data = b""
            try:
                conn = manifest.connect(out_base)
                row = conn.execute(
                    "SELECT path FROM frames WHERE frame_id = ?", (frame_id,)
                ).fetchone()
                conn.close()
                if row:
                    thumb = ensure_thumb(out_base, frame_id, Path(row[0]))
                    if thumb:
                        data = thumb.read_bytes()
            except Exception:
                data = b""
            if data:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()

        elif path == "/api/project":
            self._json({
                "project": asdict(self.project),
                "path": getattr(self.server, "project_path", None),
            })

        elif path == "/api/tags":
            try:
                import json as _json
                conn = manifest.connect(self._out_base(query))
                freq: dict[str, int] = {}
                total = 0
                for (tj,) in conn.execute(
                    "SELECT tags_json FROM captions WHERE tags_json IS NOT NULL"
                ):
                    total += 1
                    try:
                        for name, _p in _json.loads(tj):
                            freq[name] = freq.get(name, 0) + 1
                    except Exception:
                        pass
                conn.close()
                total = max(1, total)
                tags = [
                    {"tag": n, "count": c, "pct": 100.0 * c / total}
                    for n, c in sorted(freq.items(), key=lambda x: -x[1])[:120]
                ]
                self._json({"tags": tags, "captioned": total})
            except Exception as exc:
                self._json({"tags": [], "error": str(exc)})

        elif path == "/api/datasets":
            try:
                from ..builder import list_datasets
                conn = manifest.connect(self._out_base(query))
                rows = list_datasets(conn)
                conn.close()
                self._json({"datasets": rows})
            except Exception as exc:
                self._json({"datasets": [], "error": str(exc)})

        elif path == "/api/apps":
            self._json(app_status())

        elif path == "/api/sweep":
            try:
                from ..sweep import sweep_summary
                run = (query.get("run") or [""])[0]
                conn = manifest.connect(self._out_base(query))
                s = sweep_summary(conn, run)
                conn.close()
                self._json({"sweep": s})
            except Exception as exc:
                self._json({"sweep": None, "error": str(exc)})

        elif path == "/api/wizard/analyze":
            try:
                from ..wizard import analyze, detect_type
                conn = manifest.connect(self._out_base(query))
                a = analyze(conn)
                conn.close()
                self._json({"analysis": a, "ranking": detect_type(a)})
            except Exception as exc:
                self._json({"error": str(exc)})

        elif path == "/api/ab":
            try:
                a = (query.get("a") or [""])[0]
                b = (query.get("b") or [""])[0]
                conn = manifest.connect(self._out_base(query))
                if not a or not b:
                    self._json({"error": "set both A and B eval labels"})
                    return

                def rows(label):
                    return {
                        (r[1], r[2]): r[0] for r in conn.execute(
                            "SELECT eval_id, category || '|' || prompt, seed "
                            "FROM evals WHERE label=?", (label,))
                    }
                A, B = rows(a), rows(b)
                pairs = [
                    {"a": A[k], "b": B[k],
                     "category": k[0].split("|")[0], "seed": k[1]}
                    for k in sorted(set(A) & set(B))
                ][:30]
                votes = manifest.meta_get(conn, "ab_votes", []) or []
                tally: dict = {}
                for vte in votes:
                    if {vte.get("a"), vte.get("b")} == {a, b}:
                        tally[vte["winner"]] = tally.get(vte["winner"], 0) + 1
                conn.close()
                self._json({"pairs": pairs, "tally": tally})
            except Exception as exc:
                self._json({"pairs": [], "tally": {}, "error": str(exc)})

        elif path == "/api/evals":
            try:
                from ..eval.matrix import eval_summary
                conn = manifest.connect(self._out_base(query))
                summary = eval_summary(conn)
                recent = [
                    {"eval_id": r[0], "category": r[1], "seed": r[2], "likeness": r[3]}
                    for r in conn.execute(
                        "SELECT eval_id, category, seed, likeness FROM evals "
                        "ORDER BY eval_id DESC LIMIT 24"
                    )
                ]
                conn.close()
                self._json({"summary": summary, "recent": recent})
            except Exception as exc:
                self._json({"summary": [], "recent": [], "error": str(exc)})

        elif path.startswith("/eval_img/"):
            eid = path.split("/eval_img/", 1)[1].strip("/")
            data = b""
            try:
                conn = manifest.connect(self._out_base(query))
                row = conn.execute(
                    "SELECT image_path FROM evals WHERE eval_id = ?", (int(eid),)
                ).fetchone()
                conn.close()
                if row and Path(row[0]).exists():
                    data = Path(row[0]).read_bytes()
            except Exception:
                data = b""
            if data:
                self.send_response(200)
                ctype = "image/png" if row[0].endswith(".png") else "image/jpeg"
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()

        elif path == "/api/runs":
            try:
                from ..train.kohya import list_runs
                conn = manifest.connect(self._out_base(query))
                rows = list_runs(conn)
                conn.close()
                self._json({"runs": rows})
            except Exception as exc:
                self._json({"runs": [], "error": str(exc)})

        elif path == "/api/run_metrics":
            try:
                from ..train.kohya import run_metrics
                conn = manifest.connect(self._out_base(query))
                pts = run_metrics(conn, int((query.get("run") or ["0"])[0]))
                conn.close()
                self._json({"points": pts})
            except Exception as exc:
                self._json({"points": [], "error": str(exc)})

        elif path == "/api/diff":
            try:
                from ..builder import diff_datasets
                conn = manifest.connect(self._out_base(query))
                d = diff_datasets(
                    conn,
                    int((query.get("a") or ["0"])[0]),
                    int((query.get("b") or ["0"])[0]),
                )
                conn.close()
                self._json(d)
            except Exception as exc:
                self._json({"error": str(exc)})
        else:
            self._json({"error": "not found"}, 404)

    # ---------- POST ----------

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            payload = {}

        if path == "/api/stop":
            QUEUE.stop_current()
            self._json({"ok": True})
        elif path == "/api/apps/start":
            self._json(start_app(str(payload.get("app", "")),
                                 getattr(self.server, "project_path", None)))
        elif path == "/api/apps/stop":
            self._json(stop_app(str(payload.get("app", ""))))
        elif path == "/api/cancel":
            self._json({"ok": QUEUE.cancel(int(payload.get("id", 0)))})
        elif path == "/api/mergeforge/plan":
            try:
                from ..mergeforge import wizard_plan
                self._json(wizard_plan(
                    self.project,
                    [str(i) for i in (payload.get("loras") or [])]))
            except Exception as exc:
                self._json({"ready": False, "error": str(exc)})

        elif path == "/api/mergeforge/merge":
            try:
                from ..mergeforge import execute_recipe, validate_recipe
                rec, problems = validate_recipe(payload.get("recipe") or {})
                if len(rec.get("inputs") or []) < 2:
                    self._json({"ok": False,
                                "error": "; ".join(problems)
                                or "need at least 2 inputs"})
                    return
                out_base = self._out_base({})
                name = str(payload.get("output_name") or rec["name"])
                rank = int(payload.get("target_rank") or 0)
                prj = self.project

                def factory():
                    return execute_recipe(prj, out_base, rec,
                                          output_name=name,
                                          target_rank=rank)
                job_id = QUEUE.submit(f"mergeforge {name}", factory)
                self._json({"ok": True, "job_id": job_id,
                            "recipe_id": rec["recipe_id"],
                            "notes": problems})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)})

        elif path == "/api/concept/resolve":
            try:
                from .. import lora_explorer as lx
                from ..concept_control import (ConceptSliderState,
                                               resolve_controlled_stack)
                cards = lx.scan_loras_cached(self.project)
                sel = set(payload.get("loras") or [])
                chosen = [c for c in cards if c.lora_id in sel]
                state = ConceptSliderState(
                    values=payload.get("slider_state") or {})
                overrides = {k: float(v) for k, v in
                             (payload.get("weights") or {}).items()}
                stack = resolve_controlled_stack(
                    chosen, state, self.project.base_model,
                    overrides=overrides or None,
                    pinned=set(payload.get("pinned") or overrides))
                try:
                    from ..insights import log_stack_history
                    conn = manifest.connect(self._out_base({}))
                    log_stack_history(conn, {
                        "loras": ([[stack.identity_anchor.lora_id,
                                    stack.identity_anchor.weight]]
                                  if stack.identity_anchor else [])
                        + [[i.lora_id, i.weight]
                           for i in stack.concept_loras],
                        "score": stack.identity_preservation_score,
                        "risk": stack.risk_level})
                    conn.close()
                except Exception:
                    pass
                self._json(stack.to_json())
            except Exception as exc:
                self._json({"error": str(exc)})

        elif path == "/api/concept/send_playground":
            try:
                from ..pipeline_dag import write_playground_preset
                st = payload.get("stack") or {}
                loras = []
                if st.get("identity_anchor"):
                    a = st["identity_anchor"]
                    loras.append((a["lora_id"], a["weight"]))
                loras += [(i["lora_id"], i["weight"])
                          for i in st.get("concept_loras", [])]
                if not loras:
                    self._json({"ok": False, "error": "empty stack"})
                    return
                target = write_playground_preset(
                    self.project, str(payload.get("name") or "concept_stack"),
                    loras)
                self._json({"ok": True, "path": str(target)})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)})

        elif path == "/api/playground/stack/add-lora":
            try:
                from .. import lora_explorer as lx
                from ..pipeline_dag import write_playground_preset
                import json as _json
                lora_id = str(payload.get("lora_id") or "")
                cards = {c.lora_id: c for c in lx.scan_loras(self.project)}
                if lora_id not in cards:
                    self._json({"ok": False, "error": "unknown LoRA id"})
                    return
                c = cards[lora_id]
                w = float(payload.get("weight")
                          or c.profile.weight_default)
                # merge into the persistent explorer stack preset
                target = (Path(__file__).resolve().parents[2] / "outputs"
                          / "playground_presets.json")
                current: list = []
                try:
                    existing = _json.loads(target.read_text())
                    current = list(existing.get("explorer_stack", {})
                                   .get("loras", []))
                except Exception:
                    pass
                current = [it for it in current if it[0] != lora_id]
                entry = [lora_id, round(w, 2)]
                # identity LoRA priority: anchors stay first in the stack
                if c.profile.family in ("identity", "character"):
                    current.insert(0, entry)
                else:
                    current.append(entry)
                write_playground_preset(
                    self.project, "explorer_stack",
                    [(n, wt) for n, wt in current])
                self._json({"ok": True, "stack": current,
                            "entry": {"name": lora_id, "path": c.path,
                                      "weight": round(w, 2),
                                      "source": "visual_explorer",
                                      "concept_tags":
                                          c.profile.influence_tags}})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)})

        elif path == "/api/wardrobe/readiness":
            try:
                from .. import lora_explorer as lx
                from ..wardrobe import (WardrobeEditRequest,
                                        analyze_edit_readiness)
                known = WardrobeEditRequest.__dataclass_fields__
                req = WardrobeEditRequest(**{k: v for k, v in payload.items()
                                             if k in known})
                cards = lx.scan_loras_cached(self.project)
                self._json(analyze_edit_readiness(self.project, req, cards))
            except Exception as exc:
                self._json({"error": str(exc)})

        elif path == "/api/wardrobe/generate":
            try:
                from ..wardrobe import WardrobeEditRequest
                known = WardrobeEditRequest.__dataclass_fields__
                req = WardrobeEditRequest(**{k: v for k, v in payload.items()
                                             if k in known})
                if not Path(req.image_path).expanduser().exists():
                    self._json({"ok": False,
                                "error": "Character image not found - "
                                         "check the image path."})
                    return
                prj = self.project
                out_base = Path(
                    str(payload.get("output")
                        or getattr(self.server, "output_base",
                                   prj.output_base))).expanduser()

                def factory():
                    from ..wardrobe import generate_wardrobe_edit
                    conn = manifest.connect(out_base)
                    return generate_wardrobe_edit(prj, conn, req)
                job_id = QUEUE.submit(
                    f"wardrobe_edit {req.region_id}", factory)
                self._json({"ok": True, "job_id": job_id})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)})

        elif path == "/api/concept/variation/promote":
            try:
                from ..batch_variations import load_batch
                from ..pipeline_dag import write_playground_preset
                bid = str(payload.get("batch_id") or "")
                vid = str(payload.get("variation_id") or "")
                conn = manifest.connect(self._out_base({}))
                data = load_batch(conn, bid)
                conn.close()
                job = next((j for j in data.get("jobs", [])
                            if j["variation_id"] == vid), None)
                if not job:
                    self._json({"ok": False, "error": "unknown job"})
                    return
                name = str(payload.get("name")
                           or f"promoted_{bid}_{vid}")
                target = write_playground_preset(
                    self.project, name,
                    [(n, w) for n, w in job["loras"]])
                self._json({"ok": True, "path": str(target),
                            "name": name})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)})

        elif path == "/api/concept/preset_diff":
            try:
                from ..concept_control import load_presets
                from ..stack_workflow import diff_stack_presets
                conn = manifest.connect(self._out_base({}))
                presets = {pr["name"]: pr["payload"]
                           for pr in load_presets(conn)}
                conn.close()
                a = presets.get(str(payload.get("a") or ""))
                b = presets.get(str(payload.get("b") or ""))
                if a is None or b is None:
                    self._json({"error": "preset not found"})
                    return
                self._json(diff_stack_presets(a, b))
            except Exception as exc:
                self._json({"error": str(exc)})

        elif path == "/api/concept/workflow/save_preset":
            try:
                from .. import lora_explorer as lx
                from ..concept_control import (ConceptSliderState,
                                               resolve_controlled_stack,
                                               save_preset)
                from ..stack_workflow import make_stack_preset
                cards = lx.scan_loras_cached(self.project)
                sel = payload.get("weights") or {}
                chosen = [c for c in cards if c.lora_id in sel
                          or c.lora_id in (payload.get("loras") or [])]
                stack = resolve_controlled_stack(
                    chosen,
                    ConceptSliderState(values=payload.get("slider_state")
                                       or {}),
                    self.project.base_model,
                    overrides={k: float(v) for k, v in sel.items()} or None)
                rec = make_stack_preset(
                    str(payload.get("name") or "stack_preset"), stack,
                    requested_weights=sel,
                    slider_state=payload.get("slider_state") or {},
                    notes=str(payload.get("notes") or ""))
                conn = manifest.connect(Path(
                    str(payload.get("output")
                        or getattr(self.server, "output_base",
                                   self.project.output_base))).expanduser())
                save_preset(conn, rec["name"], "lora_stack", rec)
                conn.close()
                self._json({"ok": True, "preset_id": rec["preset_id"],
                            "preservation_score":
                                rec["preservation_score"]})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)})

        elif path == "/api/concept/sidecar_save":
            try:
                from .. import lora_explorer as lx
                cards = {c.lora_id: c for c in
                         lx.scan_loras_cached(self.project, force=True)}
                c = cards.get(str(payload.get("lora_id") or ""))
                if not c:
                    self._json({"ok": False, "error": "unknown LoRA id"})
                    return
                prof = c.profile
                for k in ("family", "identity_risk", "notes"):
                    if payload.get(k):
                        setattr(prof, k, str(payload[k]))
                if payload.get("weight_default"):
                    prof.weight_default = float(payload["weight_default"])
                if payload.get("known_conflicts") is not None:
                    prof.known_conflicts = [
                        s.strip() for s in
                        str(payload["known_conflicts"]).split(",")
                        if s.strip()]
                sp = lx.save_sidecar(Path(c.path), prof)
                lx._SCAN_CACHE["key"] = None      # invalidate
                self._json({"ok": True, "sidecar": str(sp)})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)})

        elif path == "/api/concept/preset_save":
            try:
                from ..concept_control import save_preset
                conn = manifest.connect(Path(
                    str(payload.get("output")
                        or getattr(self.server, "output_base",
                                   self.project.output_base))).expanduser())
                save_preset(conn, str(payload.get("name") or "preset"),
                            str(payload.get("kind") or "lora_stack"),
                            payload.get("payload") or {})
                conn.close()
                self._json({"ok": True})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)})

        elif path == "/api/concept/variation/preview":
            try:
                from ..batch_variations import (estimate_job_count,
                                                parse_axes)
                axes = parse_axes(payload.get("slider_axes") or [])
                est = estimate_job_count(
                    axes, [int(s) for s in (payload.get("seeds") or [42])],
                    str(payload.get("mode") or "low_risk"))
                est["axis_values"] = {a.slider: a.resolve_values(est["mode"])
                                      for a in axes}
                self._json(est)
            except Exception as exc:
                self._json({"error": str(exc)})

        elif path == "/api/concept/variation/generate":
            try:
                from ..batch_variations import load_batch
                bid = str(payload.get("batch_id") or "")
                out_base = Path(
                    str(payload.get("output")
                        or getattr(self.server, "output_base",
                                   self.project.output_base))).expanduser()
                conn0 = manifest.connect(out_base)
                data = load_batch(conn0, bid)
                conn0.close()
                if not data:
                    self._json({"ok": False, "error": f"unknown batch {bid}"})
                    return
                prj = self.project

                def factory():
                    from ..batch_variations import (VariationJob,
                                                    run_batch_generator)
                    conn = manifest.connect(out_base)
                    jobs = [VariationJob(
                        batch_id=bid, variation_id=j["variation_id"],
                        prompt=j.get("prompt", ""),
                        negative=j.get("negative", ""),
                        seed=j["seed"], loras=j["loras"],
                        slider_state=j["slider_state"],
                        warnings=j["warnings"], payload={},
                        output_path=j.get("output_path") or "")
                        for j in data["jobs"]]
                    # rebuild payloads deterministically from stored rows
                    from ..base_models import detect_profile
                    from ..eval.forge_api import ForgeClient
                    prof = detect_profile(prj.base_model)
                    for j in jobs:
                        j.payload = ForgeClient.build_txt2img_payload(
                            prompt=j.prompt, negative=j.negative,
                            steps=prof["steps"], cfg=prof["cfg"],
                            width=prof["width"], height=prof["height"],
                            seed=j.seed, sampler=prof["sampler"],
                            clip_skip=prof["clip_skip"],
                            loras=[(n, w) for n, w in j.loras])
                    return run_batch_generator(prj, conn, jobs)
                job_id = QUEUE.submit(f"variation_batch {bid}", factory)
                self._json({"ok": True, "job_id": job_id})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)})

        elif path == "/api/concept/batch_expand":
            try:
                from .. import lora_explorer as lx
                from ..batch_variations import (VariationGrid, expand_grid,
                                                save_batch)
                cards = lx.scan_loras_cached(self.project)
                sel = set(payload.get("loras") or [])
                chosen = [c for c in cards if c.lora_id in sel]
                grid = VariationGrid(
                    prompt_tail=str(payload.get("prompt_tail") or ""),
                    seeds=[int(s) for s in (payload.get("seeds") or [42])],
                    slider_axes=payload.get("slider_axes") or [],
                    base_state=payload.get("base_state") or {},
                    mode=str(payload.get("mode") or "low_risk"),
                    overrides={k: float(v) for k, v in
                               (payload.get("weights") or {}).items()})
                jobs = expand_grid(self.project, chosen, grid)
                conn = manifest.connect(Path(
                    str(payload.get("output")
                        or getattr(self.server, "output_base",
                                   self.project.output_base))).expanduser())
                bid = save_batch(conn, self.project, grid, jobs)
                conn.close()
                from dataclasses import asdict
                self._json({"batch_id": bid,
                            "jobs": [{k: v for k, v in asdict(j).items()
                                      if k != "payload"} for j in jobs]})
            except Exception as exc:
                self._json({"error": str(exc)})

        elif path == "/api/study_override":
            try:
                out_base = Path(
                    str(payload.get("output")
                        or getattr(self.server, "output_base", self.project.output_base))
                ).expanduser()
                fid = payload.get("frame_id") or ""
                label = payload.get("study_primary") or ""
                allowed = {"figure_study_candidate", "fashion_study_candidate",
                           "lingerie_fashion_candidate",
                           "form_proportion_candidate", "rejected_study"}
                if not fid or label not in allowed:
                    self._json({"ok": False, "error": "invalid label"})
                    return
                from ..study import set_manual_label
                conn = manifest.connect(out_base)
                set_manual_label(conn, fid, label)
                conn.close()
                self._json({"ok": True})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)})

        elif path == "/api/verdict_bulk":
            try:
                out_base = Path(
                    str(payload.get("output")
                        or getattr(self.server, "output_base", self.project.output_base))
                ).expanduser()
                ids = [str(i) for i in payload.get("frame_ids", [])][:500]
                new_status = ("rejected_manual"
                              if payload.get("action") == "reject" else "selected")
                conn = manifest.connect(out_base)
                conn.executemany(
                    "UPDATE frames SET status=? WHERE frame_id=?",
                    [(new_status, i) for i in ids],
                )
                conn.commit()
                conn.close()
                self._json({"ok": True, "updated": len(ids)})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 500)
        elif path == "/api/verdict":
            try:
                out_base = Path(
                    str(payload.get("output")
                        or getattr(self.server, "output_base", self.project.output_base))
                ).expanduser()
                frame_id = str(payload["frame_id"])
                action = payload.get("action", "keep")
                new_status = "rejected_manual" if action == "reject" else "selected"
                conn = manifest.connect(out_base)
                conn.execute(
                    "UPDATE frames SET status=? WHERE frame_id=?", (new_status, frame_id)
                )
                conn.commit()
                conn.close()
                self._json({"ok": True, "status": new_status})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 500)
        elif path == "/api/ab_vote":
            try:
                out_base = Path(
                    str(payload.get("output")
                        or getattr(self.server, "output_base", self.project.output_base))
                ).expanduser()
                conn = manifest.connect(out_base)
                votes = manifest.meta_get(conn, "ab_votes", []) or []
                votes.append({
                    "winner": str(payload.get("winner")),
                    "a": str(payload.get("a")), "b": str(payload.get("b")),
                    "category": str(payload.get("category")),
                    "seed": int(payload.get("seed", 0)), "ts": now_iso(),
                })
                manifest.meta_set(conn, "ab_votes", votes)
                conn.close()
                self._json({"ok": True})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 500)
        elif path == "/api/caption":
            try:
                out_base = Path(
                    str(payload.get("output")
                        or getattr(self.server, "output_base", self.project.output_base))
                ).expanduser()
                conn = manifest.connect(out_base)
                conn.execute(
                    "INSERT INTO captions(frame_id, caption_text, edited, updated_at) "
                    "VALUES (?,?,1,?) ON CONFLICT(frame_id) DO UPDATE SET "
                    "caption_text=excluded.caption_text, edited=1, "
                    "updated_at=excluded.updated_at",
                    (str(payload["frame_id"]), str(payload.get("caption", "")), now_iso()),
                )
                conn.commit()
                conn.close()
                self._json({"ok": True})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 500)
        elif path == "/api/project":
            try:
                data = payload.get("project", {})
                known = set(Project.__dataclass_fields__)
                if "recipes" not in data:           # Settings form has no recipe
                    data["recipes"] = self.project.recipes  # editor yet - keep them
                prj = Project(**{k: v for k, v in data.items() if k in known})
                save_path = Path(str(payload.get("save_path") or "project.toml")).expanduser()
                save_project(prj, save_path)
                self.server.project = prj  # type: ignore[attr-defined]
                self.server.project_path = str(save_path)  # type: ignore[attr-defined]
                self.server.output_base = prj.output_base  # type: ignore[attr-defined]
                self._json({"ok": True, "path": str(save_path)})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 500)
        elif path == "/api/run":
            try:
                stage = payload.get("stage")
                p = payload.get("params", {})
                if stage == "pipeline":
                    ids = []
                    for sub_stage in ("extract", "curate", "caption", "package"):
                        factory = self._build_factory(sub_stage, p.get(sub_stage, {}))
                        if factory:
                            ids.append(QUEUE.submit(sub_stage, factory))
                    self._json({"ok": True, "job_ids": ids})
                    return
                factory = self._build_factory(stage, p)
                if factory is None:
                    self._json({"ok": False, "error": f"unknown stage: {stage}"})
                    return
                job_id = QUEUE.submit(stage, factory)
                self._json({"ok": True, "job_id": job_id})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 500)
        else:
            self._json({"error": "not found"}, 404)

    # ---------- factories ----------

    def _build_factory(self, stage: str, p: dict):
        prj = self.project
        out_base = Path(str(p.get("output") or prj.output_base)).expanduser()
        self.server.output_base = str(out_base)  # type: ignore[attr-defined]

        if stage == "extract":
            vdirs = [l.strip() for l in str(p.get("video_dirs", "")).splitlines() if l.strip()]
            cfg = ExtractConfig(
                output_base=out_base,
                fps=float(p.get("fps", 0.25)),
                jpeg_quality=int(float(p.get("jq", 2))),
                segment_seconds=int(float(p.get("seg", 300))),
                use_personfromvid=bool(p.get("pfv")),
                import_photos=bool(p.get("photos")),
                photo_import_mode=str(p.get("pmode", "hardlink")),
                resume=bool(p.get("resume")),
                dry_run=bool(p.get("dry")),
                limit_videos=int(float(p.get("limit", 0) or 0)),
                overwrite=bool(p.get("overwrite")),
            )
            pdir = str(p.get("photos_dir") or prj.photos_dir)
            return lambda: pipeline_generator(vdirs or prj.video_dirs, pdir, cfg)

        if stage == "curate":
            basic_cfg = CurateConfig(
                output_base=out_base,
                hamming_threshold=int(float(p.get("ham", 4))),
                min_sharpness=float(p.get("minsharp", 35)),
                min_brightness=float(p.get("minb", 18)),
                max_brightness=float(p.get("maxb", 242)),
                workers=int(float(p.get("workers", 8))),
                rescore=bool(p.get("rescore")),
            )
            factories = [lambda: curate_generator(basic_cfg)]
            if p.get("smart"):
                smart_cfg = SmartCurateConfig(
                    output_base=out_base,
                    anchor_dir=Path(p["anchordir"]).expanduser() if p.get("anchordir") else None,
                    identity_threshold=float(p.get("idthr", 0.35)),
                    min_face_area=float(p.get("minface", 0.015)),
                )
                if p.get("buildanchor") and smart_cfg.anchor_dir:
                    factories.append(lambda: build_anchor(smart_cfg))
                factories.append(lambda: smart_curate_generator(smart_cfg))
            if p.get("cluster"):
                cluster_cfg = ClusterConfig(
                    output_base=out_base,
                    k=int(float(p.get("k", 0) or 0)),
                    batch_size=int(float(p.get("cbatch", 16))),
                )
                factories.append(lambda: cluster_generator(cluster_cfg))
            return _chain(*factories)

        if stage == "caption":
            cfg = CaptionConfig(
                output_base=out_base,
                trigger=str(p.get("trigger", prj.trigger_token)),
                class_word=str(p.get("cls", prj.class_word)),
                threshold=float(p.get("thr", 0.35)),
                max_tags=int(float(p.get("maxtags", 30))),
                blacklist=str(p.get("blacklist", "")),
                remap=str(p.get("remap", "")),
                prune=str(p.get("prune", "")),
                pony_prefix=bool(p.get("pony")),
                force=bool(p.get("force")),
            )
            return lambda: caption_generator(cfg)

        if stage == "build":
            from ..builder import BuildConfig, build_generator
            bcfg = BuildConfig(
                output_base=out_base,
                recipe=str(p.get("recipe", "")),
                note=str(p.get("note", "")),
            )
            return lambda: build_generator(prj, bcfg)

        if stage == "train":
            from ..train.kohya import TrainConfig, train_generator
            tcfg = TrainConfig(
                output_base=out_base,
                dataset_version=int(float(p.get("dataset", 0) or 0)),
                preset=str(p.get("preset", "character")),
                name=str(p.get("name", "")),
                dry_run=bool(p.get("dry")),
            )
            return lambda: train_generator(prj, tcfg)

        if stage == "sweep":
            from ..sweep import SweepConfig, sweep_generator
            scfg = SweepConfig(
                output_base=out_base, run=str(p.get("run", "")),
                backend=str(p.get("backend", "forge")),
                max_gap=float(p.get("gap", 0.15)),
                checkpoint=prj.base_model,
            )
            return lambda: sweep_generator(prj, scfg)

        if stage == "mega":
            from ..pipeline_dag import MegaConfig, mega_generator
            mgcfg = MegaConfig(output_base=out_base)
            return lambda: mega_generator(prj, mgcfg)

        if stage == "wizard":
            from ..wizard import WizardConfig, wizard_generator
            wcfg = WizardConfig(
                output_base=out_base,
                lora_type=str(p.get("type", "auto")),
                trigger=str(p.get("trigger", "")),
                class_word=str(p.get("cls", "")),
                name=str(p.get("name", "")),
                train=bool(p.get("train")),
                matrix=bool(p.get("matrix")),
                matrix_backend=str(p.get("backend", "forge")),
            )
            return lambda: wizard_generator(prj, wcfg)

        if stage == "merge":
            from ..eval.matrix import parse_lora_specs
            from ..merge import MergeConfig, merge_generator
            from ..util import safe_slug as _slug
            loras = parse_lora_specs(str(p.get("loras", "")).replace("\n", ","))
            blocks = {k: float(p.get(k, 1.0)) for k in ("te", "down", "mid", "up")}
            name = str(p.get("name", "merged"))
            mcfg = MergeConfig(
                output_base=out_base, loras=loras, output_name=name,
                default_blocks=blocks,
            )
            factories = [lambda: merge_generator(prj, mcfg)]
            if p.get("preview"):
                from ..eval.matrix import MatrixConfig, matrix_generator
                lora_dir = Path(prj.lora_output_dir
                                or (Path(prj.output_base) / "LORA_OUTPUT")).expanduser()
                merged_path = lora_dir / f"{_slug(name)}.safetensors"
                xcfg = MatrixConfig(
                    output_base=out_base, lora=str(merged_path),
                    label=f"{_slug(name)}_preview", categories=["likeness"],
                    backend=str(p.get("backend", "forge")),
                    checkpoint=prj.base_model,
                )
                factories.append(lambda: matrix_generator(prj, xcfg))
            return _chain(*factories)

        if stage == "matrix":
            from ..eval.matrix import CATEGORIES, MatrixConfig, matrix_generator
            cats = ([c for c in str(p.get("categories", "")).split(",") if c]
                    or list(CATEGORIES))
            mcfg = MatrixConfig(
                output_base=out_base, lora=str(p.get("lora", "")),
                categories=cats, backend=str(p.get("backend", "forge")),
                forge_url=str(p.get("url", "http://127.0.0.1:7860")),
                checkpoint=prj.base_model,
                lora_weight=float(p.get("weight", 0.85)),
                steps=int(float(p.get("steps", 28))),
            )
            return lambda: matrix_generator(prj, mcfg)

        if stage == "testgen":
            from ..eval.matrix import parse_lora_specs

            def _testgen():
                from ..util import now_iso as _now
                loras = parse_lora_specs(str(p.get("loras", "")).replace("\n", ","))
                backend = str(p.get("backend", "forge"))
                prompt = str(p.get("prompt", ""))
                seed = int(float(p.get("seed", 42)))
                out_dir = out_base / "evals" / "_single"
                out_dir.mkdir(parents=True, exist_ok=True)
                fpath = out_dir / f"gen_{_now().replace(':', '')}_s{seed}.png"
                yield f"Generating ({backend}) seed {seed}..."
                if backend == "forge":
                    from ..eval.forge_api import ForgeClient
                    client = ForgeClient(str(p.get("url", "http://127.0.0.1:7860")))
                    if not client.alive():
                        yield "FATAL: Forge API not reachable (start reForge with --api)"
                        return
                    init = str(p.get("init", "")).strip()
                    if init:
                        png = client.img2img(
                            Path(init).expanduser().read_bytes(), prompt=prompt,
                            negative=str(p.get("neg", "")),
                            strength=float(p.get("strength", 0.6)),
                            steps=int(float(p.get("steps", 28))),
                            cfg=float(p.get("cfg", 6.0)), seed=seed, loras=loras,
                        )
                    else:
                        png = client.txt2img(
                            prompt=prompt, negative=str(p.get("neg", "")),
                            steps=int(float(p.get("steps", 28))),
                            cfg=float(p.get("cfg", 6.0)),
                            width=int(float(p.get("w", 1024))),
                            height=int(float(p.get("h", 1024))),
                            seed=seed, loras=loras,
                        )
                    fpath.write_bytes(png)
                else:
                    from ..eval.pipeline import diffusers_available, get_pipeline
                    ok, reason = diffusers_available()
                    if not ok:
                        yield f"FATAL: {reason}"
                        return
                    tp = get_pipeline(prj.base_model, "", loras)
                    init = str(p.get("init", "")).strip()
                    if init:
                        from PIL import Image
                        im = Image.open(Path(init).expanduser()).convert("RGB")
                        img = tp.img2img(im, prompt, str(p.get("neg", "")),
                                         float(p.get("strength", 0.6)),
                                         int(float(p.get("steps", 28))),
                                         float(p.get("cfg", 6.0)), seed)
                    else:
                        img = tp.txt2img(prompt, str(p.get("neg", "")),
                                         int(float(p.get("steps", 28))),
                                         float(p.get("cfg", 6.0)),
                                         int(float(p.get("w", 1024))),
                                         int(float(p.get("h", 1024))), seed)
                    img.save(fpath)
                conn = manifest.connect(out_base)
                conn.execute(
                    "INSERT INTO evals(lora, label, category, prompt, seed, backend, "
                    "image_path, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (";".join(f"{l}:{w}" for l, w in loras) or "(none)", "_single",
                     "manual", prompt, seed, backend, str(fpath), _now()),
                )
                conn.commit()
                yield f"DONE -> {fpath}\nVisible in Eval Results after refresh."
            return _testgen

        if stage == "package":
            cfg = PackageConfig(
                output_base=out_base,
                token=str(p.get("token", prj.trigger_token)),
                class_word=str(p.get("cls", prj.class_word)),
                repeats=int(float(p.get("repeats", 10))),
                max_per_video=int(float(p.get("maxpv", 40))),
                max_total=int(float(p.get("maxtot", 0) or 0)),
                caption_text=str(p.get("caption", "")),
                write_captions=bool(p.get("captions")),
                quota=str(p.get("quota", DEFAULT_QUOTA)),
            )
            return lambda: package_generator(cfg)
        return None


def main_ui(project: Project, port: Optional[int] = None,
            project_path: Optional[str] = None) -> None:
    global FORGE_ROOT
    FORGE_ROOT = project.forge_root or None
    port = port or project.ui_port
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), StudioHandler)
    except OSError as exc:
        if getattr(exc, "errno", None) == 48 or "in use" in str(exc).lower():
            print(f"\nPort {port} is already in use - an older studio server "
                  f"is probably still running.\n"
                  f"  Stop it:   lsof -ti :{port} | xargs kill\n"
                  f"  Or run on another port:   lora-studio ui --port {port + 1}\n")
            return
        raise
    server.project = project  # type: ignore[attr-defined]
    server.project_path = project_path  # type: ignore[attr-defined]
    server.output_base = project.output_base  # type: ignore[attr-defined]
    url = f"http://127.0.0.1:{port}"
    print(f"\n  LoRA Designer Studio [{project.name}] -> {url}\n  (Ctrl+C to quit)\n")
    try:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
