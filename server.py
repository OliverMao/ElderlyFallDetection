"""
Web real-time preview server for the fall detection pipeline.

Streams HUD-rendered frames and per-track metadata to a browser over
WebSocket, with a small REST API to start/stop the pipeline.

Usage:
    python server.py                          # http://127.0.0.1:8000
    python server.py --port 8000 --source demo
    python server.py --source 0 --imgsz 480 --stride 2
"""
import argparse
import asyncio
import base64
import glob
import os
import sys
import threading
import time
import webbrowser
from typing import Any, Dict, Optional

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.config import load_config
from main import FallDetectionPipeline
from test_generator import generate_fall_test_video

app = FastAPI(title="Fall Detection Web Preview")

JPG_QUALITY = 85

# ---------------------------------------------------------------------------
# Stream session (one pipeline run at a time, worker thread + latest-frame slot)
# ---------------------------------------------------------------------------

class StreamSession:
    """Runs the pipeline in a worker thread and exposes the latest frame + metadata."""

    def __init__(self, cfg_dir: Optional[str] = None):
        self.cfg = load_config(cfg_dir)
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.pipeline: Optional[FallDetectionPipeline] = None
        self.source: Optional[str] = None
        self.started_at: Optional[float] = None
        self.seq = 0                          # monotonically increasing frame sequence
        self.latest: Optional[Dict[str, Any]] = None  # {"jpeg": b64, "meta": dict}
        self.stats: Dict[str, Any] = {"frames": 0, "fps": 0.0, "state": "IDLE", "detail": None}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- public API ---------------------------------------------------------

    def start(self, source: str, imgsz: int = 0, num_threads: int = 0, stride: int = 0) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError("a pipeline run is already in progress")
            self._stop_evt.clear()
            self.seq = 0
            self.latest = None
            self.pipeline = None
            self.source = source
            self.started_at = time.time()
            self.stats = {"frames": 0, "fps": 0.0, "state": "STARTING", "detail": None}
            self._thread = threading.Thread(
                target=self._run, args=(source, imgsz, num_threads, stride),
                name="pipeline-worker", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def state(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "source": self.source,
            "uptime_sec": round(time.time() - self.started_at, 1) if self.started_at else 0,
            **self.stats,
        }

    # -- worker -------------------------------------------------------------

    def _resolve_source(self, source: str) -> str:
        source = self.cfg.resolve_source(source)
        if str(source) == "demo":
            demo_file = "demo_fall_with_occlusion.mp4"
            if not os.path.exists(demo_file):
                generate_fall_test_video(demo_file)
            return demo_file
        return source

    def _run(self, source: str, imgsz: int, num_threads: int, stride: int) -> None:
        cap = None
        try:
            source = self._resolve_source(source)
            is_cam = str(source).isdigit()
            if is_cam:
                if sys.platform.startswith("win"):
                    cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(int(source))
            else:
                cap = cv2.VideoCapture(str(source))

            if not cap.isOpened():
                self.stats.update(state="ERROR", detail=f"could not open source '{source}'")
                return

            ret, frame = cap.read()
            if not ret or frame is None:
                self.stats.update(state="ERROR", detail=f"could not read frames from '{source}'")
                cap.release()
                return

            height, width = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or float(self.cfg.get("default_fps", section="app"))
            if fps <= 0 or fps > 120:
                fps = float(self.cfg.get("default_fps", section="app"))

            pipeline = FallDetectionPipeline(
                cfg=self.cfg, fps=fps, imgsz=imgsz, num_threads=num_threads
            )
            if imgsz < 0:
                pipeline.detector.imgsz = min(640, int(max(width, height) * 0.85))
            stride = max(1, int(stride or self.cfg.get("stride", section="app")))
            pipeline.load_zones_for_source(source)
            self.pipeline = pipeline

            self.stats.update(state="RUNNING", detail=f"{width}x{height} @ {fps:.1f} FPS, imgsz={pipeline.detector.imgsz}, stride={stride}")

            frame_count = 0
            last_detections = None
            ema_fps = 0.0
            t0 = time.time()

            while not self._stop_evt.is_set():
                frame_count += 1
                t_frame_start = time.time()

                # Stride inference: run the DNN every `stride`-th frame, reusing
                # the last detections for the skipped frames in between.
                run_dnn = (frame_count % stride == 0) or last_detections is None
                vis_frame, alerts = pipeline.process_frame(
                    frame,
                    detections=None if run_dnn else last_detections
                )
                if run_dnn:
                    last_detections = pipeline.detector._last_detections

                t_elapsed = time.time() - t_frame_start
                current_fps = 1.0 / max(t_elapsed, 1e-4)
                ema_fps = current_fps if ema_fps == 0 else ema_fps * 0.9 + current_fps * 0.1
                pipeline.fps = current_fps

                self._publish(vis_frame, alerts, frame_count, ema_fps)

                if frame_count % 60 == 0:
                    print(f"[web] {source}: {frame_count} frames @ {ema_fps:.1f} FPS")

                ret, frame = cap.read()
                if not ret or frame is None:
                    if is_cam:
                        self._stop_evt.set()
                        break
                    self.stats.update(state="ENDED", detail="video ended")
                    break

        except Exception as e:  # noqa: BLE001 - report any crash to the UI
            self.stats.update(state="ERROR", detail=f"{type(e).__name__}: {e}")
        finally:
            if cap is not None:
                cap.release()
            if not self.stats.get("state") in ("ERROR", "ENDED"):
                self.stats.update(state="STOPPED", detail="stopped by user")
            self.started_at = None

    def _publish(self, frame: Any, alerts: list, frame_count: int, ema_fps: float) -> None:
        meta: Dict[str, Any] = {
            "frame": frame_count,
            "fps": round(ema_fps, 1),
            "tracks": [],
            "alerts": []
        }
        if self.pipeline is not None:
            for t in self.pipeline.tracker.tracklets.values():
                feats = getattr(t, "last_features", None)
                meta["tracks"].append({
                    "id": t.track_id,
                    "state": t.state,
                    "torso_angle": round(feats["torso_angle"], 1) if feats else None,
                    "vertical_velocity": round(feats["vertical_velocity"], 2) if feats else None,
                    "in_occlusion_zone": bool(t.in_occlusion_zone),
                    "occ_zone_name": t.occlusion_zone_name
                })
        for a in alerts:
            meta["alerts"].append({
                "track_id": a["track_id"],
                "state": a["state"],
                "message": a.get("info") or f"Fall ({a['state']}) for Person #{a['track_id']}"
            })

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
        if not ok:
            return
        self.seq += 1
        self.stats.update(frames=frame_count, fps=round(ema_fps, 1))
        self.latest = {"seq": self.seq, "jpeg": base64.b64encode(buf.tobytes()).decode("ascii"), "meta": meta}


session = StreamSession()

# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------


class StartRequest(BaseModel):
    source: str
    imgsz: int = 0
    threads: int = 0
    stride: int = 0


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/state")
async def api_state():
    return session.state()


@app.get("/api/sources")
async def api_sources():
    videos = []
    for pattern in ("*.mp4", "*.avi", "*.mkv", "*.mov", os.path.join("sample_videos", "*.*")):
        for p in glob.glob(pattern):
            if p.lower().endswith((".mp4", ".avi", ".mkv", ".mov")) and os.path.getsize(p) > 1024:
                if os.path.basename(p) in ("demo_fall_with_occlusion.mp4", "fall_detection_output.mp4"):
                    continue
                if os.path.basename(p).startswith(("output_", "test", "live_")):
                    continue
                videos.append(p.replace(os.sep, "/"))
    return {"webcam": "0", "demo": "demo", "videos": sorted(set(videos))}


@app.post("/api/start")
async def api_start(req: StartRequest):
    try:
        session.start(req.source, imgsz=req.imgsz, num_threads=req.threads, stride=req.stride)
    except RuntimeError as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=409, content={"ok": False, "error": str(e)})
    return {"ok": True, "state": session.state()}


@app.post("/api/stop")
async def api_stop():
    session.stop()
    return {"ok": True, "state": session.state()}


# ---------------------------------------------------------------------------
# WebSocket frame stream
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    last_seq = 0
    last_state_ping = 0.0
    try:
        while True:
            snap = session.latest
            if snap is not None and snap["seq"] != last_seq:
                last_seq = snap["seq"]
                await ws.send_json({
                    "type": "frame",
                    "seq": snap["seq"],
                    "jpeg": snap["jpeg"],
                    "meta": snap["meta"]
                })
            now = time.time()
            if now - last_state_ping > 1.0:
                last_state_ping = now
                await ws.send_json({"type": "state", **session.state()})
            # Poll for client messages/disconnect without blocking the send loop.
            try:
                await asyncio.wait_for(ws.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, RuntimeError):
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global session
    parser = argparse.ArgumentParser(description="Fall Detection Web Preview Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--source", type=str, default=None,
                        help="auto-start the pipeline on a source (webcam index, video path, 'demo')")
    parser.add_argument("--imgsz", type=int, default=0, help="0 = use config value")
    parser.add_argument("--threads", type=int, default=0, help="0 = use config value / auto")
    parser.add_argument("--stride", type=int, default=0, help="0 = use config value")
    parser.add_argument("--config", type=str, default=None, help="config directory")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser automatically")
    args = parser.parse_args()

    session = StreamSession(args.config)
    url = f"http://{args.host}:{args.port}"
    print("=" * 60)
    print("  Fall Detection Web Preview")
    print(f"  {url}")
    print("=" * 60)

    if args.source:
        session.start(args.source, imgsz=args.imgsz, num_threads=args.threads, stride=args.stride)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
