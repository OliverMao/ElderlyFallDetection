"""
Main Pipeline Runner for AI Fall Detection & Occlusion Resilience System
Executes real-time pose estimation, occlusion recovery, kinematic tracking,
and state machine fall verification.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

from core.detector import PoseDetector
from core.tracker import SimpleTracker, Tracklet
from core.occlusion import OcclusionEngine
from core.kinematics import KinematicExtractor
from core.state_machine import FallStateMachine
from core.visualizer import Visualizer
from core.config import Config, load_config
from test_generator import generate_fall_test_video


class FallDetectionPipeline:
    """
    Main orchestration class that coordinates all modules.
    All thresholds and model parameters are taken from a Config object
    (see core/config.py and the configs/ directory).
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        fps: float = 30.0,
        conf: Optional[float] = None,
        device: Optional[str] = None,
        imgsz: int = 0,
        num_threads: int = 0
    ):
        cfg = cfg or load_config()
        self.cfg = cfg
        self.fps = fps

        m = cfg.section("pose_model")
        self.detector = PoseDetector(
            model_name=m["name"],
            conf_thresh=conf if conf is not None else m["conf_thresh"],
            device=device if device is not None else m["device"],
            imgsz=imgsz or m["imgsz"],
            num_threads=num_threads if num_threads > 0 else m["num_threads"],
            min_visible_joints=m["min_visible_joints"],
            keypoint_conf=m["keypoint_conf"],
            min_bbox_w=m["min_bbox_w"],
            min_bbox_h=m["min_bbox_h"]
        )

        t = cfg.section("tracker")
        self.tracker = SimpleTracker(
            iou_thresh=t["iou_thresh"],
            max_missing=t["max_missing"],
            max_history=t["max_history"]
        )

        o = cfg.section("occlusion")
        self.occlusion_engine = OcclusionEngine(
            conf_threshold=o["conf_threshold"],
            max_extrapolation_frames=o["max_extrapolation_frames"],
            extrapolation_damping=o["extrapolation_damping"],
            synthetic_conf=o["synthetic_conf"],
            bone_prior_conf=o["bone_prior_conf"],
            knee_offset_ratio=o["knee_offset_ratio"],
            ankle_offset_ratio=o["ankle_offset_ratio"],
            severe_max_visible=o["severe_max_visible"],
            min_visible_for_partial=o["min_visible_for_partial"]
        )

        k = cfg.section("kinematics")
        self.kinematics = KinematicExtractor(fps=fps, stillness_window=k["stillness_window"])

        s = cfg.section("state_machine")
        self.state_machine = FallStateMachine(
            fps=fps,
            fall_velocity_thresh=s["fall_velocity_thresh"],
            angular_collapse_thresh=s["angular_collapse_thresh"],
            slow_fall_velocity=s["slow_fall_velocity"],
            occlusion_zone_velocity=s["occlusion_zone_velocity"],
            flat_angle_thresh=s["flat_angle_thresh"],
            standing_angle_thresh=s["standing_angle_thresh"],
            inactivity_sec=s["inactivity_sec"],
            pre_alert_sec=s["pre_alert_sec"],
            stillness_energy_thresh=s["stillness_energy_thresh"],
            flat_aspect_ratio=s["flat_aspect_ratio"],
            stand_up_velocity=s["stand_up_velocity"],
            recovery_motion_energy=s["recovery_motion_energy"]
        )

        a = cfg.section("app")
        self.window_name = a.get("window_name", "AI Fall Detection & Occlusion Resilience")
        self.window_max_width = int(a.get("window_max_width", 1024))

        self.visualizer = Visualizer()

    def add_furniture_zone(self, name: str, polygon: List[tuple]):
        self.occlusion_engine.add_occlusion_zone(name, polygon)

    def load_zones_for_source(self, source: str):
        """
        (Re)load occlusion zones from configs/zones_config.json that match the
        given input source. Zones without a "source" field apply to all sources.
        """
        self.occlusion_engine.occlusion_zones.clear()
        for zone in self.cfg.zones_for_source(source):
            self.occlusion_engine.add_occlusion_zone(
                zone["name"], zone["polygon"], zone.get("type", "furniture")
            )

    def process_frame(self, frame: np.ndarray, detections: Optional[List[Dict[str, Any]]] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Runs full pipeline on a single frame.

        Args:
            frame: BGR image (H, W, 3)
            detections: Optional pre-computed pose detections. When provided (e.g. reuse
                        between DNN frames under stride inference), skips the DNN call.

        Returns:
            rendered_frame: np.ndarray
            alerts: List of active emergency alerts
        """
        # Step 1: Detect Persons & Keypoints (skip DNN when detections are reused)
        if detections is None:
            detections = self.detector.detect(frame)

        # Step 2: Update Multi-Person Tracker
        tracklets = self.tracker.update(detections)

        tracklet_data: Dict[int, Dict[str, Any]] = {}
        alerts: List[Dict[str, Any]] = []

        for track in tracklets:
            t_id = track.track_id
            curr_bbox = track.bbox_history[-1]
            raw_kps = track.raw_kps_history[-1]

            # Step 3: Occlusion Analysis & Keypoint Imputation
            imputed_kps, occ_mask, occ_stats = self.occlusion_engine.process_keypoints(
                current_kps=raw_kps,
                history_kps=track.imputed_kps_history,
                bbox=curr_bbox
            )
            # Update tracklet imputed history
            track.imputed_kps_history[-1] = imputed_kps

            # Check if person intersects an occlusion zone
            centroid = (
                (curr_bbox[0] + curr_bbox[2]) / 2.0,
                (curr_bbox[1] + curr_bbox[3]) / 2.0
            )
            in_zone, zone_name = self.occlusion_engine.is_in_occlusion_zone(centroid)
            track.in_occlusion_zone = in_zone
            track.occlusion_zone_name = zone_name

            # Step 4: Compute Kinematic Features
            features = self.kinematics.extract_features(
                keypoints_history=track.imputed_kps_history,
                bbox_history=track.bbox_history,
                occlusion_stats=occ_stats
            )

            # Step 5: Advance State Machine
            state, state_info = self.state_machine.step(track, features, occ_stats)

            tracklet_data[t_id] = {
                "imputed_kps": imputed_kps,
                "occlusion_mask": occ_mask,
                "occlusion_stats": occ_stats,
                "features": features,
                "state": state,
                "state_info": state_info
            }

            if state in [FallStateMachine.STATE_FALL_DETECTED, FallStateMachine.STATE_IMPACT, FallStateMachine.STATE_CONFIRMED_FALL]:
                alerts.append({
                    "track_id": t_id,
                    "state": state,
                    "info": state_info,
                    "features": features
                })

        # Step 6: Render HUD Visualizer
        rendered_frame = self.visualizer.render(
            frame=frame,
            tracklets=tracklets,
            tracklet_data=tracklet_data,
            occlusion_zones=self.occlusion_engine.occlusion_zones,
            fps_val=self.fps
        )

        return rendered_frame, alerts


def run_pipeline(
    source: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    conf: Optional[float] = None,
    device: Optional[str] = None,
    imgsz: int = 0,
    num_threads: int = 0,
    stride: int = 0,
    config_dir: Optional[str] = None
):
    """
    Args:
        source: video file path, webcam index (str digit), 'demo', or a name
                defined in app_config.yaml `app.sources`. None = config default.
        save_path: output video path. None = config default.
        conf: pose detection confidence. None = config value.
        imgsz: DNN inference size. 0 = use config value; negative = auto-ease
               on the source resolution.
        num_threads: torch CPU threads. 0 = use config value.
        stride: run the DNN every `stride`-th frame and reuse the last
                detections in between. 0 = use config value; 1 = every frame.
        config_dir: directory with app_config.yaml / model_config.yaml /
                    zones_config.json. None = <project>/configs.
    """
    cfg = load_config(config_dir)
    app_cfg = cfg.section("app")

    print("=" * 70)
    print("  REAL-TIME PERSON & ELDERLY FALL DETECTION SYSTEM")
    print("  With Advanced Occlusion Resilience & Multi-Stage State Machine")
    print("=" * 70)
    print(f"[Pipeline] Configuration: {cfg.config_dir}")

    if source is None:
        source = app_cfg.get("default_source", "demo")
    if save_path is None:
        save_path = app_cfg.get("output_save_path", "fall_detection_output.mp4")
    if stride <= 0:
        stride = int(app_cfg.get("stride", 1))

    # Resolve a named source (app.sources) to its URL, if applicable
    source = cfg.resolve_source(source)

    if str(source) == "demo":
        demo_file = "demo_fall_with_occlusion.mp4"
        if not os.path.exists(demo_file):
            print("[Pipeline] Generating synthetic test video with occlusion barrier...")
            generate_fall_test_video(demo_file)
        source = demo_file

    is_cam = str(source).isdigit()
    if is_cam:
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        print(f"[Error] Could not open video source: '{source}'. Check file path or camera permissions.")
        return

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        print(f"[Error] Could not read frames from video source: '{source}'")
        cap.release()
        return

    height, width = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or float(app_cfg.get("default_fps", 30.0))
    if fps <= 0 or fps > 120:
        fps = float(app_cfg.get("default_fps", 30.0))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0

    print(f"[Pipeline] Initialized Source: {source} ({width}x{height} @ {fps:.1f} FPS)")

    # Video Writer
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*str(app_cfg.get("fourcc", "mp4v")))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"[Pipeline] Output will be saved to: {save_path}")

    pipeline = FallDetectionPipeline(cfg=cfg, conf=conf, device=device, fps=fps, imgsz=imgsz, num_threads=num_threads)

    # Auto-ease imgsz to the scale of the source when explicitly requested (< 0)
    if imgsz < 0:
        pipeline.detector.imgsz = min(640, int(max(width, height) * 0.85))
    imgsz = pipeline.detector.imgsz
    stride = max(1, int(stride))
    print(f"[Pipeline] Inference imgsz={imgsz}, stride={stride}, torch_threads={pipeline.detector.num_threads}")

    # Load occlusion zones registered for this source (configs/zones_config.json)
    pipeline.load_zones_for_source(source)
    if pipeline.occlusion_engine.occlusion_zones:
        zone_names = ", ".join(z["name"] for z in pipeline.occlusion_engine.occlusion_zones)
        print(f"[Pipeline] Active occlusion zones: {zone_names}")

    frame_count = 0
    t_start = time.time()
    window_name = pipeline.window_name

    if show:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            display_w = min(width, pipeline.window_max_width)
            display_h = int(height * (display_w / max(width, 1)))
            cv2.resizeWindow(window_name, display_w, display_h)
        except Exception:
            pass

    try:
        frame = first_frame
        last_detections = None   # cached DNN results for reuse during skipped frames
        while True:
            if frame is None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

            frame_count += 1
            t_frame_start = time.time()

            # Stride inference: run the DNN every `stride`-th frame, reusing the last
            # detections (and thus last keypoints) for the skipped frames in between.
            run_dnn = (frame_count % stride == 0) or last_detections is None
            vis_frame, alerts = pipeline.process_frame(
                frame,
                detections=None if run_dnn else last_detections
            )
            if run_dnn:
                last_detections = pipeline.detector._last_detections

            t_elapsed = time.time() - t_frame_start
            current_fps = 1.0 / max(t_elapsed, 1e-4)
            pipeline.fps = current_fps

            # Log any active emergency alerts
            for alert in alerts:
                if alert["state"] in ["FALL_DETECTED", "IMPACT", "CONFIRMED_FALL"]:
                    print(f"🚨 [EMERGENCY ALERT] Frame {frame_count}: Fall detected for Person #{alert['track_id']}! "
                          f"(Torso Angle: {alert['features']['torso_angle']:.1f}°)")

            if writer:
                writer.write(vis_frame)

            if show:
                cv2.imshow(window_name, vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    print("[Pipeline] User requested exit.")
                    break

            if frame_count % 30 == 0 and total_frames > 0:
                print(f"[Pipeline] Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

            frame = None

    finally:
        cap.release()
        if writer:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    total_time = time.time() - t_start
    avg_fps = frame_count / max(total_time, 1e-4)
    print("=" * 70)
    print(f"  Execution Complete! Processed {frame_count} frames in {total_time:.2f}s (Avg FPS: {avg_fps:.1f})")
    if save_path:
        print(f"  Processed video saved at: {os.path.abspath(save_path)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Fall Detection & Occlusion System")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to video file, webcam index (0), 'demo', or a named source from app_config.yaml "
                             "(default: from config)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save output video (default: from config)")
    parser.add_argument("--show", action="store_true", default=False, help="Display live OpenCV window")
    parser.add_argument("--conf", type=float, default=None,
                        help="Pose detection confidence threshold (default: from config)")
    parser.add_argument("--device", type=str, default=None, help="'cpu', 'cuda', or None for auto")
    parser.add_argument("--imgsz", type=int, default=0,
                        help="DNN inference size (320/480/640); 0 = use config value; negative = auto on source size")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = use config value / auto)")
    parser.add_argument("--stride", type=int, default=0, help="Run DNN every N-th frame (0 = use config value; 1 = off)")
    parser.add_argument("--config", type=str, default=None,
                        help="Directory with config files (default: <project>/configs)")

    args = parser.parse_args()
    run_pipeline(source=args.source, save_path=args.save, show=args.show, conf=args.conf, device=args.device,
                 imgsz=args.imgsz, num_threads=args.threads, stride=args.stride, config_dir=args.config)
