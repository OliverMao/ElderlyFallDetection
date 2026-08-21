"""
Pose Detection Module using YOLO26-Pose
Extracts 17 COCO Keypoints per detected person in real-time.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from ultralytics import YOLO


class PoseDetector:
    """
    Wraps YOLO26-Pose model for efficient single/multi-person landmark estimation.
    """
    
    COCO_KEYPOINTS = [
        "nose",          # 0
        "left_eye",      # 1
        "right_eye",     # 2
        "left_ear",      # 3
        "right_ear",     # 4
        "left_shoulder", # 5
        "right_shoulder",# 6
        "left_elbow",    # 7
        "right_elbow",   # 8
        "left_wrist",    # 9
        "right_wrist",   # 10
        "left_hip",      # 11
        "right_hip",     # 12
        "left_knee",     # 13
        "right_knee",    # 14
        "left_ankle",    # 15
        "right_ankle"    # 16
    ]

    def __init__(self, model_name: str = "yolo26n-pose.pt", conf_thresh: float = 0.35, device: Optional[str] = None, imgsz: int = 640, num_threads: Optional[int] = None,
                 min_visible_joints: int = 4, keypoint_conf: float = 0.35,
                 min_bbox_w: int = 20, min_bbox_h: int = 30):
        """
        Args:
            model_name: YOLO pose model path/name (default: lightweight yolo26n-pose.pt)
            conf_thresh: Minimum detection confidence threshold
            device: 'cpu', 'cuda', or None for auto-detection
            imgsz: Inference input size (int, square). Lower values speed up CPU inference
                   substantially (e.g. 320 >> 640). Set 0 to auto-ease on source size.
            num_threads: Torch CPU threads. On small pose models, fewer threads (1-4) often
                         beat the default all-core setting due to thread contention.
            min_visible_joints: Minimum number of confident joints for a detection to be kept
            keypoint_conf: Minimum per-joint confidence counted as "visible"
            min_bbox_w / min_bbox_h: Minimum bounding-box size (pixels) to accept a detection
        """
        self.conf_thresh = conf_thresh
        self.imgsz = int(imgsz) if imgsz else 640

        # Detection filter parameters (configurable via configs/model_config.yaml)
        self.min_visible_joints = int(min_visible_joints)
        self.keypoint_conf = float(keypoint_conf)
        self.min_bbox_w = int(min_bbox_w)
        self.min_bbox_h = int(min_bbox_h)
        
        # Expose an override hook so the pipeline can adapt imgsz at runtime
        self._pipeline_imgsz: Optional[int] = None
        self._last_detections: List[Dict[str, Any]] = []

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Optimize CPU thread count (configurable). Default: cap below core count to avoid
        # contention that hurts small-model latency.
        if num_threads is not None:
            self.num_threads = None if num_threads <= 0 else num_threads
        elif self.device == "cpu":
            cores = torch.get_num_threads()
            self.num_threads = min(4, max(1, cores))
        else:
            self.num_threads = None

        if self.num_threads is not None and self.device == "cpu":
            torch.set_num_threads(self.num_threads)

        print(f"[PoseDetector] Loading {model_name} on device: {self.device}")
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs pose estimation on a single frame.

        Returns list of dicts:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "det_conf": float,
                "keypoints": np.ndarray of shape (17, 3) where columns are [x, y, confidence]
            },
            ...
        ]
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            imgsz=self._pipeline_imgsz if self._pipeline_imgsz else self.imgsz,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            self._last_detections = detections
            return detections

        result = results[0]
        
        if result.boxes is None or result.keypoints is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        
        # Keypoints data shape: (N, 17, 3) -> [x, y, conf] or (N, 17, 2)
        kps_data = result.keypoints.data.cpu().numpy()

        for i in range(len(boxes)):
            bbox = boxes[i].tolist()
            det_conf = float(confs[i])
            kps = kps_data[i]  # shape (17, 3) or (17, 2)

            if kps.shape[-1] == 2:
                # If confidence column missing, synthesize full confidence for visible points
                kps_with_conf = np.zeros((17, 3), dtype=np.float32)
                kps_with_conf[:, :2] = kps
                kps_with_conf[:, 2] = np.where((kps[:, 0] > 0) | (kps[:, 1] > 0), det_conf, 0.0)
                kps = kps_with_conf

            # Filter out spurious false positives with fewer than the required visible joints
            confident_joints = np.sum((kps[:, 2] > self.keypoint_conf) & (kps[:, 0] > 0) & (kps[:, 1] > 0))
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]

            # Reject tiny or low-joint noise
            if confident_joints < self.min_visible_joints or bbox_w < self.min_bbox_w or bbox_h < self.min_bbox_h:
                continue

            detections.append({
                "bbox": bbox,
                "det_conf": det_conf,
                "keypoints": kps
            })

        self._last_detections = detections
        return detections
