"""
Occlusion Handling & Pose Imputation Engine
Recovers missing keypoints through temporal momentum, biomechanical bone priors,
and spatial occlusion zone tracking.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class OcclusionEngine:
    """
    Handles keypoint occlusion detection, topological bone imputation,
    and furniture occlusion zone tracking.
    """

    # COCO Keypoint connectivity graph for human kinematic skeleton
    SKELETON_PAIRS = [
        (0, 1), (0, 2), (1, 3), (2, 4),            # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # Arms & Upper Body
        (5, 11), (6, 12), (11, 12),                # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)     # Legs
    ]

    # Keypoint groupings
    UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    TORSO_INDICES = [5, 6, 11, 12]
    LOWER_BODY_INDICES = [13, 14, 15, 16]

    def __init__(self, conf_threshold: float = 0.25, max_extrapolation_frames: int = 15,
                 extrapolation_damping: float = 0.9, synthetic_conf: float = 0.45,
                 bone_prior_conf: float = 0.35, knee_offset_ratio: float = 0.30,
                 ankle_offset_ratio: float = 0.25, severe_max_visible: int = 5,
                 min_visible_for_partial: int = 5):
        """
        Args:
            conf_threshold: Landmark confidence below which a point is deemed occluded.
            max_extrapolation_frames: Max frames to extrapolate a lost joint before resetting.
            extrapolation_damping: Per-frame damping factor applied to extrapolation velocity.
            synthetic_conf: Confidence label assigned to temporally extrapolated joints.
            bone_prior_conf: Confidence label assigned to bone-prior synthesized joints.
            knee_offset_ratio: Knee offset below hip as a fraction of bbox height.
            ankle_offset_ratio: Ankle offset below knee as a fraction of bbox height.
            severe_max_visible: Fewer visible joints than this => severe occlusion.
            min_visible_for_partial: Visible joints at or above this => partial occlusion.
        """
        self.conf_threshold = conf_threshold
        self.max_extrapolation_frames = max_extrapolation_frames
        self.extrapolation_damping = float(extrapolation_damping)
        self.synthetic_conf = float(synthetic_conf)
        self.bone_prior_conf = float(bone_prior_conf)
        self.knee_offset_ratio = float(knee_offset_ratio)
        self.ankle_offset_ratio = float(ankle_offset_ratio)
        self.severe_max_visible = int(severe_max_visible)
        self.min_visible_for_partial = int(min_visible_for_partial)
        self.occlusion_zones: List[Dict[str, Any]] = []

    def add_occlusion_zone(self, name: str, polygon: List[Tuple[int, int]], zone_type: str = "furniture"):
        """
        Adds a polygon region (e.g. bed, table, sofa) where occlusions are expected.
        """
        self.occlusion_zones.append({
            "name": name,
            "polygon": np.array(polygon, dtype=np.int32),
            "type": zone_type
        })

    def process_keypoints(
        self,
        current_kps: np.ndarray,
        history_kps: List[np.ndarray],
        bbox: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Takes raw keypoints (17, 3), analyzes occlusion, and applies imputation.

        Returns:
            imputed_kps: np.ndarray of shape (17, 3) with imputed coordinates
            occlusion_mask: np.ndarray of bool shape (17,) True if occluded
            occlusion_stats: Dict with visibility percentages for upper/lower body
        """
        imputed_kps = np.copy(current_kps)
        occlusion_mask = np.zeros(17, dtype=bool)

        # 1. Identify raw occluded keypoints
        for i in range(17):
            conf = current_kps[i, 2]
            x, y = current_kps[i, 0], current_kps[i, 1]
            if conf < self.conf_threshold or (x == 0 and y == 0):
                occlusion_mask[i] = True

        # 2. Compute Visibility Metrics
        total_kps = 17
        visible_total = np.sum(~occlusion_mask)
        upper_vis = np.mean(~occlusion_mask[self.UPPER_BODY_INDICES])
        lower_vis = np.mean(~occlusion_mask[self.LOWER_BODY_INDICES])
        torso_vis = np.mean(~occlusion_mask[self.TORSO_INDICES])

        # 3. Impute Missing Joints via Temporal Kinematics & Bone Length Priors
        if len(history_kps) >= 2:
            prev_kps = history_kps[-1]
            prev_prev_kps = history_kps[-2]

            for i in range(17):
                if occlusion_mask[i]:
                    # Temporal Extrapolation (Velocity Continuity)
                    if prev_kps[i, 0] > 0 and prev_prev_kps[i, 0] > 0:
                        vx = prev_kps[i, 0] - prev_prev_kps[i, 0]
                        vy = prev_kps[i, 1] - prev_prev_kps[i, 1]
                        
                        # Apply damping factor to prevent runaway extrapolation
                        extrapolated_x = prev_kps[i, 0] + vx * self.extrapolation_damping
                        extrapolated_y = prev_kps[i, 1] + vy * self.extrapolation_damping
                        
                        imputed_kps[i, 0] = extrapolated_x
                        imputed_kps[i, 1] = extrapolated_y
                        imputed_kps[i, 2] = self.synthetic_conf  # Mark as synthetic confidence

        # 4. Bone Constraint Validation for Missing Lower Limbs
        # If knees or ankles are missing but hips are visible, synthesize approximate neutral orientation
        if occlusion_mask[13] and not occlusion_mask[11]:  # Left Knee missing, Left Hip visible
            imputed_kps[13, 0] = imputed_kps[11, 0]
            imputed_kps[13, 1] = imputed_kps[11, 1] + (bbox[3] - bbox[1]) * self.knee_offset_ratio
            imputed_kps[13, 2] = self.bone_prior_conf
            
        if occlusion_mask[14] and not occlusion_mask[12]:  # Right Knee missing, Right Hip visible
            imputed_kps[14, 0] = imputed_kps[12, 0]
            imputed_kps[14, 1] = imputed_kps[12, 1] + (bbox[3] - bbox[1]) * self.knee_offset_ratio
            imputed_kps[14, 2] = self.bone_prior_conf

        if occlusion_mask[15] and imputed_kps[13, 1] > 0:  # Left Ankle
            imputed_kps[15, 0] = imputed_kps[13, 0]
            imputed_kps[15, 1] = imputed_kps[13, 1] + (bbox[3] - bbox[1]) * self.ankle_offset_ratio
            imputed_kps[15, 2] = self.bone_prior_conf

        if occlusion_mask[16] and imputed_kps[14, 1] > 0:  # Right Ankle
            imputed_kps[16, 0] = imputed_kps[14, 0]
            imputed_kps[16, 1] = imputed_kps[14, 1] + (bbox[3] - bbox[1]) * self.ankle_offset_ratio
            imputed_kps[16, 2] = self.bone_prior_conf

        stats = {
            "total_visibility": float(visible_total / total_kps),
            "upper_body_visibility": float(upper_vis),
            "lower_body_visibility": float(lower_vis),
            "torso_visibility": float(torso_vis),
            "is_partially_occluded": bool(visible_total < total_kps and visible_total >= self.min_visible_for_partial),
            "is_severely_occluded": bool(visible_total < self.severe_max_visible)
        }

        return imputed_kps, occlusion_mask, stats

    def is_in_occlusion_zone(self, point: Tuple[float, float]) -> Tuple[bool, Optional[str]]:
        """
        Checks if a given coordinate (e.g. person centroid or feet) is inside any defined occlusion ROI.
        """
        import cv2
        pt = (int(point[0]), int(point[1]))
        for zone in self.occlusion_zones:
            # pointPolygonTest returns >= 0 if point is inside or on edge
            dist = cv2.pointPolygonTest(zone["polygon"], pt, measureDist=False)
            if dist >= 0:
                return True, zone["name"]
        return False, None
