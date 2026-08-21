"""
Configuration Management Module

Loads and merges the configuration files under configs/:
    app_config.yaml     pipeline parameters and per-module thresholds
    model_config.yaml   pose model weights and inference parameters
    zones_config.json   occlusion zone (furniture) polygons

Precedence: built-in defaults < config files < runtime/CLI overrides.
Missing files or missing keys fall back to the built-in defaults, so the
pipeline keeps working even if the configs/ directory is absent.
"""

import os
import json
import fnmatch
from copy import deepcopy
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

# Built-in defaults: guarantee the pipeline runs even if a config file or key is missing.
# These mirror the values documented in DESIGN.md.
DEFAULTS: Dict[str, Any] = {
    "app": {
        "default_source": "demo",
        "sources": [],
        "output_save_path": "fall_detection_output.mp4",
        "fourcc": "mp4v",
        "default_fps": 30.0,
        "stride": 1,
        "window_name": "AI Fall Detection & Occlusion Resilience",
        "window_max_width": 1024,
    },
    "pose_model": {
        "name": "yolo26n-pose.pt",
        "conf_thresh": 0.35,
        "device": None,
        "imgsz": 640,
        "num_threads": 0,
        "min_visible_joints": 4,
        "keypoint_conf": 0.35,
        "min_bbox_w": 20,
        "min_bbox_h": 30,
    },
    "tracker": {
        "iou_thresh": 0.25,
        "max_missing": 20,
        "max_history": 60,
    },
    "occlusion": {
        "conf_threshold": 0.25,
        "max_extrapolation_frames": 15,
        "extrapolation_damping": 0.9,
        "synthetic_conf": 0.45,
        "bone_prior_conf": 0.35,
        "knee_offset_ratio": 0.30,
        "ankle_offset_ratio": 0.25,
        "severe_max_visible": 5,
        "min_visible_for_partial": 5,
    },
    "kinematics": {
        "stillness_window": 15,
    },
    "state_machine": {
        "fall_velocity_thresh": 1.2,
        "angular_collapse_thresh": 35.0,
        "slow_fall_velocity": 0.6,
        "occlusion_zone_velocity": 0.8,
        "flat_angle_thresh": 55.0,
        "standing_angle_thresh": 70.0,
        "inactivity_sec": 4.0,
        "pre_alert_sec": 5.0,
        "stillness_energy_thresh": 25.0,
        "flat_aspect_ratio": 0.85,
        "stand_up_velocity": -0.3,
        "recovery_motion_energy": 20.0,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` on top of `base` and return a new dict."""
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class Config:
    """Config container with section lookup, default fallback and zone matching."""

    def __init__(self, data: Dict[str, Any], zones: Optional[List[Dict[str, Any]]] = None):
        self.data: Dict[str, Any] = data
        self.zones: List[Dict[str, Any]] = zones or []
        self.config_dir: Optional[str] = None

    def section(self, name: str) -> Dict[str, Any]:
        """Return the full config section as a dict (empty dict if absent)."""
        return self.data.get(name, {})

    def get(self, key: str, section: str = "app", default: Any = None) -> Any:
        """Return a single value, e.g. cfg.get("imgsz", section="pose_model")."""
        return self.data.get(section, {}).get(key, default)

    def zones_for_source(self, source: str) -> List[Dict[str, Any]]:
        """Return occlusion zones active for the given input source.

        A zone applies when it has no "source" field, or when the source
        string matches its "source" pattern (fnmatch syntax, e.g. "demo*").
        """
        result = []
        for zone in self.zones:
            pattern = zone.get("source")
            if pattern is None or fnmatch.fnmatch(str(source), str(pattern)):
                result.append(zone)
        return result

    def resolve_source(self, source: str) -> str:
        """Resolve a source name against the configured `app.sources` list.

        Entries like {name: front_room, url: rtsp://...} allow referencing
        cameras by short name on the command line.
        """
        for src in self.data.get("app", {}).get("sources", []) or []:
            if isinstance(src, dict) and src.get("name") == source:
                return src["url"]
        return source

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self.data)


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    if yaml is None:
        print(f"[Config] Warning: PyYAML is not installed; skipping {os.path.basename(path)} "
              f"(built-in defaults will be used)")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_config(config_dir: Optional[str] = None) -> Config:
    """
    Load the configuration stack:
        built-in defaults < app_config.yaml < model_config.yaml
    plus the occlusion zone polygons from zones_config.json.

    Args:
        config_dir: directory containing the config files.
                    Defaults to <project root>/configs.
    """
    config_dir = config_dir or DEFAULT_CONFIG_DIR

    data = deepcopy(DEFAULTS)
    data = _deep_merge(data, _load_yaml(os.path.join(config_dir, "app_config.yaml")))
    data = _deep_merge(data, _load_yaml(os.path.join(config_dir, "model_config.yaml")))

    zones_data = _load_json(os.path.join(config_dir, "zones_config.json"))
    zones = zones_data.get("zones", []) if isinstance(zones_data, dict) else []
    if not isinstance(zones, list):
        zones = []

    cfg = Config(data, zones)
    cfg.config_dir = config_dir
    return cfg
