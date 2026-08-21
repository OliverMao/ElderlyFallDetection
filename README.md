# Real-Time Person & Elderly Fall Detection System

> **中文版本 / Chinese version**: [README.zh-CN.md](README.zh-CN.md)

An edge-friendly real-time fall detection system built on pose estimation. It detects falls even under severe occlusion (behind furniture, tables, beds, and sofas) and uses a multi-stage state machine to distinguish falls from activities of daily living (sitting down, tying shoelaces, resting in bed), keeping false alarms low. It extracts 17 COCO keypoints via [Ultralytics YOLO26-Pose](https://github.com/ultralytics/ultralytics) and runs on plain CPU without a dedicated GPU.

---

## Key Features

- **Instant trigger detection**: Triggers alerts the moment a rapid descent (`v_y > 0.7 h/s`) or a fast posture collapse (`Δθ/Δt > 35°/s`) is detected.
- **False-alarm rejection**: Distinguishes accidental falls from horizontal resting/sleeping in bed to avoid false alarms from daily activities.
- **4-tier occlusion resilience**:
  1. Bone-constraint imputation: reconstructs occluded knees and ankles from anatomical ratios.
  2. Temporal velocity extrapolation carries motion vectors through occlusion.
  3. Furniture-ROI tracking follows descents behind known barriers.
  4. Proximity & bone-length clamping removes spurious skeleton lines.
- **Lightweight model**: About 6.5 MB pose model; runs on a normal CPU with no custom training.
- **Interactive runner**: CLI menu with auto-discovery of videos and drag-and-drop support.

---

## Quickstart

One command:

```bash
python run.py
```

On Windows you can also double-click `run.bat`.

### Interactive menu

```text
======================================================================
  AI PERSON & ELDERLY FALL DETECTION SYSTEM
  Advanced Occlusion Resilience & Kinematic Tracking
======================================================================
  Select an input option:
   [1] Live Webcam (Real-time monitoring)
   [2] Custom Video File (Auto-detects videos + Drag & Drop)
   [3] Synthetic Occlusion Fall Demo (Built-in test)
   [4] Exit
======================================================================
```

- Option 1 (Webcam): Live camera feed with real-time skeleton and telemetry overlay.
- Option 2 (Video file): Lists available test videos for numeric selection, or accepts a dragged `.mp4` / `.avi` file.
- Option 3 (Synthetic demo): Generates and plays a synthetic fall demo with an occlusion barrier.
- Option 4 (Exit): Closes the application.

---

## Benchmark videos (`sample_videos/`)

The repo includes these benchmark fall clips:

| # | File | Scenario & Purpose | Expected |
|---|---|---|---|
| 1 | `standing_fall_1.mp4` | Sudden collapse from standing | `[ALERT] FALL DETECTED` |
| 2 | `standing_fall_2.mp4` | Slip & backward fall | `[ALERT] FALL DETECTED` |
| 3 | `sitting_to_fall.mp4` | Senior falling off a chair | `[ALERT] FALL DETECTED` |
| 4 | `bed_rollout_fall.mp4` | Roll-out fall from bed | `[ALERT] FALL DETECTED` |
| 5 | `normal_activity_no_fall.mp4` | Walking, sitting, sleeping (ADL baseline) | No false alarm |

---

## Architecture & Pipeline

```
  ┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
  │ Video / Webcam  │ ──> │ YOLO26-Pose Backbone │ ──> │  Occlusion Resilience  │
  │     Stream      │     │  Keypoint Extractor  │     │   & Bone Imputation    │
  └─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                    │
  ┌─────────────────┐     ┌──────────────────────┐                  ▼
  │ HUD Visualizer  │ <── │ Multi-Stage Fall FSM │ <── ┌────────────────────────┐
  │  & Alert Engine │     │  State Transitions   │     │  Kinematic Extractor   │
  └─────────────────┘     └──────────────────────┘     │ (v_y, Angle, ω, Energy)│
                                                       └────────────────────────┘
```

### Biomechanical features

- **Torso angle (`θ_torso`)**: Angle of the mid-shoulder–mid-hip line vs. the floor (`90°` standing, `0°` flat).
- **Vertical descent rate (`v_y`)**: Normalized downward velocity in body-box heights per second.
- **Angular collapse rate (`ω = dθ/dt`)**: Rate of postural angle drop in degrees per second.
- **Stillness energy (`E_motion`)**: Temporal variance of visible keypoints used to confirm post-impact immobility.

### State machine

```
 [ MONITORING ] ──( v_y > 0.7 h/s OR ω > 35°/s )──> [ FALL_DETECTED ]
       ▲                                                     │
       │                                            ( > 1.5s Stillness )
  ( Stands Up )                                              │
       │                                                     ▼
 [ MONITORING ] <──────( Upright Motion )─────── [ CONFIRMED_FALL ]
```

---

## Repository layout

```
├── run.bat                          # One-click launcher
├── run.py                           # Interactive CLI runner
├── main.py                          # CLI entry point (run_pipeline)
├── configs/
│   ├── app_config.yaml              # Pipeline params & module thresholds
│   ├── model_config.yaml            # Pose model weights & inference params
│   └── zones_config.json            # Furniture occlusion zone polygons
├── core/
│   ├── config.py                    # Config loader & default fallback
│   ├── detector.py                  # Pose wrapper & keypoints
│   ├── tracker.py                   # Multi-person tracklet manager
│   ├── occlusion.py                 # Occlusion & imputation
│   ├── kinematics.py                # Kinematic feature engine
│   ├── state_machine.py             # Multi-stage fall state machine
│   └── visualizer.py                # Telemetry HUD & alert banners
├── sample_videos/                   # Benchmark test videos
├── test_generator.py                # Synthetic demo generator
├── requirements.txt                 # Dependencies
├── DESIGN.md                        # Design specification (local only, not tracked in git)
└── README.md                        # This document
```

---

## Direct CLI usage

Besides the menu, you can run directly with CLI arguments:

```bash
# Webcam (device 0)
python main.py --source 0 --show

# A specific video file
python main.py --source sample_videos/standing_fall_1.mp4 --show --save output_result.mp4

# Synthetic occlusion demo
python main.py --source demo --show
```

---

## Configuration

All thresholds and pipeline parameters live in `configs/` (no hard-coded magic numbers in the detection logic):

| File | Scope |
|---|---|
| `app_config.yaml` | Pipeline defaults, tracker, occlusion engine, kinematics, fall state machine thresholds, named camera sources |
| `model_config.yaml` | Pose model weights, confidence, `imgsz`, CPU threads, detection filter |
| `zones_config.json` | Furniture occlusion zone polygons (pixel coordinates) |

Precedence: **built-in defaults < config files < CLI arguments**. Missing files or keys fall back to built-in defaults, so the pipeline keeps working without `configs/`.

```bash
# Use an alternate config directory (e.g. per-camera tuning)
python main.py --source 0 --show --config configs_room2
```

Typical tuning knobs:

```yaml
# configs/app_config.yaml
state_machine:
  fall_velocity_thresh: 1.2   # rapid descent trigger (h/s)
  inactivity_sec: 4.0         # seconds prone before CONFIRMED_FALL
  flat_angle_thresh: 55.0     # torso angle considered "flat" (deg)

# configs/zones_config.json  - register a bed/sofa for real deployments
"zones": [{ "name": "Bed", "type": "furniture", "polygon": [[...]] }]
```

Named camera sources can be referenced by short name:

```yaml
# configs/app_config.yaml
app:
  sources:
    - name: "front_room"
      url: "rtsp://192.168.1.100:554/stream1"
```

```bash
python main.py --source front_room --show
```

---

## CPU performance tuning

The pose model runs on CPU by default. Inference dominates; tune these options to trade a little accuracy/latency for throughput:

```bash
# Balanced: smaller input (~1.3-1.5x)
python main.py --source video.mp4 --imgsz 480

# Fast: DNN every 2nd frame (~2x more)
python main.py --source video.mp4 --imgsz 480 --stride 2 --threads 4

# Limit threads to avoid contention
python main.py --source video.mp4 --imgsz 640 --threads 4
```

| Option | Default | Description |
|---|---|---|
| `--imgsz` | `0` (config: `640`) | Input size; smaller is faster but reduces accuracy on tiny subjects. Negative = auto on source size |
| `--stride` | `0` (config: `1`) | Run DNN every N frames and reuse keypoints in between |
| `--threads` | `0` (config: auto) | CPU threads; small pose models parallelize poorly, `4` often beats all cores |

*Measured on a 16-core CPU: 640px=8.5 FPS → 480px+4 threads=10.7 FPS → 480px+stride2=17.3 FPS → 480px+stride3=21.3 FPS.*
For larger gains, consider ONNX Runtime / OpenVINO export.

---

## Reference

See `DESIGN.md` (local file, not tracked in git) for the mathematical formulations, biomechanical algorithms, Kalman state equations, and multi-view homography.
