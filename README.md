# Real-Time Person & Elderly Fall Detection System
# 实时单人及老人跌倒检测系统

一个面向边缘计算、基于姿态估计（Pose Estimation）的实时跌倒检测系统。系统在严重遮挡场景（如家具、桌子、床、沙发背后）下检测跌倒，并通过多阶段状态机区分跌倒与日常生活活动（坐下、系鞋带、卧床休息等），以降低误报。系统基于 [Ultralytics YOLO26-Pose](https://github.com/ultralytics/ultralytics) 提取 17 个 COCO 关键点，在纯 CPU 上即可运行，无需独立 GPU。

An edge-friendly real-time fall detection system built on pose estimation. It detects falls even under severe occlusion (behind furniture, tables, beds, and sofas) and uses a multi-stage state machine to distinguish falls from activities of daily living (sitting down, tying shoelaces, resting in bed), keeping false alarms low. It extracts 17 COCO keypoints via [Ultralytics YOLO26-Pose](https://github.com/ultralytics/ultralytics) and runs on plain CPU without a dedicated GPU.

---

## 特性 / Key Features

- **瞬时触发检测 / Instant trigger detection**：当发生快速下坠（垂直速度 `v_y > 0.7 h/s`）或姿态快速坍塌（`Δθ/Δt > 35°/s`）时立即触发报警。Triggers alerts the moment a rapid descent (`v_y > 0.7 h/s`) or a fast posture collapse (`Δθ/Δt > 35°/s`) is detected.
- **误报抑制 / False-alarm rejection**：在生物力学层面区分意外跌倒与平躺休息/睡眠，避免日常活动触发误报。Distinguishes accidental falls from horizontal resting/sleeping in bed to avoid false alarms from daily activities.
- **四级遮挡复原 / 4-tier occlusion resilience**：
  1. 骨骼约束插补：依据人体解剖比例重建被遮挡的膝盖、脚踝。/ Bone-constraint imputation: reconstructs occluded knees and ankles from anatomical ratios.
  2. 时间速度外推：在视觉遮挡期间延续运动矢量。/ Temporal velocity extrapolation carries motion vectors through occlusion.
  3. 家具区域跟踪：跟踪进入已知遮挡区域（咖啡桌、床、沙发）的下坠轨迹。/ Furniture-ROI tracking follows descents behind known barriers.
  4. 邻近与骨骼长度约束：限制骨架边界，剔除异常连线。/ Proximity & bone-length clamping removes spurious skeleton lines.
- **轻量模型 / Lightweight model**：姿态估计模型约 6.5 MB，可在普通 CPU 上运行，无需训练自定义模型。/ About 6.5 MB pose model; runs on a normal CPU with no custom training.
- **交互式运行器 / Interactive runner**：命令行交互菜单，支持视频自动发现与拖拽文件。/ CLI menu with auto-discovery of videos and drag-and-drop support.

---

## 快速开始 / Quickstart

只需一条命令 / One command:

```bash
python run.py
```

Windows 下也可直接双击 `run.bat`。/ On Windows you can also double-click `run.bat`.

### 交互菜单 / Interactive menu

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

- 选项 1（摄像头 / Webcam）：实时采集摄像头画面，叠加骨架与遥测信息。/ Live camera feed with real-time skeleton and telemetry overlay.
- 选项 2（视频文件 / Video file）：自动列出目录内的测试视频，支持数字键选择，或直接拖入 `.mp4` / `.avi` 文件。/ Lists available test videos for numeric selection, or accepts a dragged `.mp4` / `.avi` file.
- 选项 3（合成演示 / Synthetic demo）：生成并播放一段包含遮挡场景的合成跌倒演示。/ Generates and plays a synthetic fall demo with an occlusion barrier.
- 选项 4（退出 / Exit）：关闭程序。/ Closes the application.

---

## 自带基准视频 / Benchmark videos (`sample_videos/`)

仓库包含以下基准跌倒测试视频。/ The repo includes these benchmark fall clips:

| 编号 / # | 视频文件 / File | 场景与用途 / Scenario & Purpose | 预期结果 / Expected |
|---|---|---|---|
| 1 | `standing_fall_1.mp4` | 站立状态下骤然跌倒 / Sudden collapse from standing | `[ALERT] FALL DETECTED` |
| 2 | `standing_fall_2.mp4` | 站立状态下滑倒、后仰 / Slip & backward fall | `[ALERT] FALL DETECTED` |
| 3 | `sitting_to_fall.mp4` | 老人从椅子上跌落 / Senior falling off a chair | `[ALERT] FALL DETECTED` |
| 4 | `bed_rollout_fall.mp4` | 从床上滚落 / Roll-out fall from bed | `[ALERT] FALL DETECTED` |
| 5 | `normal_activity_no_fall.mp4` | 行走、坐下、睡眠等日常活动（基线）/ Walking, sitting, sleeping (ADL baseline) | 保持 `MONITORING (RESTING)`，不误报 / No false alarm |

---

## 技术架构与处理流程 / Architecture & Pipeline

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

### 生物力学特征 / Biomechanical features

- **躯干角度（`θ_torso`）/** Torso angle：肩部中点与髋部中点连线相对水平面的角度（`90°` 站立，`0°` 平躺）。Angle of the mid-shoulder–mid-hip line vs. the floor (`90°` standing, `0°` flat).
- **垂直下落速率（`v_y`）/** Vertical descent rate：以人体边界框高度为单位的垂直速度（`h/s`）。Normalized downward velocity in body-box heights per second.
- **角度坍塌速率（`ω = dθ/dt`）/** Angular collapse rate：姿态角度的下降速率（度/秒）。Rate of postural angle drop in degrees per second.
- **静止能量（`E_motion`）/** Stillness energy：可见关键点的时序变化量，用于确认落地后是否静止。Temporal variance of visible keypoints used to confirm post-impact immobility.

### 状态机逻辑 / State machine

```
 [ MONITORING ] ──( v_y > 0.7 h/s OR ω > 35°/s )──> [ FALL_DETECTED ]
       ▲                                                     │
       │                                            ( > 1.5s Stillness )
  ( Stands Up )                                              │
       │                                                     ▼
 [ MONITORING ] <──────( Upright Motion )─────── [ CONFIRMED_FALL ]
```

---

## 目录结构 / Repository layout

```
├── run.bat                          # Windows 一键启动 / One-click launcher
├── run.py                           # 交互式命令行运行器 / Interactive CLI runner
├── fall_detection_all_in_one.py     # 单文件全流程管线 / Standalone all-in-one pipeline
├── core/
│   ├── detector.py                  # YOLO26-Pose 封装与关键点 / Pose wrapper & keypoints
│   ├── tracker.py                   # 多人时间轨迹管理 / Multi-person tracklet manager
│   ├── occlusion.py                 # 遮挡引擎：外推与骨骼插补 / Occlusion & imputation
│   ├── kinematics.py                # 生物力学运动特征引擎 / Kinematic feature engine
│   ├── state_machine.py             # 多阶段跌倒状态机 / Multi-stage fall state machine
│   └── visualizer.py                # 遥测 HUD 可视化 / Telemetry HUD & alert banners
├── sample_videos/                   # 基准跌倒测试数据集 / Benchmark test videos
├── test_generator.py                # 合成遮挡场景生成器 / Synthetic demo generator
├── requirements.txt                 # 依赖清单 / Dependencies
├── FALL_DETECTION_SYSTEM_DESIGN.md  # 系统与数学设计文档 / Design specification
└── README.md                        # 本说明文档 / This document
```

---

## 命令行直接调用 / Direct CLI usage

除了交互菜单，也可以直接使用命令行参数调用。/ Besides the menu, you can run directly with CLI arguments:

```bash
# 摄像头（设备 0）/ Webcam (device 0)
python fall_detection_all_in_one.py --source 0 --show

# 指定视频文件 / A specific video file
python fall_detection_all_in_one.py --source sample_videos/standing_fall_1.mp4 --show --save output_result.mp4

# 合成遮挡演示 / Synthetic occlusion demo
python fall_detection_all_in_one.py --source demo --show
```

---

## CPU 性能调优 / CPU performance tuning

姿态模型默认在 CPU 上运行。推理是主要的性能开销，可通过以下参数在精度与吞吐量之间权衡。/ The pose model runs on CPU by default. Inference dominates; tune these options to trade a little accuracy/latency for throughput:

```bash
# 平衡：降低推理尺寸（推荐），约提升 1.3-1.5 倍 / Balanced: smaller input (~1.3-1.5x)
python main.py --source video.mp4 --imgsz 480

# 快速：每 2 帧运行一次推理，中间帧复用检测结果，再提升约 2 倍 / Fast: DNN every 2nd frame (~2x more)
python main.py --source video.mp4 --imgsz 480 --stride 2 --threads 4

# 限制线程数，减少小模型在多数核下的争用 / Limit threads to avoid contention
python main.py --source video.mp4 --imgsz 640 --threads 4
```

| 参数 / Option | 默认 / Default | 说明 / Description |
|---|---|---|
| `--imgsz` | `640` | 推理输入尺寸。`480`/`320` 可显著提升 CPU 推理速度，但过小会降低小目标的精度 / Input size; smaller is faster but reduces accuracy on tiny subjects |
| `--stride` | `1` | 每 N 帧运行一次推理，其余帧复用上次关键点。`2`-`3` 可近似线性提升帧率 / Run DNN every N frames and reuse keypoints in between |
| `--threads` | `0`（自动 / auto） | Torch CPU 线程数。小姿态模型并行效率差，`4` 通常优于默认的全核配置 / CPU threads; small pose models parallelize poorly, `4` often beats all cores |

*实测（16 核 CPU）/ Measured on a 16-core CPU：640px=8.5 FPS → 480px+4线程/4 threads=10.7 FPS → 480px+stride2=17.3 FPS → 480px+stride3=21.3 FPS。*
如需更大提升，可考虑导出为 ONNX Runtime / OpenVINO 推理。/ For larger gains, consider ONNX Runtime / OpenVINO export.

---

## 参考文档 / Reference

数学公式、生物力学算法、Kalman 滤波器状态方程及多视角单应矩阵等详细说明，参见 [FALL_DETECTION_SYSTEM_DESIGN.md](FALL_DETECTION_SYSTEM_DESIGN.md)。/ See FALL_DETECTION_SYSTEM_DESIGN.md for the mathematical formulations, biomechanical algorithms, Kalman state equations, and multi-view homography.
