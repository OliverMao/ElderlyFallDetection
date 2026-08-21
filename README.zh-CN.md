# 实时单人及老人跌倒检测系统

> **English version**: [README.md](README.md)

一个面向边缘计算、基于姿态估计（Pose Estimation）的实时跌倒检测系统。系统在严重遮挡场景（如家具、桌子、床、沙发背后）下检测跌倒，并通过多阶段状态机区分跌倒与日常生活活动（坐下、系鞋带、卧床休息等），以降低误报。系统基于 [Ultralytics YOLO26-Pose](https://github.com/ultralytics/ultralytics) 提取 17 个 COCO 关键点，在纯 CPU 上即可运行，无需独立 GPU。

---

## 特性

- **瞬时触发检测**：当发生快速下坠（垂直速度 `v_y > 0.7 h/s`）或姿态快速坍塌（`Δθ/Δt > 35°/s`）时立即触发报警。
- **误报抑制**：在生物力学层面区分意外跌倒与平躺休息/睡眠，避免日常活动触发误报。
- **四级遮挡复原**：
  1. 骨骼约束插补：依据人体解剖比例重建被遮挡的膝盖、脚踝。
  2. 时间速度外推：在视觉遮挡期间延续运动矢量。
  3. 家具区域跟踪：跟踪进入已知遮挡区域（咖啡桌、床、沙发）的下坠轨迹。
  4. 邻近与骨骼长度约束：限制骨架边界，剔除异常连线。
- **轻量模型**：姿态估计模型约 6.5 MB，可在普通 CPU 上运行，无需训练自定义模型。
- **交互式运行器**：命令行交互菜单，支持视频自动发现与拖拽文件。

---

## 快速开始

只需一条命令：

```bash
python run.py
```

Windows 下也可直接双击 `run.bat`。

### 交互菜单

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

- 选项 1（摄像头）：实时采集摄像头画面，叠加骨架与遥测信息。
- 选项 2（视频文件）：自动列出目录内的测试视频，支持数字键选择，或直接拖入 `.mp4` / `.avi` 文件。
- 选项 3（合成演示）：生成并播放一段包含遮挡场景的合成跌倒演示。
- 选项 4（退出）：关闭程序。

---

## 自带基准视频（`sample_videos/`）

仓库包含以下基准跌倒测试视频：

| 编号 | 视频文件 | 场景与用途 | 预期结果 |
|---|---|---|---|
| 1 | `standing_fall_1.mp4` | 站立状态下骤然跌倒 | 触发 `[ALERT] FALL DETECTED` |
| 2 | `standing_fall_2.mp4` | 站立状态下滑倒、后仰 | 触发 `[ALERT] FALL DETECTED` |
| 3 | `sitting_to_fall.mp4` | 老人从椅子上跌落 | 触发 `[ALERT] FALL DETECTED` |
| 4 | `bed_rollout_fall.mp4` | 从床上滚落 | 触发 `[ALERT] FALL DETECTED` |
| 5 | `normal_activity_no_fall.mp4` | 行走、坐下、睡眠等日常活动（基线） | 保持 `MONITORING (RESTING)`，不误报 |

---

## 技术架构与处理流程

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

### 生物力学特征

- **躯干角度（`θ_torso`）**：肩部中点与髋部中点连线相对水平面的角度（`90°` 站立，`0°` 平躺）。
- **垂直下落速率（`v_y`）**：以人体边界框高度为单位的垂直速度（`h/s`）。
- **角度坍塌速率（`ω = dθ/dt`）**：姿态角度的下降速率（度/秒）。
- **静止能量（`E_motion`）**：可见关键点的时序变化量，用于确认落地后是否静止。

### 状态机逻辑

```
 [ MONITORING ] ──( v_y > 0.7 h/s OR ω > 35°/s )──> [ FALL_DETECTED ]
       ▲                                                     │
       │                                            ( > 1.5s Stillness )
  ( Stands Up )                                              │
       │                                                     ▼
 [ MONITORING ] <──────( Upright Motion )─────── [ CONFIRMED_FALL ]
```

---

## 目录结构

```
├── run.bat                          # Windows 一键启动脚本
├── run.py                           # 交互式命令行运行器
├── fall_detection_all_in_one.py     # 单文件全流程管线
├── core/
│   ├── detector.py                  # YOLO26-Pose 封装与关键点提取
│   ├── tracker.py                   # 多人时间轨迹管理
│   ├── occlusion.py                 # 遮挡引擎：外推与骨骼插补
│   ├── kinematics.py                # 生物力学运动特征引擎
│   ├── state_machine.py             # 多阶段跌倒状态机
│   └── visualizer.py                # 遥测 HUD 可视化与报警横幅
├── sample_videos/                   # 基准跌倒测试数据集
├── test_generator.py                # 合成遮挡场景生成器
├── requirements.txt                 # 依赖清单
├── FALL_DETECTION_SYSTEM_DESIGN.md  # 系统与数学设计文档
└── README.md                        # 说明文档（英文版）
```

---

## 命令行直接调用

除了交互菜单，也可以直接使用命令行参数调用：

```bash
# 摄像头（设备 0）
python fall_detection_all_in_one.py --source 0 --show

# 指定视频文件
python fall_detection_all_in_one.py --source sample_videos/standing_fall_1.mp4 --show --save output_result.mp4

# 合成遮挡演示
python fall_detection_all_in_one.py --source demo --show
```

---

## CPU 性能调优

姿态模型默认在 CPU 上运行。推理是主要的性能开销，可通过以下参数在精度与吞吐量之间权衡：

```bash
# 平衡：降低推理尺寸（推荐），约提升 1.3-1.5 倍
python main.py --source video.mp4 --imgsz 480

# 快速：每 2 帧运行一次推理，中间帧复用检测结果，再提升约 2 倍
python main.py --source video.mp4 --imgsz 480 --stride 2 --threads 4

# 限制线程数，减少小模型在多数核下的争用
python main.py --source video.mp4 --imgsz 640 --threads 4
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--imgsz` | `640` | 推理输入尺寸。`480`/`320` 可显著提升 CPU 推理速度，但过小会降低小目标的精度 |
| `--stride` | `1` | 每 N 帧运行一次推理，其余帧复用上次关键点。`2`-`3` 可近似线性提升帧率 |
| `--threads` | `0`（自动） | Torch CPU 线程数。小姿态模型并行效率差，`4` 通常优于默认的全核配置 |

*实测（16 核 CPU）：640px=8.5 FPS → 480px+4线程=10.7 FPS → 480px+stride2=17.3 FPS → 480px+stride3=21.3 FPS。*
如需更大提升，可考虑导出为 ONNX Runtime / OpenVINO 推理。

---

## 参考文档

数学公式、生物力学算法、Kalman 滤波器状态方程及多视角单应矩阵等详细说明，参见 [FALL_DETECTION_SYSTEM_DESIGN.zh-CN.md](FALL_DETECTION_SYSTEM_DESIGN.zh-CN.md)。
