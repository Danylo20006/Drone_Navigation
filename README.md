# UAV Visual Navigation System (GPS-Denied Map Matching)

An end-to-end computer vision pipeline for **autonomous UAV localization and trajectory reconstruction** in GPS-denied / Electronic Warfare (EW) environments. By matching onboard camera frames against a pre-loaded georeferenced satellite map, the system estimates the drone's position, footprint, and heading in both offline analytics and real-time live flight modes.

## Table of Contents

* [Overview & Problem Statement](#overview--problem-statement)

* [Key Features](#key-features)

* [System Architecture](#system-architecture)

* [Project Structure](#project-structure)

* [Installation](#installation)

* [Usage](#usage)

  * [1. Offline Pipeline (Basic & Advanced Tasks)](#1-offline-pipeline-basic--advanced-tasks)

  * [2. Live Flight Demo with a Smartphone Camera](#2-live-flight-demo-with-a-smartphone-camera)

  * [3. Engineering Benchmarks](#3-engineering-benchmarks)

* [Benchmark Results](#benchmark-results)

* [Output Format](#output-format)

* [Performance Tuning](#performance-tuning)

* [Future Roadmap](#future-roadmap)

## Overview & Problem Statement

Under Electronic Warfare (EW) jamming or spoofing, satellite navigation (GNSS/GPS) becomes unavailable or unreliable. This project implements a **visual odometry / map-matching navigation system** that relies solely on optical imagery:

1. **Basic Mode:** Localizes a sequence of aerial image crops onto a high-resolution satellite map and reconstructs the complete flight path.

2. **Advanced Mode:** Processes real-world or screen-recorded drone video streams subject to glare, moiré artifacts, motion blur, and perspective distortion.

3. **Real-Time Mode (`live_demo.py`):** Streams live video from a USB camera or smartphone, displaying a split-screen **Tactical HUD** with real-time position tracking, heading estimation, and automatic relocalization after signal blackout.

## Key Features

* **Adaptive ROI-Based Tracking:** After initial localization, feature matching is constrained to a local Region of Interest (`roi_radius`) around the last known coordinates—dramatically increasing FPS (up to $7.05\times$ speedup for SIFT) and suppressing false matches across repetitive urban textures.

* **5-Level Homography Sanity Checks:** Every RANSAC homography candidate $H \in \mathbb{R}^{3 \times 3}$ is validated against physical constraints:

  1. **Minimum Inliers:** Rejects weak geometric matches (`min_inliers`).

  2. **Determinant Check:** Enforces $\det(H_{2 \times 2}) > 10^{-6}$ to eliminate reflections and degenerate projections.

  3. **Map Bounds:** Ensures the projected UAV center lies within satellite map boundaries.

  4. **Quadrilateral Convexity:** Verifies that the projected 4-corner footprint forms a convex polygon.

  5. **Scale & Jump Limits:** Bounds the projected area ratio (`min_area_ratio` to `max_area_ratio`) and maximum per-frame displacement (`max_jump_px`).

* **EW / Blackout Resilience & Auto-Relocalization:** On consecutive lost frames (smoke, occlusion, jamming), the ROI dynamically expands (`expand_factor = 1.8`). After `max_lost_before_global = 4` misses, the tracker seamlessly falls back to full-map `GLOBAL` search until lock is re-acquired.

* **Trajectory Post-Processing:** Linearly interpolates missing waypoints during brief signal losses and applies a **Savitzky-Golay filter** (`scipy.signal.savgol_filter`) to remove high-frequency jitter.

* **Tactical HUD & Visualization:** Generates static route maps with semi-transparent frame footprints, smoothstep-interpolated zoomed MP4 flyover videos, and a live split-screen HUD with real-time **Heading** ($\theta = \operatorname{atan2}(H_{1,0}, H_{0,0})$).

## System Architecture

```
┌───────────────────┐     ┌─────────────────────────┐     ┌───────────────────────────────┐
│   Input Source    │────▶│     Pre-processing      │────▶│       Feature Matching        │
│ • Image Directory │     │ • Downscaling (scale)   │     │ • SIFT (Accuracy) / ORB (FPS) │
│ • Video File      │     │ • Optional CLAHE        │     │ • Ratio Test + RANSAC H (3x3) │
│ • USB / Phone Cam │     │   (glare/contrast norm) │     └──────────────┬────────────────┘
└───────────────────┘     └─────────────────────────┘                    │
                                                                         ▼
┌───────────────────┐     ┌─────────────────────────┐     ┌───────────────────────────────┐
│      Outputs      │◀────│     Post-processing     │◀────│         DroneTracker          │
│ • trajectory.json │     │ • Linear Interpolation  │     │ • Local ROI / Global Fallback │
│ • Route Map PNG   │     │ • Savitzky-Golay Smooth │     │ • 5-Level Sanity Checks       │
│ • Zoomed MP4      │     └─────────────────────────┘     │ • Dynamic ROI Expansion       │
│ • Live Split HUD  │                                     └───────────────────────────────┘
└───────────────────┘

```

## Project Structure

```
drone-navigation/
├── main.py                  # Offline CLI pipeline (Basic & Advanced tasks)
├── live_demo.py             # Real-time split-screen demo with Tactical HUD
├── requirements.txt         # Python dependencies
├── src/
│   ├── io_utils.py          # Frame loaders for image directories and video files
│   ├── matcher.py           # SIFT/ORB feature matching, CLAHE, and homography estimation
│   ├── tracker.py           # DroneTracker (ROI, sanity checks, relocalization) & post-processing
│   ├── visualizer.py        # Static route map & smoothstep zoomed MP4 renderer
│   └── schemas.py           # FrameResult dataclass & JSON serialization helpers
├── tests/
│   └── benchmark.py         # Engineering benchmarks (SIFT vs ORB, ROI speedup, CLAHE impact)
└── data/                    # (Git-ignored) Global satellite map, crops, and test videos

```

## Installation

### Prerequisites

* Python **3.10+**

* Git

* *(Optional for Live Demo)* A USB webcam or smartphone with **Iriun Webcam** / **DroidCam**

### Setup

```
# 1. Clone the repository
git clone https://github.com/<your-username>/drone-navigation.git
cd drone-navigation

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

```

Place your satellite map and input datasets inside the `data/` directory (e.g., `data/global_map.png`, `data/frames/`, `data/drone_video1.mp4`).

## Usage

### 1. Offline Pipeline (Basic & Advanced Tasks)

Run `main.py` to localize frames, export `trajectory.json`, draw `route_visualization.png`, and render `trajectory_zoom.mp4`.

**Basic Task (Folder of image crops):**

```
python main.py \
  --map data/global_map.png \
  --input data/frames \
  --output-dir output \
  --method sift \
  --crop-scale 0.25

```

**Advanced Task (Screen-recorded or onboard drone video):**
Enable `--use-clahe` to normalize lighting/glare and `--stride` to sample video frames:

```
python main.py \
  --map data/global_map.png \
  --input data/drone_video1.mp4 \
  --output-dir output_advanced \
  --method sift \
  --crop-scale 0.25 \
  --use-clahe \
  --stride 5

```

**Key CLI Options (`main.py`):**

| Flag | Default | Description | 
| ----- | ----- | ----- | 
| `--map` | *(required)* | Path to the global satellite map image. | 
| `--input` | *(required)* | Path to a directory of frames or a video file (`.mp4`, `.avi`). | 
| `--output-dir` | `output` | Directory for saving JSON, PNG, and MP4 results. | 
| `--method` | `sift` | Feature descriptor: `sift` (high accuracy) or `orb` (high speed). | 
| `--crop-scale` | `0.25` | Frame downscale factor before feature extraction. | 
| `--use-clahe` | `False` | Apply CLAHE contrast normalization (recommended for video/camera). | 
| `--stride` | `5` | Frame sampling interval when `--input` is a video file. | 
| `--no-smooth` | `False` | Disable Savitzky-Golay trajectory smoothing. | 
| `--no-video` | `False` | Skip MP4 rendering (outputs JSON and static PNG only). | 

### 2. Live Flight Demo with a Smartphone Camera

`live_demo.py` simulates an onboard real-time UAV navigation system using a live camera feed pointed at a satellite map (displayed on a secondary monitor or printed on paper).

#### Step 1: Connect Your Smartphone as a USB Webcam

1. Install **Iriun Webcam** (recommended) or **DroidCam** on both your smartphone and PC.

2. Connect your phone via **USB cable** *(avoid Wi-Fi during live demos to prevent latency and frame drops)*.

   * **Android:** Enable *USB Debugging* in Developer Options.

   * **iOS:** Trust the computer when prompted.

3. Launch the app on both devices. Your phone will appear as a system camera (typically index `0`, or `1` if your laptop has a built-in webcam).

#### Step 2: Prepare the Map Surface

* **Best Option:** Print `data/global_map.png` on A3/A4 color paper (eliminates screen glare and pixel moiré).

* **Screen Option:** Display `data/global_map.png` fullscreen on a second monitor with moderate brightness and run with `--use-clahe`.

* Hold the phone roughly **25–30 cm** above the map and move smoothly.

#### Step 3: Launch Live Tracking

**Mode A — Real-Time Camera Feed (USB Phone / Webcam):**

```
# High-speed mode (ORB) — recommended for fluid live tracking
python live_demo.py --map data/global_map.png --source 0 --method orb --use-clahe --crop-scale 0.35

# High-precision mode (SIFT) — stronger under rotation and scale changes
python live_demo.py --map data/global_map.png --source 0 --method sift --use-clahe --crop-scale 0.25

```

*(If the wrong camera opens, change `--source 0` to `--source 1`.)*

**Mode B — Video Stream Fallback (Stage Safety Net):**
Pass a video file to `--source` to run the exact same live HUD interface with seamless looping:

```
python live_demo.py --map data/global_map.png --source data/drone_video1.mp4 --method orb --use-clahe --crop-scale 0.35 --stride 5

```

#### Interactive Hotkeys

| Key | Action | 
| ----- | ----- | 
| **`Space`** | Pause / Resume the live stream | 
| **`R`** | Reset tracker and trigger immediate `GLOBAL` relocalization | 
| **`C`** | Clear the drawn trajectory trail without losing lock | 
| **`Q` / `Esc`** | Exit the application | 

#### Demonstrating EW / Blackout Recovery Live

1. Move the camera smoothly over the map until the HUD displays **`STATUS: LOCKED | MODE: ROI`**.

2. Cover the camera lens with your hand for 2–3 seconds to simulate jamming or cloud cover. Observe that the sanity checks prevent false jumps while the status switches from **`LOST`** to **`RELOCALIZING (GLOBAL)`**.

3. Uncover the lens over a completely different region of the map—the tracker automatically re-acquires the global position and resumes **`ROI`** tracking.

### 3. Engineering Benchmarks

Run the benchmark suite from the repository root using Python's module flag (`-m`) to compare `SIFT` vs `ORB`, measure ROI speedup, and evaluate CLAHE impact:

```
# Quick sanity benchmark (first 10 frames per dataset)
python -m tests.benchmark \
  --map data/global_map.png \
  --crops data/frames \
  --video data/drone_video1.mp4 \
  --stride 5 \
  --scale 0.25 \
  --limit 10

# Full benchmark (all frames)
python -m tests.benchmark \
  --map data/global_map.png \
  --crops data/frames \
  --video data/drone_video1.mp4 \
  --stride 5 \
  --scale 0.25

```

## Benchmark Results

Measured on CPU using `tests/benchmark.py` across both Basic (cropped frames) and Advanced (screen-recorded video) datasets:

### Basic Dataset (50 frames, `crop_scale=0.25`)

| Mode | Avg ms/frame | Speedup | Success | Avg Inliers | 
| ----- | ----- | ----- | ----- | ----- | 
| Full-map SIFT | 6584.4 | 1.00x | 50/50 (100.0%) | 256.6 | 
| **ROI SIFT** | **934.4** | **7.05x** | **50/50 (100.0%)** | **269.1** | 
| ROI SIFT + CLAHE | 993.4 | 6.63x | 50/50 (100.0%) | 256.3 | 
| ROI ORB | 535.9 | 12.29x | 0/50 (0.0%) | 0.0 | 
| ROI ORB + CLAHE | 523.9 | 12.57x | 0/50 (0.0%) | 0.0 | 

### Video Dataset: `drone_video1.mp4` (94 frames, `stride=5`, `crop_scale=0.25`)

| Mode | Avg ms/frame | Speedup | Success | Avg Inliers | 
| ----- | ----- | ----- | ----- | ----- | 
| Full-map SIFT | 7212.9 | 1.00x | 94/94 (100.0%) | 954.3 | 
| ROI SIFT | 1518.0 | 4.75x | 94/94 (100.0%) | 1032.6 | 
| ROI SIFT + CLAHE | 2169.6 | 3.32x | 94/94 (100.0%) | 1127.5 | 
| ROI ORB | 507.0 | 14.23x | 75/94 (79.8%) | 160.4 | 
| **ROI ORB + CLAHE** | **458.1** | **15.74x** | **94/94 (100.0%)** | **118.4** | 

### Key Takeaways

* **ROI Tracking Speedup:** Constraining matching to a local ROI yields a $7.05\times$ **speedup** on the Basic dataset while maintaining **100% accuracy** and slightly increasing average inliers ($256.6 \rightarrow 269.1$) by eliminating false global matches.

* **CLAHE Impact on Video:** On screen-recorded video, enabling CLAHE boosts `ROI ORB` tracking success from **79.8% to 100.0%** and achieves a $15.74\times$ **speedup** ($458.1$ ms/frame) by preventing frame drops and expensive global relocalizations.

* **SIFT vs. ORB Trade-off:** `SIFT + ROI` excels on high-variance crops with significant scale/angle differences where binary ORB descriptors fail ($0/50$), while `ORB + CLAHE + ROI` is the optimal engine for fluid real-time video and live camera streams.

## Output Format

The pipeline exports localized coordinates to `trajectory.json` as an array of serialized `FrameResult` objects:

```
[
  {
    "frame_id": 0,
    "center": [1245.4, 832.1],
    "corners": [
      [1110.2, 740.5],
      [1382.6, 745.8],
      [1378.1, 924.0],
      [1107.9, 918.2]
    ],
    "inliers": 184,
    "valid": true,
    "interpolated": false
  }
]

```

## Performance Tuning

* **Offline Analytics (Maximum Accuracy):** Use `--method sift --crop-scale 0.25`. SIFT provides superior scale and rotation invariance for high-altitude or angled views.

* **Real-Time Live Demo (Maximum FPS):** Use `--method orb --crop-scale 0.35 --use-clahe` with a tighter `TrackerConfig(roi_radius=300)` to achieve smooth real-time tracking on CPU.

* **Screen / Glare Compensation:** Always pass `--use-clahe` when matching against screen-recorded video or live camera feeds pointed at a monitor.

## Future Roadmap

* **Learned Feature Matchers:** Integrate lightweight GPU-accelerated models (e.g., SuperPoint + LightGlue) for cross-seasonal and night/thermal localization.

* **Sensor Fusion:** Combine visual homography fixes with IMU / Optical Flow via an Extended Kalman Filter (EKF) for high-speed maneuvers.

* **Edge Deployment:** Optimize and deploy the `ORB + ROI` pipeline to onboard companion computers (NVIDIA Jetson Orin Nano / Raspberry Pi 5).