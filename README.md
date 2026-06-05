# Football Analysis CV — Graduation Project

A computer vision system that analyzes football match footage frame-by-frame, producing an annotated output video and a self-contained HTML report with per-player and per-team statistics.

---

## Features

- **Player & Ball Detection** — YOLOv8 fine-tuned on a football-specific dataset detects players, goalkeepers, referees, and the ball at every frame.
- **Multi-Object Tracking** — ByteTrack assigns persistent IDs across frames so each player is followed throughout the clip.
- **Automatic Team Assignment** — K-Means clustering on jersey colors separates players into two teams without any manual labeling. Goalkeeper edge-case logic prevents misclassification.
- **Ball Possession** — Each frame is tagged with the player (and therefore team) currently in possession, driving a live ball-control percentage overlay.
- **Camera Motion Compensation** — Lucas-Kanade optical flow estimates pan/tilt movement each frame so player positions are expressed relative to the pitch, not the camera.
- **Perspective Transform** — A homography maps pixel coordinates to real-world pitch meters (68 m × 23.32 m reference plane), enabling accurate physical measurements.
- **Speed & Distance** — Per-player speed (km/h) and cumulative distance (m) are computed in real-world units and rendered on-screen.
- **Pass Detection** — Ball-possession transitions between teammates are counted as passes, with running totals shown on the video and in the report.
- **Heatmaps** — Gaussian-smoothed positional density maps show where each team occupied the pitch.
- **HTML Report** — A dark-themed, self-contained report is generated with match-summary cards, side-by-side heatmaps, and per-player stats tables ranked by distance covered.
- **Stub Caching** — Detection and camera-movement results are pickled on first run so subsequent runs skip the expensive inference step.

---

## Pipeline Overview

```
Input Video
    │
    ▼
YOLOv8 Detection  ──►  ByteTrack  ──►  tracks dict
                                            │
              ┌─────────────────────────────┤
              │                             │
   Camera Movement (LK Optical Flow)        │
   Perspective Transform (Homography)       │
   Team Assignment (K-Means)                │
   Ball Possession                          │
   Speed & Distance                         │
   Pass Detection                           │
              │                             │
              ▼                             ▼
    Annotated Output Video          HTML Analysis Report
```

---

## Project Structure

```
├── main.py                          # Entry point — orchestrates the full pipeline
├── yolo_inference.py                # Quick standalone YOLOv8 inference script
│
├── trackers/                        # YOLOv8 detection + ByteTrack integration
├── team_assigner/                   # K-Means jersey-color team classification
├── player_ball_assigner/            # Ball-to-player proximity assignment
├── camera_movement_estimator/       # Lucas-Kanade optical flow compensation
├── view_transformer/                # Perspective transform (pixels → metres)
├── speed_and_distance_estimator/    # Real-world speed & distance computation
├── pass_detector/                   # Pass counting from possession transitions
├── heatmap_visualizer/              # Standalone heatmap utility
├── report_generator/                # Self-contained HTML report builder
├── utils/                           # Shared geometry helpers
│
├── training/
│   ├── football_training_yolo_v5.ipynb   # Training notebook
│   └── football-players-detection-1/     # Roboflow dataset config
│
├── development_and_analysis/
│   └── color_assignment.ipynb            # Jersey-color exploration notebook
│
├── models/                          # Place best.pt here (not tracked by git)
├── input_videos/                    # Place source footage here (not tracked)
├── output_videos/                   # Annotated video & report written here
└── stubs/                           # Cached detection/camera pkl files
```

---

## Setup

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU is recommended for real-time inference speed.

### Install Dependencies

```bash
pip install ultralytics supervision opencv-python numpy pandas scikit-learn
```

### Download the Model

Place the fine-tuned weights at `models/best.pt`.  
The model was trained on the [Roboflow Football Players Detection dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1) (CC BY 4.0).  
Training code is in [`training/football_training_yolo_v5.ipynb`](training/football_training_yolo_v5.ipynb).

---

## Usage

1. Place your input video in `input_videos/`.
2. Open `main.py` and update `VIDEO_PATH`:

```python
VIDEO_PATH = 'input_videos/your_clip.mp4'
```

3. Run:

```bash
python main.py
```

**First run** — runs full YOLOv8 inference and saves stubs to `stubs/`.  
**Subsequent runs** — loads stubs instantly, skipping inference.

### Output

| File | Description |
|---|---|
| `output_videos/output_video.avi` | Annotated video with bounding boxes, IDs, speed, distance, pass counts, and ball-control overlay |
| `output_videos/report.html` | Self-contained HTML match report (open in any browser) |

---

## Report

The HTML report includes:

- **Match Summary** — ball control %, pass count, total distance, and top speed per team
- **Player Heatmaps** — pitch-overlay heatmaps showing positional density for each team
- **Player Statistics Tables** — distance, average speed, and max speed ranked per player with gold/silver/bronze medals for the top 3

---

## Key Technical Details

| Component | Technique |
|---|---|
| Object detection | YOLOv8x fine-tuned on 4 classes: `ball`, `goalkeeper`, `player`, `referee` |
| Multi-object tracking | ByteTrack (via Supervision) |
| Team classification | K-Means (k=2) on upper-body jersey RGB, with position-based goalkeeper override |
| Camera compensation | Lucas-Kanade sparse optical flow on static background regions |
| Coordinate mapping | OpenCV `getPerspectiveTransform` + `perspectiveTransform` |
| Speed/distance | Sliding 5-frame window over real-world coordinates at 24 fps |
| Pass detection | Possession-transfer events with minimum 100 px displacement filter |
| Heatmaps | Gaussian blur (σ=25) over sampled player positions, blended over a green pitch |

---

## Technologies

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Supervision](https://github.com/roboflow/supervision) (ByteTrack)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/) (K-Means)

---

## Dataset

Training data sourced from Roboflow Universe:  
**Football Players Detection** — [roboflow-jvuqo/football-players-detection-3zvbc](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)  
License: CC BY 4.0
