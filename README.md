# FaceID System

This project implements a comprehensive FaceID system with:
- **RetinaFace**: Face detection and alignment
- **ArcFace**: Face recognition with 512-d embeddings
- **3D Face Alignment**: Deep 3D Morphable Models for pose estimation and mesh reconstruction
- **XceptionNet-FAS**: Face Anti-Spoofing (Texture/Depth)
- **rPPG**: Remote Photoplethysmography (Heart rate based liveness)

## Features

### 3D Face Alignment ✨
- **3D Face Reconstruction**: Reconstructs 3D face mesh from 2D images
- **Head Pose Estimation**: Real-time pitch, yaw, roll angle estimation
- **68 3D Landmarks**: Extracts detailed 3D facial landmarks
- **Interactive Visualization**: Toggle mesh, pose axes, and landmarks in real-time

## Setup

1.  **Install Dependencies**:
    ```bash
    py install_insightface.py
    py -m pip install -r requirements.txt
    ```

2.  **Download Models**:
    -   **InsightFace Models**: Run the setup script to trigger automatic download:
        ```bash
        py setup_models.py
        ```
    -   **FAS Model**: The `setup_models.py` script will check for this. You still need to manually download `MiniFASNetV2.onnx` and save it as `models/fas.onnx` if you want real liveness detection.

## Usage

1.  **Run the System**:
    ```bash
    py main.py
    ```

2.  **Keyboard Controls**:
    -   `q` - Exit the application
    -   `r` - Register a new face (requires exactly one face visible)
    -   `t` - Toggle 3D mesh wireframe visualization
    -   `p` - Toggle 3D pose axes (X, Y, Z)
    -   `l` - Toggle 3D facial landmarks display

3.  **Registration Mode**:
    -   Press `r` when only one face is visible
    -   Type the person's name
    -   Press `Enter` to save or `Esc` to cancel

## Modules

### Core Modules
-   `modules/detector.py`: Face detection using RetinaFace
-   `modules/recognizer.py`: Face embedding extraction using ArcFace
-   `modules/tracker.py`: Multi-face tracking across frames

### 3D Alignment (NEW)
-   `modules/alignment_3d.py`: 3D face alignment using Deep 3D Morphable Models
    - Pose estimation (pitch, yaw, roll)
    - 3D mesh reconstruction
    - 3D landmark extraction
-   `modules/utils_3d.py`: 3D visualization utilities
    - Mesh wireframe rendering
    - Pose axes drawing
    - 3D landmark visualization

### Liveness Detection
-   `modules/liveness/fas.py`: Face Anti-Spoofing using deep learning
-   `modules/liveness/ppg.py`: rPPG-based liveness using heart rate signals

## 3D Visualization Guide

The system displays:
- **Green wireframe**: 3D face mesh (toggle with `t`)
- **RGB axes**: Head pose orientation (toggle with `p`)
  - Red: X-axis (left-right)
  - Green: Y-axis (up-down)
  - Blue: Z-axis (forward-backward)
- **Colored landmarks**: 68 3D facial points (toggle with `l`)
  - Cyan: Face contour
  - Green: Eyebrows
  - Orange: Nose
  - Blue: Eyes
  - Red: Mouth
- **Pose angles**: Real-time pitch, yaw, roll values (top-left corner)

## Technical Details

### 3D Morphable Model
- Based on Basel Face Model (BFM) parameters
- 68-point 3D landmark model
- PnP-based pose estimation using OpenCV
- Real-time mesh reconstruction and projection

### Performance
- ~30 FPS on modern CPUs (with GPU acceleration for InsightFace)
- Lightweight 3D alignment (~5ms per face)
- Multi-face support with individual tracking
