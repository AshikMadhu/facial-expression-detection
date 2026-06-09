# Local-First Project Architecture Summary

This document describes the architectural layout and processing pipeline of the local-first **EmotionSense AI** system.

---

## 🏛️ System Overview

EmotionSense AI processes video frames from a local webcam, detects and aligns human faces, and classifies facial expressions using a deep convolutional neural network. The results are fed into a telemetry metrics engine to analyze user engagement.

All calculations, face alignment, and model inferences are conducted **completely locally** on your machine's CPU/GPU, ensuring data privacy and low-latency execution.

---

## 🧱 Key Components

```
┌─────────────────────────────────────────────────────────────────┐
│                          run.py (CLI)                           │
└────────┬───────────────────┬───────────────────┬────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   train.py      │ │   webcam mode   │ │ dashboard mode  │
│  (src/train.py) │ │(realtime_webcam)│ │(src/dashboard.py)
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│             inference.py (EmotionInferenceEngine)               │
│  Loads blaze_face_short_range.tflite & best_model.h5 weights    │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Unified CLI Launcher (`run.py`)
Acts as the single entry point to execute the system in one of its four modes: `dashboard`, `webcam`, `evaluate`, or `train`.

### 2. Config Module (`src/config.py`)
Uses `pathlib.Path` objects to manage file paths dynamically and contains custom checks to validate hyperparameters during initialization.

### 3. Inference Engine (`src/inference.py`)
*   **MediaPipe Face Detection**: Uses a lightweight Google BlazeFace short-range TFLite model to extract facial regions of interest (ROI) and keypoints.
*   **Eye Alignment & Normalization**: Computes eye angles to rotate, warp, crop, and resize the face to `48x48` grayscale pixels.
*   **Keras Classification**: Runs inference on the cropped face using the transfer learning model (EfficientNetV2B0) and outputs predictions for the 7 standard emotions.

### 4. OpenCV Webcam Engine (`src/realtime_webcam.py`)
Runs a high-performance local video display loop.
*   Uses a normal resizable window (`cv2.WINDOW_NORMAL`).
*   Supports toggling fullscreen via the **`f`** key.
*   Detects if the user closes the window manually via window manager controls.

### 5. Streamlit Dashboard (`src/dashboard.py`)
Provides a web interface displaying live metrics (current dominant emotion, valence index, and distraction logs), probability histograms, and session reports. Displays are optimized to scale responsively.

### 6. Pipeline & Training Managers (`src/fer2013_pipeline.py`, `src/train.py`)
Handles data cleaning, class balance corrections, image augmentations, and execution of the two-stage model training.
