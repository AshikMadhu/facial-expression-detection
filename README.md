# 🎭 EmotionSense AI

> **A Production-Grade, Local-First Facial Emotion Recognition & User Attention Analytics Platform**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/tensorflow-2.15-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.58-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 1. Project Overview

**EmotionSense AI** is a state-of-the-art, local-first computer vision and deep learning platform designed to analyze facial expressions, gaze dynamics, and user attention in real-time. By utilizing advanced alignment algorithms and transfer learning models, the system maps facial landmarks to specific emotion categories while calculating secondary analytics like focus tracking and emotional valence indices.

### Why It Was Built
Traditional facial recognition systems rely on heavy cloud services, introducing latency, security risks, and bandwidth constraints. EmotionSense AI is engineered to be **entirely local-first**, processing image frames, running inference, and compiling analytics dashboards completely on the host system without sending any frame data over the network.

### Real-World Applications
- **EdTech Analytics**: Monitor student focus, engagement, and emotional comprehension during virtual lectures.
- **Usability Testing**: Track user reactions, frustration spikes, and visual attention patterns during application design audits.
- **Healthcare**: Assist clinicians in tracking emotional responsiveness and motor reactions over time.

---

## ✨ 2. Key Features

- **Real-Time Webcam Mapping**: Zero-latency facial emotion overlays using local camera streams, featuring bounding box scaling and full-screen support.
- **MediaPipe Facial Alignment**: Corrects for head tilt, pitch, and roll to guarantee face crops remain normalized.
- **Interactive Streamlit Dashboard**: Ingests session data to chart emotional timelines, visual engagement levels, and focus tracking.
- **Offline Analytics Engine**: Computes focus scores, valence indices (net positivity), and reports statistics in structured JSON reports.
- **Robust Environment Pre-flight Checks**: Startup checks prevent launch crashes by verifying model and data locations automatically.

---

## 🛠️ 3. Technology Stack

| Library | Purpose | How It Is Used |
| :--- | :--- | :--- |
| **TensorFlow** | Machine Learning Backend | Compiles the transfer model architecture and executes deep learning tensor calculations. |
| **EfficientNetV2** | CNN Architecture | Serves as the pre-trained feature extractor, optimizing classification accuracy on mobile/desktop CPUs. |
| **MediaPipe** | Face Tracking & Alignment | Detects face locations and aligns faces using 3D facial landmark meshes. |
| **OpenCV** | Computer Vision Framework | Handles video stream inputs, processes resizing, and draws graphical overlays. |
| **NumPy** | Numerical Computations | Performs fast multi-dimensional matrix operations on image channels and arrays. |
| **Pandas** | Data Wrangling | Manages dataset indexing, sanitization, class weighting, and metrics loading. |
| **Scikit-Learn** | ML Utilities | Computes metrics like confusion matrices, precision, recall, and F1 scores. |
| **Plotly** | Interactive Visualizations | Generates dynamic engagement curves and emotional distribution plots for the dashboard. |
| **Streamlit** | UI Framework | Powers the local dashboard web interface. |
| **Matplotlib** | Static Reporting Plots | Renders the baseline validation dashboard and error distribution charts. |
| **Seaborn** | Heatmaps & Plot Styling | Styles the confusion matrix visualizations with professional color palettes. |

---

## 📐 4. System Architecture

Below is the conceptual flow showing how camera frames are captured, processed, evaluated, and displayed entirely on the local system:

```text
[ Local Webcam ] ---> ( OpenCV Frame Grabber )
                            │
                            ▼
               ( MediaPipe Face Detector )
                            │  [Detects Landmarks]
                            ▼
               ( Crop & Rotation Alignment )
                            │  [Normalized 160x160 Grayscale Frame]
                            ▼
               ( EfficientNetV2 Model )
                            │  [Runs Model Inference]
                            ▼
               ( Emotion Inference Engine ) ───> [ Real-time OpenCV GUI ]
                            │  [Outputs Emotions & Latency]
                            ▼
               ( Emotion Analytics Engine )
                            │  [Computes Gaze, Valence & Focus]
                            ▼
               [ Streamlit Web Dashboard ] <─── [ User Interactive UI ]
```

---

## 📁 5. Repository Structure

```text
emotion/
├── run.py                    # central centralized mode launcher & environment check script
├── requirements.txt          # Python library dependencies (pinned for compatibility)
├── .gitignore                # Prevents tracking virtualenvs, datasets, and cache files
├── README.md                 # Project user manual and specifications
├── data/                     # Data folder
│   ├── fer2013.csv           # Raw FER2013 dataset (ignored by Git)
│   └── reports/              # Schema and preparation audit outputs
├── models/                   # Local model checkpoints
│   ├── best_model.h5         # Trained CNN model weights
│   └── blaze_face_short_range.tflite # MediaPipe face landmark model
├── src/                      # Source code directory
│   ├── __init__.py
│   ├── config.py             # Hyperparameters, labels, and file paths
│   ├── model.py              # EfficientNetV2 model compilation logic
│   ├── fer2013_pipeline.py   # Dataset sanitization and input pipelines
│   ├── train.py              # Stage 1 and Stage 2 model fitting loop
│   ├── evaluate.py           # Metrics calculator and plot generator
│   ├── realtime_webcam.py    # Local OpenCV camera stream UI
│   ├── dashboard.py          # Streamlit browser UI
│   ├── inference.py          # Wrapper for loading weights and prediction
│   ├── analytics.py          # Score tracking (Valence, Gaze, Focus)
│   └── plot_curves.py        # Charting utility
├── scripts/                  # Pre-flight diagnostic scripts
│   ├── verify_installation.py# Checks packages and version matches
│   └── system_diagnostics.py  # Inspects platform hardware and GPU visibility
└── tests/                    # Unit testing suite
    └── test_suite.py         # 10/10 mock-supported testing assertions
```

---

## 🚀 6. Installation Guide

Follow these steps to establish a clean, isolated environment on a local Windows machine.

### Prerequisites
- **Git**: Ensure Git is installed ([git-scm.com](https://git-scm.com/)).
- **Python 3.11**: Install Python 3.11.9. 
  - *PowerShell Installer (Recommended)*: Open powershell as admin and run:
    ```powershell
    winget install --id Python.Python.3.11 --exact --silent --accept-package-agreements
    ```
  - *Manual Installer*: Download from [python.org](https://www.python.org/downloads/release/python-3119/), selecting the **"Add Python to PATH"** checkbox before installing.

### Setup Instructions

1.  **Clone the Project**:
    ```bash
    git clone https://github.com/AshikMadhu/facial-expression-detection.git
    cd facial-expression-detection
    ```
2.  **Initialize Virtual Environment**:
    Create a localized environment named `venv311`:
    ```bash
    py -3.11 -m venv venv311
    ```
3.  **Activate Environment**:
    ```powershell
    # In Windows PowerShell:
    .\venv311\Scripts\Activate.ps1
    ```
4.  **Install Library Requirements**:
    ```powershell
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

---

## 🎮 7. Running the Project

All execution modes are centrally launched via the `run.py` launcher script:

### A. Webcam Mode
Launches the zero-latency local OpenCV display window. Shows facial bounding boxes, real-time classifications, and inference latency.
```bash
python run.py webcam
```
*Keyboard Shortcuts inside Webcam window:*
- **`q`**: Quit stream.
- **`f`**: Toggle Fullscreen mode.

### B. Dashboard Mode
Starts the local Streamlit server for interactive visualization of session records and metrics.
```bash
python run.py dashboard
```
Open your browser and navigate to: `http://localhost:8501`.

### C. Evaluation Mode
Runs model validation on the test partition, outputting logs and charts to `evaluation_results/`.
```bash
python run.py evaluate
```

### D. Training Mode
Executes the transfer learning and fine-tuning pipelines using raw data configurations.
```bash
python run.py train
```

---

## 📊 8. Dataset Information

The system uses the **FER2013** dataset (Facial Expression Recognition 2013).
- **Format**: `fer2013.csv` contains pixel values represented as space-separated string integers.
- **Image Dimensions**: $48 \times 48$ pixels in grayscale.
- **Categories (7 Classes)**:
  - `0`: Angry
  - `1`: Disgust
  - `2`: Fear
  - `3`: Happy
  - `4`: Sad
  - `5`: Surprise
  - `6`: Neutral
- **Preprocessing Pipeline**:
  - Drops rows with invalid shapes or out-of-bound class values.
  - Normalizes grayscale values to range $[0.0, 1.0]$.
  - Resizes frames dynamically from $48 \times 48$ to $160 \times 160$ to fit input requirements of EfficientNetV2.

---

## 🤖 9. Model Architecture

EmotionSense AI employs a two-stage Transfer Learning strategy based on **EfficientNetV2-B0**:

1.  **Feature Extraction (Phase 1)**:
    - Base convolutional blocks of EfficientNetV2 are **frozen**.
    - The top classifier layers (Global Average Pooling, Dropout, Dense classification) are trained with a learning rate of $1\times 10^{-3}$ for 10 epochs.
2.  **Fine-Tuning (Phase 2)**:
    - Base blocks from layer index 135 onwards are **unfrozen**.
    - Weights are optimized using Cosine Learning Rate Decay starting at $1\times 10^{-4}$ for 25 epochs.
3.  **Loss Function**:
    - Uses **Categorical Focal Loss** with gamma $\gamma = 2.0$ and label smoothing $0.1$ to mitigate class imbalances.

---

## 📈 10. Analytics Engine

The background analytics compiler processes raw predictions to output session metrics:
- **Emotion Timeline**: Tracks the frequency and distribution of classifications over time.
- **Valence Index**: Measures net positivity:
  $$\text{Valence} = \frac{\text{Happy} - (\text{Angry} + \text{Sad} + \text{Fear} + \text{Disgust})}{\text{Total Frame Count}}$$
  Values range between $-1.0$ (highly negative) and $+1.0$ (highly positive).
- **Attention Score**: Calculates engagement using MediaPipe gaze vectors:
  - Focus is flagged as high if gaze is centered within typical focal coordinates.
  - Distractions are marked if gaze coordinates drift beyond focal bounds.

---

## 🔍 11. Troubleshooting

### 1. "No suitable Python runtime found"
- **Cause**: Python 3.11 is missing or not bound to the `py` launcher.
- **Fix**: Re-run the installation command with `--exact` and ensure Python is added to the system PATH environment variable.

### 2. "Failed to open video capture device"
- **Cause**: Webcam is unplugged or busy.
- **Fix**: Ensure camera is plugged in. Change camera ID index inside `src/realtime_webcam.py` on line 18 (e.g., set to `1` or `2` if using external USB webcams).

### 3. OpenCV Window Fails to Open (`cv2.error`)
- **Cause**: Headless package installed by mistake.
- **Fix**: Run `pip uninstall opencv-python-headless` and then `pip install opencv-python` to restore GUI window configurations.

---

## ⚙️ 12. Verification & Diagnostics

Centralized verification commands to confirm system integrity:

```bash
# 1. Run environment verification script
python scripts/verify_installation.py

# 2. Run system hardware diagnostics
python scripts/system_diagnostics.py

# 3. Run unit test suite
python -m unittest tests/test_suite.py
```

---

## 🏆 13. Performance Summary

- **Accuracy**: Base classification accuracy on FER2013 achieves approximately **64% - 66%** on test sets.
- **CPU Inference Latency**: ~150 ms per frame on standard Intel/AMD processors (optimized via frame skipping and resizing).
- **Known Limitations**: Gaze detection accuracy may degrade under low light or severe head rotations.

---

## 👤 14. Contributors
- **Ashik Madhu** - *Principal Developer* - [GitHub Profile](https://github.com/AshikMadhu)

---

## 📄 15. License
This project is licensed under the MIT License - see the LICENSE file for details.
