# EmotionSense AI: Real-Time Human Emotion Recognition & Engagement Analytics Platform

EmotionSense AI is an industry-grade, real-time emotion recognition and engagement analytics platform. It leverages a hybrid edge-cloud architecture where computer vision operations (face detection and emotion classification) run locally on the client's device (edge) via WebAssembly and ONNX Runtime to guarantee absolute privacy, low latency, and zero server GPU costs. Telemetry and aggregated analytics are summarized on a premium dark-themed Streamlit dashboard.

---

## 📂 Repository Structure

The workspace follows a clean, modular monorepo layout:

```
emotion/
├── packages/
│   └── ml-models/                      # Core Machine Learning & Logic Package
│       ├── requirements.txt            # Python dependencies lists
│       ├── src/                        # Source files
│       │   ├── config.py               # Central Training & Path Configurations
│       │   ├── fer2013_pipeline.py     # Data Pipeline: cleaning, validation, TF.data streams
│       │   ├── model.py                # Model: MobileNetV2 Transfer & Fine-tuning blocks
│       │   ├── train.py                # Trainer: Two-stage compile and fit callback triggers
│       │   ├── evaluate.py             # Evaluator: Conf-matrix, error logs, metrics dashboard
│       │   ├── inference.py            # Inference: Preprocessing and prediction API
│       │   ├── analytics.py            # Analytics: Gaze tracking, head pose, valence & engagement score
│       │   └── dashboard.py            # Streamlit UI: Interactive live charts & webcam logs
│       └── tests/                      # Automated Test Suites
│           └── test_suite.py           # Unit, Integration, Model, and Performance tests
└── README.md                           # Main setup documentation
```

---

## ⚡ Quick Start Guide

### 1. Environment Setup
Install all Python dependencies inside a virtual environment:

```bash
# Clone/Open workspace and navigate to the ML package
cd packages/ml-models
pip install -r requirements.txt
```

### 2. Run the Automated Test Suite
Verify that all unit logic, validation throughput limits, and integration report builders pass by executing the complete test suite:

```bash
# Execute tests from workspace root (C:\Users\ASHIK\Desktop\emotion)
python -m unittest packages/ml-models/tests/test_suite.py
```

### 3. Training & Fine-Tuning the Model
1. Download the raw FER2013 dataset (CSV format) from Kaggle.
2. Place the file at `data/fer2013.csv` (relative to project root).
3. Trigger the two-stage training execution pipeline (Phase 1: Feature Extraction on frozen base, Phase 2: Fine-Tuning top base blocks via Cosine Decay scheduling):

```bash
python packages/ml-models/src/train.py
```
*Trained model binaries and weights will be saved to `packages/ml-models/models/` and `packages/ml-models/checkpoints/`.*

### 4. Running Model Evaluation
After training, compile performance metrics, list high-confidence prediction errors, and export the multi-panel evaluation metrics dashboard (`evaluation_dashboard.png`):

```bash
python packages/ml-models/src/evaluate.py
```

### 5. Running Real-Time Webcam Inference (Console Frame View)
Execute the real-time opencv console loop to see color-coded bounding boxes and active FPS stats rendered live:

```bash
python packages/ml-models/src/realtime_webcam.py
```
*Press the **`q`** key inside the active frame window to close the video capture.*

### 6. Launching the Interactive Streamlit Dashboard
Launch the dashboard to record sessions, track attention/distraction levels, plot timeline charts, and download serialized JSON session reports:

```bash
streamlit run packages/ml-models/src/dashboard.py
```

---

## 🛡️ Privacy & Biometric Security
EmotionSense AI enforces a strict **Zero-Image-Retention** policy. Raw images from webcam streams are processed purely in volatile local memory (RAM) and are discarded immediately after emotion classification. Only low-bandwidth, non-biometric numerical metadata (probabilities, pitch/yaw, and timestamps) are processed for session analytics.
