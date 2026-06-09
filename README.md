# EmotionSense AI: Real-Time Human Emotion Recognition & Engagement Analytics Platform

EmotionSense AI is an industry-grade, real-time emotion recognition and engagement analytics platform. It leverages a local computer vision pipeline where face detection and emotion classification run completely locally on the client's machine (CPU/GPU) to guarantee absolute privacy, low latency, and ease of deployment.

---

## 📂 Project Directory Structure

```
emotion/
├── run.py                      # Central CLI Launcher
├── requirements.txt            # Local Python dependencies
├── README.md                   # Main documentation
├── .gitignore                  # Git exclusions
│
├── src/                        # Source codebase
│   ├── config.py               # Central Path & Hyperparameter Config
│   ├── inference.py            # Preprocessing & Inference Engine
│   ├── realtime_webcam.py      # OpenCV Webcam Frame Loop
│   ├── dashboard.py            # Streamlit Analytics Dashboard
│   ├── model.py                # EfficientNetV2 Model Definition
│   ├── train.py                # Two-Stage Model Trainer
│   ├── evaluate.py             # Evaluation Reports Generator
│   ├── fer2013_pipeline.py     # Data pipeline & augmentation layers
│   ├── dataset_validator.py    # CSV schema and pixel validator
│   ├── generate_mock_dataset.py# Utility to generate simulated data
│   └── analytics.py            # Metrics, Valence & Attention logic
│
├── models/                     # Deep learning models
│   ├── best_model.h5           # Converted Keras HDF5 model
│   └── blaze_face_short_range.tflite # MediaPipe Face Detector
│
├── tests/                      # Testing frameworks
│   └── test_suite.py           # Unit, Integration, and Latency tests
│
├── docs/                       # Guides and Technical Manuals
│   ├── installation_guide.md   # Step-by-step local setup
│   ├── troubleshooting_guide.md # Local execution debugging tips
│   └── project_architecture_summary.md # System architecture details
│
└── evaluation_results/         # Evaluation dumps & curves (Git ignored)
```

---

## 🛠️ Quick Start Guide

For full installation details, see the [Installation Guide](docs/installation_guide.md).

### 1. Environment Setup
Create a virtual environment and install all dependencies:
```bash
python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Test Suite
Ensure everything is correctly set up by running the test suite:
```bash
python -m unittest tests/test_suite.py
```

### 3. Generate Mock Data (Optional)
Generate a mock FER2013 dataset if you want to test the data pipelines:
```bash
python src/generate_mock_dataset.py
```

### 4. Running the Code Modes

Use `run.py` to launch the different system modules:

*   **Interactive Streamlit Dashboard**:
    ```bash
    python run.py dashboard
    ```
*   **High-Performance Webcam GUI (OpenCV Window)**:
    ```bash
    python run.py webcam
    ```
    *(Resizes dynamically. Press **`f`** to toggle fullscreen, and **`q`** or close the window to exit).*
*   **Model Training & Fine-Tuning**:
    ```bash
    python run.py train
    ```
*   **Evaluation Report & Loss Curves generation**:
    ```bash
    python run.py evaluate
    ```

---

## 🛡️ Privacy & Biometric Security
EmotionSense AI enforces a strict **Local Processing Policy**. Raw camera video streams are captured and processed purely in volatile local memory (RAM) and are discarded immediately after face alignment and emotion classification. No images are written to disk or sent to external servers.
