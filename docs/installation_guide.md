# EmotionSense AI: Installation & Setup Guide

This guide details the step-by-step instructions to configure, build, and run the EmotionSense AI platform on local machines and servers.

---

## 1. System Requirements & Prerequisites

### 1.1 Supported Operating Systems
*   **Windows**: Windows 10/11 (64-bit). WSL2 is recommended but native PowerShell is fully supported.
*   **macOS**: macOS Big Sur 11.0 or newer (supports Apple Silicon M1/M2/M3 natively via WebNN/Metal).
*   **Linux**: Ubuntu 20.04 LTS or newer (highly optimized for headless Docker runs).

### 1.2 Core Dependencies
*   **Python (Version 3.10 to 3.12)**: *Note: Python 3.14 (installed on the current runner) is not yet supported by official TensorFlow releases. To run model training and local inference with GPU acceleration, you must install Python 3.11.*
*   **OpenCV System Prerequisites**:
    *   *Windows*: Requires Visual C++ Redistributable.
    *   *Linux*: Requires `libgl1-mesa-glx` and `libglib2.0-0` (fully handled by our Docker configuration).
*   **Webcam Hardware**: An integrated or external USB webcam (for real-time capture).

---

## 2. Local Installation Setup

Follow these commands to install the project from your terminal:

### Step 2.1: Clone the Repository & Open Workspace
```bash
git clone https://github.com/yourusername/emotionsense-ai.git
cd emotionsense-ai
```

### Step 2.2: Create a Dedicated Virtual Environment (Forced to Python 3.11)
Using a virtual environment prevents dependency conflicts. Enforce Python 3.11 during creation:

**On Windows (PowerShell):**
```powershell
# Select Python 3.11 specifically to support TensorFlow
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
# Select Python 3.11 specifically to support TensorFlow
python3.11 -m venv venv
source venv/bin/activate
```

### Step 2.3: Install Production Packages
Install packages using the root manifest file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Verifying the Installation

To verify that the ML library modules and graphics backends are correctly configured, run the automated test suite:

```bash
python -m unittest packages/ml-models/tests/test_suite.py
```

All integration, unit, and validation throughput checks must return `OK`.

---

## 4. Troubleshooting & FAQ

### Q1: "ERROR: No matching distribution found for tensorflow"
*   **Cause**: You are running a Python version (e.g., Python 3.14) for which TensorFlow has not compiled stable wheels yet.
*   **Solution**: Install Python 3.11 alongside your existing version, and initialize your virtual environment using that specific binary:
    ```bash
    py -3.11 -m venv venv
    ```

### Q2: "cv2.error: OpenCV(4.x) ... -215:Assertion failed" (Webcam Error)
*   **Cause**: OpenCV is trying to open camera index `0` but the hardware is either unplugged, blocked by another process (like Zoom or Teams), or requires privacy access permissions.
*   **Solution**:
    1.  Ensure all other camera-dependent apps are completely closed.
    2.  Check OS camera privacy settings and ensure "Allow apps to access your camera" is enabled.
    3.  If using an external USB camera, change the camera ID parameter in `realtime_webcam.py` from `0` to `1` or `2`.

### Q3: "Out of Memory (OOM) Errors" during training
*   **Cause**: GPU VRAM is overloaded by a large batch size.
*   **Solution**: Reduce `batch_size` in `packages/ml-models/src/config.py` from `128` to `64` or `32`.
