# Onboarding & Installation Manual

Welcome to **EmotionSense AI**! This document provides step-by-step instructions to set up the project locally on your machine.

---

## 📋 Prerequisites

Before starting, ensure you have the following installed:
*   **Python**: Version **3.10.x** or **3.11.x** (3.11 is recommended for local execution).
*   **Git**: For cloning the repository.
*   **Webcam**: An integrated or USB-connected webcam.

---

## 🛠️ Step-by-Step Local Setup

### 1. Clone the Repository
Open a terminal (Command Prompt/PowerShell on Windows, or terminal on macOS/Linux) and run:
```bash
git clone https://github.com/AshikMadhu/facial-expression-detection.git
cd facial-expression-detection
```

### 2. Create a Virtual Environment
Isolating dependencies ensures a clean run and prevents version conflicts.

*   **Windows (PowerShell)**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
*   **Windows (Command Prompt)**:
    ```cmd
    python -m venv venv
    .\venv\Scripts\activate.bat
    ```
*   **macOS / Linux**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3. Install Dependencies
Install all requirements from the root directory:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> If you have an NVIDIA GPU and wish to enable GPU acceleration, install the GPU version of TensorFlow:
> ```bash
> pip install tensorflow[and-cuda]==2.15.0
> ```
> (Note: Standard `tensorflow==2.15.0` will automatically run in CPU mode if no compatible GPU or CUDA drivers are present).

### 4. Create Mock Dataset (For quick testing/validation)
If you do not have the full FER2013 dataset, you can generate a mock dataset to verify pipeline execution:
```bash
python src/generate_mock_dataset.py
```
This will create a `data/fer2013.csv` file representing the expected schema of the dataset.

---

## 🚀 Running the Project

Use the central wrapper `run.py` to trigger different system execution modes:

1.  **OpenCV Webcam Window**:
    ```bash
    python run.py webcam
    ```
    *(Press **`q`** to quit, or **`f`** to toggle fullscreen).*
    
2.  **Streamlit Dashboard**:
    ```bash
    python run.py dashboard
    ```
    *(This starts a local Streamlit web server and opens the dashboard in your default browser).*

3.  **Pipeline Training**:
    ```bash
    python run.py train
    ```
    *(Trains the EfficientNetV2 transfer learning model).*

4.  **Evaluation Dashboards**:
    ```bash
    python run.py evaluate
    ```
    *(Generates report text and plots in `evaluation_results/`).*
