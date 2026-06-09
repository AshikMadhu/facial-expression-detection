# Local Troubleshooting Guide

This guide covers common issues and resolutions when running **EmotionSense AI** locally.

---

## 🎥 1. Webcam / Video Stream Issues

### Issue: `Failed to open video capture device with ID: 0`
*   **Cause**: The application cannot access your webcam.
*   **Solutions**:
    1.  **Permissions**: Ensure that webcam permissions are enabled in your OS (Settings -> Privacy & Security -> Camera on Windows/macOS).
    2.  **Device ID**: If you have multiple webcams (or virtual cameras like OBS), try changing the camera ID. In `src/realtime_webcam.py`, change `camera_id` (default `0`) to `1` or `2`.
    3.  **Conflict**: Make sure no other application (Zoom, Teams, Discord, OBS) is currently using the camera.

### Issue: OpenCV window throws `not implemented` error
*   **Cause**: `opencv-python-headless` is installed instead of `opencv-python`. Headless OpenCV lacks the GUI modules needed to draw windows on your screen.
*   **Solution**:
    ```bash
    pip uninstall opencv-python-headless
    pip install opencv-python==4.8.0.76
    ```

---

## 🧠 2. TensorFlow & Model Loading Issues

### Issue: `Trained model file NOT found at ...`
*   **Cause**: The inference engine cannot find `best_model.h5` inside the `models/` directory.
*   **Solution**:
    1.  Generate a mock dataset and run training to generate the model:
        ```bash
        python src/generate_mock_dataset.py
        python run.py train
        ```
    2.  Alternatively, make sure you have downloaded or copied `best_model.h5` into the `models/` folder.

### Issue: `AttributeError` or `ValueError` during model load
*   **Cause**: Model weights were saved using a different version of TensorFlow or Keras.
*   **Solution**:
    We recommend using **Python 3.10 / 3.11** with the pinned requirements in `requirements.txt` (`tensorflow==2.15.0` and `keras==2.15.0`) to avoid serialization format conflicts.

---

## 📊 3. Streamlit Dashboard Issues

### Issue: `ModuleNotFoundError: No module named 'streamlit'`
*   **Cause**: Streamlit was not installed, or your virtual environment is not active.
*   **Solution**:
    Activate your virtual environment and install requirements:
    ```bash
    # Windows
    .\venv\Scripts\Activate.ps1
    # macOS/Linux
    source venv/bin/activate
    
    pip install -r requirements.txt
    ```

### Issue: Streamlit camera feed is slow or lagging
*   **Cause**: The local OpenCV frame loop runs in a single thread, and Streamlit reruns scripts on state updates.
*   **Solution**:
    Close the Streamlit dashboard and use the native OpenCV mode, which is much faster and offers superior performance:
    ```bash
    python run.py webcam
    ```
