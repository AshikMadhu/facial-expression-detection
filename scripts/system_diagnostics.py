import os
import sys
import platform
import multiprocessing
from pathlib import Path

def run_diagnostics():
    print("EmotionSense AI System Diagnostics")
    print("===================================")
    
    # Platform Details
    print(f"OS Platform      : {platform.system()} {platform.release()} (version {platform.version()})")
    print(f"Architecture     : {platform.machine()}")
    print(f"Processor        : {platform.processor()}")
    print(f"CPU Cores (Total): {multiprocessing.cpu_count()}")
    
    # Python Details
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version   : {sys.version}")
    
    # Directory Mapping
    project_root = Path(__file__).resolve().parent.parent
    print(f"Project Root Path: {project_root}")
    
    # Dependencies
    print("\nPackage Diagnostics:")
    
    # TensorFlow
    try:
        import tensorflow as tf
        print(f"  TensorFlow version: {tf.__version__}")
        
        # GPU detection
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  GPU Acceleration  : DETECTED")
            for gpu in gpus:
                print(f"    - Device: {gpu}")
        else:
            print(f"  GPU Acceleration  : NOT DETECTED (Running on CPU)")
    except Exception as e:
        print(f"  TensorFlow: FAILED to load ({str(e)})")
        
    # Keras
    try:
        import keras
        print(f"  Keras version     : {keras.__version__}")
    except Exception as e:
        print(f"  Keras: FAILED to load ({str(e)})")
        
    # OpenCV
    try:
        import cv2
        print(f"  OpenCV version    : {cv2.__version__}")
        try:
            gui_supported = hasattr(cv2, "namedWindow") and hasattr(cv2, "imshow")
            print(f"  OpenCV GUI support: {'YES' if gui_supported else 'NO'}")
        except Exception:
            print(f"  OpenCV GUI support: UNKNOWN ERROR")
    except Exception as e:
        print(f"  OpenCV: FAILED to load ({str(e)})")
        
    # MediaPipe
    try:
        import mediapipe as mp
        print(f"  MediaPipe version : {getattr(mp, '__version__', 'unknown')}")
    except Exception as e:
        print(f"  MediaPipe: FAILED to load ({str(e)})")
        
    # Streamlit
    try:
        import streamlit as st
        print(f"  Streamlit version : {st.__version__}")
    except Exception as e:
        print(f"  Streamlit: FAILED to load ({str(e)})")
        
    # Workspace Data Validation
    print("\nWorkspace Resource Verification:")
    paths = {
        "Dataset (fer2013.csv)": project_root / "data" / "fer2013.csv",
        "CNN Model Weights (best_model.h5)": project_root / "models" / "best_model.h5",
        "Face Detector Model (blaze_face_short_range.tflite)": project_root / "models" / "blaze_face_short_range.tflite"
    }
    
    for name, p in paths.items():
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [FOUND] {name}: size {size_mb:.2f} MB ({p})")
        else:
            print(f"  [MISSING] {name} at {p}")
            
    print("\nDiagnostics execution completed.")

if __name__ == "__main__":
    run_diagnostics()
