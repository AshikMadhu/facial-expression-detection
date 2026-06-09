import sys
import importlib.metadata
from pathlib import Path

def verify():
    print("EmotionSense AI Installation Verifier")
    print("======================================")
    
    passed = True
    
    # 1. Check Python version
    python_ver = sys.version_info
    print(f"Python interpreter version: {sys.version} (Expected: 3.11.x)")
    if python_ver.major != 3 or python_ver.minor != 11:
        print("[FAIL] Python version is not 3.11.x!")
        passed = False
    else:
        print("[PASS] Python version is 3.11.x")
        
    # 2. Check key dependency versions
    expected_deps = {
        "tensorflow": "2.15.0",
        "keras": "2.15.0",
        "numpy": "1.26.4",
        "pandas": "2.1.4",
        "scikit-learn": "1.3.2",
        "mediapipe": "0.10.11",
        "pillow": "10.2.0",
        "streamlit": "1.58.0",
        "plotly": "5.15.0",
        "tornado": "6.5.6"
    }
    
    for dep, expected_v in expected_deps.items():
        try:
            installed_v = importlib.metadata.version(dep)
            if installed_v != expected_v:
                print(f"[WARN] {dep}: Installed version is {installed_v}, expected {expected_v}")
            else:
                print(f"[PASS] {dep} version matches {expected_v}")
        except importlib.metadata.PackageNotFoundError:
            print(f"[FAIL] {dep} is NOT installed!")
            passed = False
            
    # 3. Check opencv-python specifically (ensure GUI version is installed, not headless)
    try:
        cv_version = importlib.metadata.version("opencv-python")
        print(f"[PASS] opencv-python (GUI version) is installed: {cv_version}")
    except importlib.metadata.PackageNotFoundError:
        # Check if headless is installed instead
        try:
            headless_v = importlib.metadata.version("opencv-python-headless")
            print(f"[FAIL] Headless OpenCV ('opencv-python-headless') is installed instead of GUI version!")
        except importlib.metadata.PackageNotFoundError:
            print("[FAIL] opencv-python is NOT installed!")
        passed = False
        
    # 4. Check critical model files
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    best_model = models_dir / "best_model.h5"
    blazeface = models_dir / "blaze_face_short_range.tflite"
    
    if not best_model.exists():
        print(f"[FAIL] models/best_model.h5 is missing!")
        passed = False
    else:
        print("[PASS] models/best_model.h5 exists")
        
    if not blazeface.exists():
        print(f"[FAIL] models/blaze_face_short_range.tflite is missing!")
        passed = False
    else:
        print("[PASS] models/blaze_face_short_range.tflite exists")
        
    # 5. Summary
    print("\n--------------------------------------")
    if passed:
        print("VERIFICATION SUCCESSFUL: The environment is fully operational!")
        sys.exit(0)
    else:
        print("VERIFICATION FAILED: Please fix the issues listed above.")
        sys.exit(1)

if __name__ == "__main__":
    verify()
