import sys
from pathlib import Path
import subprocess
import importlib

# Resolve absolute path of src/ directory
src_dir = Path(__file__).resolve().parent / "src"

# Programmatically append src/ to the system search path to resolve relative imports
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def launch_module(module_name: str):
    """Dynamically imports and runs the specified module's main loop."""
    try:
        print(f"[Launcher] Initializing execution of module '{module_name}'...")
        # Import the module
        module = importlib.import_module(module_name)
        
        # Run main function
        if hasattr(module, "main"):
            module.main()
        else:
            print(f"[Launcher] Error: Module '{module_name}' does not define a main() function.", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"[Launcher] Exception while executing '{module_name}': {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def launch_dashboard():
    """Launches the Streamlit dashboard as a subprocess to prevent lifecycle conflicts."""
    dashboard_path = Path(__file__).resolve().parent / "src" / "dashboard.py"
    if not dashboard_path.exists():
        print(f"[Launcher] Error: Dashboard script not found at {dashboard_path}", file=sys.stderr)
        sys.exit(1)
        
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
    # Pass along any additional arguments
    cmd.extend(sys.argv[2:])
    
    print("[Launcher] Starting Streamlit server...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[Launcher] Dashboard server terminated by user request.")
    except subprocess.CalledProcessError as e:
        print(f"[Launcher] Dashboard server exited with error code: {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

def validate_environment(mode: str):
    """Validates model and data paths before executing the requested mode."""
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    data_dir = project_root / "data"
    
    # 1. Create required output directories automatically
    required_dirs = [
        project_root / "evaluation_results",
        project_root / "logs",
        project_root / "checkpoints",
        data_dir / "reports"
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    # 2. Check model files if required by mode
    model_modes = ["webcam", "dashboard", "evaluate"]
    if mode in model_modes:
        best_model = models_dir / "best_model.h5"
        blazeface = models_dir / "blaze_face_short_range.tflite"
        
        missing = []
        if not best_model.exists():
            missing.append("models/best_model.h5 (trained CNN model weights)")
        if not blazeface.exists():
            missing.append("models/blaze_face_short_range.tflite (face detection model)")
            
        if missing:
            print("[Startup Validator] ERROR: Critical model checkpoint file(s) missing:", file=sys.stderr)
            for m in missing:
                print(f"  - {m}", file=sys.stderr)
            print("\nPlease ensure model checkpoints are downloaded/placed in 'models/' directory.", file=sys.stderr)
            print("To train a new model from scratch, run: python run.py train", file=sys.stderr)
            sys.exit(1)
            
    # 3. Check dataset file if required by mode
    dataset_modes = ["train", "evaluate"]
    if mode in dataset_modes:
        fer_csv = data_dir / "fer2013.csv"
        if not fer_csv.exists():
            print("[Startup Validator] ERROR: FER2013 dataset file missing!", file=sys.stderr)
            print(f"  Expected path: {fer_csv}", file=sys.stderr)
            print("\nPlease place 'fer2013.csv' in the 'data/' folder to run training or evaluation.", file=sys.stderr)
            sys.exit(1)
            
    print(f"[Startup Validator] Environment checks passed for mode '{mode}'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("EmotionSense AI Production Launcher")
        print("===================================")
        print("Usage: python run.py <mode> [additional args]")
        print("Available modes:")
        print("  dashboard  : Launches interactive Streamlit web dashboard")
        print("  webcam     : Launches local OpenCV real-time webcam feed GUI")
        print("  evaluate   : Executes test dataset evaluation and dumps reports")
        print("  train      : Starts model training pipeline")
        sys.exit(1)
        
    mode = sys.argv[1]
    
    # Run environment checks first
    validate_environment(mode)
    
    # Mapping modes to scripts/launchers
    if mode == "dashboard":
        launch_dashboard()
    elif mode == "webcam":
        launch_module("realtime_webcam")
    elif mode == "evaluate":
        launch_module("evaluate")
    elif mode == "train":
        launch_module("train")
    else:
        print(f"[Launcher] Unknown mode: {mode}", file=sys.stderr)
        print("Supported modes: dashboard | webcam | evaluate | train", file=sys.stderr)
        sys.exit(1)
