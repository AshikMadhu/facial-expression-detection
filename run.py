import sys
import os
import subprocess
import importlib

# Resolve absolute path of src/ directory
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))

# Programmatically append src/ to the system search path to resolve relative imports
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

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
    dashboard_path = os.path.join(src_dir, "dashboard.py")
    if not os.path.exists(dashboard_path):
        print(f"[Launcher] Error: Dashboard script not found at {dashboard_path}", file=sys.stderr)
        sys.exit(1)
        
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
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
