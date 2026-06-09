import time
import logging
from typing import Dict, Tuple, Union, List
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import TrainingConfig

# Configure logger
logger = logging.getLogger("FER2013Inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class EmotionInferenceEngine:
    """Reusable Inference Engine for real-time and batch facial emotion classification."""

    def __init__(self, model_path: Union[str, Path] = None, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.model_path = Path(model_path or self.config.saved_model_path)
        
        # Log hardware capabilities
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"TensorFlow GPU acceleration detected. Running on: {len(gpus)} GPU(s)")
        else:
            logger.info("TensorFlow GPU acceleration not detected. Running on CPU mode.")
            
        # Load label definitions
        self.labels = [self.config.emotion_labels[i] for i in sorted(self.config.emotion_labels.keys())]
        
        # Load model into memory (using compile=False for robust loading without custom loss/optimizer issues)
        self.model = self._load_model()
        
        # Initialize MediaPipe Tasks Face Detector using the TFLite model asset
        model_file_path = Path(self.config.project_root) / 'models' / 'blaze_face_short_range.tflite'
        if not model_file_path.exists():
            # Fallback path just in case
            model_file_path = Path(__file__).resolve().parent.parent / 'models' / 'blaze_face_short_range.tflite'
            
        if not model_file_path.exists():
            raise FileNotFoundError(f"MediaPipe Face Detector model file not found at: {model_file_path}")
            
        logger.info(f"Initializing MediaPipe Face Detector from: {model_file_path}...")
        base_options = python.BaseOptions(model_asset_path=str(model_file_path))
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.face_detector = vision.FaceDetector.create_from_options(options)
        
        # State for temporal smoothing
        self.prev_probs = None
        self.history_window = []

    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_detector'):
            try:
                self.face_detector.close()
            except Exception:
                pass

    def _load_model(self) -> tf.keras.Model:
        """Loads the saved Keras model checkpoint."""
        if not self.model_path.exists():
            error_msg = (
                f"\n[Model Load Error] Trained model file NOT found at: {self.model_path.resolve()}\n"
                "========================================================================\n"
                "To resolve this, please choose one of these options:\n"
                "  1. Run training to generate a new model: python run.py train\n"
                "  2. Download or copy 'best_model.h5' and place it inside the 'models/' folder.\n"
                "========================================================================\n"
            )
            logger.error(error_msg)
            raise FileNotFoundError(f"Model file missing: {self.model_path}")
            
        logger.info(f"Initializing model load from: {self.model_path}...")
        start_time = time.perf_counter()
        
        # Force channels_last for standard inference layout compatibility
        tf.keras.backend.set_image_data_format('channels_last')
        
        # Load with compile=False to avoid dependency on Custom Loss/Optimizer during inference
        model = tf.keras.models.load_model(str(self.model_path), compile=False)
        
        elapsed = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"Model successfully loaded in {elapsed:.2f} ms.")
        return model

    def reset_history(self):
        """Clears the temporal smoothing history."""
        self.prev_probs = None
        self.history_window = []

    def detect_and_align_faces(self, frame: np.ndarray) -> List[dict]:
        """
        Uses MediaPipe Tasks Face Detection to locate faces in a BGR frame,
        applies eye alignment affine transform, and returns a list of dicts with:
        - cropped_face (np.ndarray)
        - bbox (Tuple[int, int, int, int])
        - M (np.ndarray)
        - crop_coords (Tuple[int, int, int, int])
        """
        h, w = frame.shape[:2]
        
        # Convert BGR frame to mp.Image as expected by Tasks API
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run detection
        results = self.face_detector.detect(mp_image)
        
        detections = []
        if not results or not results.detections:
            return detections
            
        for detection in results.detections:
            # 1. Align face using eye keypoints
            keypoints = detection.keypoints
            if not keypoints or len(keypoints) < 2:
                continue
                
            right_eye_norm = keypoints[0]  # Person's right eye (viewer's left)
            left_eye_norm = keypoints[1]   # Person's left eye (viewer's right)
            
            right_eye = (right_eye_norm.x * w, right_eye_norm.y * h)
            left_eye = (left_eye_norm.x * w, left_eye_norm.y * h)
            
            # Calculate alignment angle
            dy = left_eye[1] - right_eye[1]
            dx = left_eye[0] - right_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Rotate around eye center
            eye_center = ((right_eye[0] + left_eye[0]) / 2.0, (right_eye[1] + left_eye[1]) / 2.0)
            M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
            
            aligned_frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            # 2. Crop face using bounding box coordinates
            bbox = detection.bounding_box
            bx = bbox.origin_x
            by = bbox.origin_y
            bw = bbox.width
            bh = bbox.height
            
            # Pad bounding box to ensure ears/chin are included
            pad_w = int(bw * 0.1)
            pad_h = int(bh * 0.1)
            
            x_start = max(0, bx - pad_w)
            y_start = max(0, by - pad_h)
            x_end = min(w, bx + bw + pad_w)
            y_end = min(h, by + bh + pad_h)
            
            cropped_face = aligned_frame[y_start:y_end, x_start:x_end]
            
            detections.append({
                "cropped_face": cropped_face,
                "bbox": (bx, by, bw, bh),
                "M": M,
                "crop_coords": (x_start, y_start, x_end, y_end)
            })
            
        return detections

    def preprocess_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocesses raw input into a normalized grayscale array.
        Supported formats: File path (str), Byte array (bytes), PIL Image, or NumPy array.
        Outputs dimensions: (1, 48, 48, 1), normalized to [0.0, 1.0].
        """
        # 1. Convert various input types to a PIL Image object
        if isinstance(image_input, str):
            # Read from file path
            img = Image.open(image_input)
        elif isinstance(image_input, bytes):
            # Read from byte buffer
            import io
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, np.ndarray):
            # Check if array is already processed
            if image_input.size > 0 and image_input.shape == (self.config.raw_image_size, self.config.raw_image_size, 1):
                # Normalize and add batch dimension
                x = image_input.astype(np.float32)
                if np.max(x) > 1.0:
                    x /= 255.0
                return np.expand_dims(x, axis=0)
            
            # Check for empty ROI (safety fallbacks)
            if image_input.size == 0 or image_input.shape[0] == 0 or image_input.shape[1] == 0:
                image_input = np.zeros((self.config.raw_image_size, self.config.raw_image_size), dtype=np.uint8)
                
            img = Image.fromarray(image_input)
        else:
            raise TypeError("Unsupported image input type. Use file path, bytes, PIL Image, or NumPy array.")

        # 2. Conversion to Grayscale ('L' mode)
        if img.mode != 'L':
            img = img.convert('L')
            
        # 3. Resize to model's base expected resolution (48x48)
        img = img.resize((self.config.raw_image_size, self.config.raw_image_size), Image.Resampling.BILINEAR)
        
        # 4. Convert to float32 NumPy array and scale pixels to [0.0, 1.0]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # 5. Expand dimensions to (1, 48, 48, 1) to represent Batch size of 1 and 1 channel
        img_array = np.expand_dims(img_array, axis=-1)  # (48, 48, 1)
        img_array = np.expand_dims(img_array, axis=0)   # (1, 48, 48, 1)
        
        return img_array

    def predict(self, 
                image_input: Union[str, bytes, np.ndarray, Image.Image], 
                smooth: bool = False, 
                beta: float = 0.7,
                state: dict = None,
                use_tta: bool = True) -> Tuple[str, float, Dict[str, float], float]:
        """
        Executes inference on a single image.
        Supports session-specific state dictionary to ensure thread/session safety.
        Returns:
            dominant_emotion (str): The label of the highest-confidence emotion.
            confidence_score (float): The probability of the dominant emotion.
            probabilities (Dict[str, float]): A dictionary mapping labels to float probabilities.
            inference_latency (float): Execution time in milliseconds.
        """
        # 1. Preprocess input
        processed_img = self.preprocess_image(image_input)
        
        # 2. Run inference and time execution speed (with horizontal-flip TTA)
        start_time = time.perf_counter()
        prediction_probs = self.model.predict(processed_img, verbose=0)[0]
        
        if use_tta:
            # Flip horizontally along the width dimension (axis=2 for shape [1, 48, 48, 1])
            processed_flipped = np.flip(processed_img, axis=2)
            prediction_probs_flipped = self.model.predict(processed_flipped, verbose=0)[0]
            prediction_probs = 0.5 * (prediction_probs + prediction_probs_flipped)
            
        latency = (time.perf_counter() - start_time) * 1000.0
        
        # 3. Apply Hybrid Temporal Smoothing (EMA beta=0.7 + Sliding Window Voting N=5)
        if smooth:
            if state is not None:
                prev_EMA_probs = state.get("prev_EMA_probs", None)
                history_window = state.get("history_window", [])
            else:
                prev_EMA_probs = self.prev_probs
                history_window = self.history_window
                
            # Update EMA
            if prev_EMA_probs is None:
                prev_EMA_probs = prediction_probs
            else:
                prev_EMA_probs = beta * prev_EMA_probs + (1.0 - beta) * prediction_probs
            EMA_probs = prev_EMA_probs
            
            # Update sliding window voting history
            raw_dominant_idx = int(np.argmax(prediction_probs))
            history_window.append(raw_dominant_idx)
            if len(history_window) > 5:
                history_window.pop(0)
                
            # Compute voting probability distribution
            window_vote_probs = np.zeros_like(prediction_probs)
            for idx in history_window:
                window_vote_probs[idx] += 1.0
            window_vote_probs /= len(history_window)
            
            # Combine EMA and window votes
            prediction_probs = 0.5 * (EMA_probs + window_vote_probs)
            
            # Save state back
            if state is not None:
                state["prev_EMA_probs"] = prev_EMA_probs
                state["history_window"] = history_window
            else:
                self.prev_probs = prev_EMA_probs
                self.history_window = history_window
            
        # 4. Apply Per-class Calibration and Adaptive Fallback
        thresholds = {
            "Happy": 0.55,
            "Surprise": 0.50,
            "Angry": 0.45,
            "Sad": 0.35,
            "Fear": 0.30,
            "Disgust": 0.25
        }
        
        # Find which non-neutral classes exceed their threshold
        valid_classes = {}
        for c, th in thresholds.items():
            c_idx = self.labels.index(c)
            prob = float(prediction_probs[c_idx])
            if prob >= th:
                valid_classes[c] = prob
                
        if valid_classes:
            # Predict the valid class with the highest probability
            dominant_emotion = max(valid_classes, key=valid_classes.get)
            dominant_idx = self.labels.index(dominant_emotion)
            confidence_score = float(prediction_probs[dominant_idx])
        else:
            # Adaptive fallback to Neutral when all classes are uncertain
            dominant_emotion = "Neutral"
            neutral_idx = self.labels.index("Neutral")
            confidence_score = float(prediction_probs[neutral_idx])
            
        # 5. Create probability map dictionary
        probabilities = {self.labels[i]: float(prob) for i, prob in enumerate(prediction_probs)}
        
        return dominant_emotion, confidence_score, probabilities, latency

if __name__ == "__main__":
    # Test class execution (uses simulated inputs)
    config = TrainingConfig()
    try:
        engine = EmotionInferenceEngine(config=config)
        # Create a dummy gray image array for simulation testing
        dummy_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        emotion, conf, probs, lat = engine.predict(dummy_image)
        print("Inference Test Run Success:")
        print(f"Emotion: {emotion} | Confidence: {conf:.4f} | Latency: {lat:.2f} ms")
    except Exception as e:
        logger.error(f"Inference run failed: {str(e)}")
        print("Setup incomplete. Ensure trained models exist in packages/ml-models/models/")
