import time
import logging
import cv2
import numpy as np

from config import TrainingConfig
from inference import EmotionInferenceEngine

# Configure logger
logger = logging.getLogger("RealtimeWebcamEngine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class RealtimeEmotionDetector:
    """Orchestrates OpenCV video streams with facial detection and emotion inference."""

    def __init__(self, config: TrainingConfig = None, camera_id: int = 0):
        self.config = config or TrainingConfig()
        self.camera_id = camera_id
        
        # 1. Initialize inference engine
        self.inference_engine = EmotionInferenceEngine(config=self.config)
        
        # 2. MediaPipe Face Detector is initialized inside self.inference_engine.
        logger.info("MediaPipe Face Detector initialized via Inference Engine.")

        # 3. Define color scheme for UI bounding boxes per emotion
        # BGR Format for OpenCV
        self.emotion_colors = {
            "Angry": (0, 0, 255),       # Red
            "Disgust": (0, 75, 150),    # Brownish Orange
            "Fear": (255, 0, 255),      # Magenta
            "Happy": (0, 255, 0),       # Green
            "Sad": (255, 0, 0),         # Blue
            "Surprise": (0, 255, 255),   # Yellow
            "Neutral": (200, 200, 200)  # Light Grey
        }

    def start_detection_loop(self):
        """Starts real-time webcam stream, processes frame detections, and renders UI overlays."""
        # Initialize video capture stream
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open video capture device with ID: {self.camera_id}")
            return
            
        logger.info(f"Successfully connected to webcam. Starting display window...")
        logger.info("Press 'q' key in the video frame window to exit.")
        
        # Performance calculation helpers
        prev_time = time.perf_counter()
        frame_count = 0
        fps = 0.0
        
        # Frame count for skipping
        loop_counter = 0
        cached_faces = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Grabbed blank frame. Exiting loop.")
                    break
                    
                frame_count += 1
                
                # --- PERFORMANCE OPTIMIZATION: IMAGE RESIZING ---
                # Downscale frame width to 640px to accelerate face detection latency
                h, w = frame.shape[:2]
                target_width = 640
                scale = target_width / float(w)
                target_height = int(h * scale)
                resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                
                # Convert to grayscale for Haar Cascade
                # Reset history on session start (once at start)
                if loop_counter == 0:
                    self.inference_engine.reset_history()

                # --- PERFORMANCE OPTIMIZATION: DYNAMIC DETECTIONS ---
                # 1. Face detection runs every 4 frames (throttling)
                # 2. Inference runs every 2 frames (skipping)
                # On non-detection inference frames, crop the face using cached M and crop boundaries.
                if loop_counter % 4 == 0:
                    try:
                        # Use MediaPipe Face Detection + Alignment on the resized frame
                        detections = self.inference_engine.detect_and_align_faces(resized_frame)
                        
                        new_cached_faces = []
                        for det in detections:
                            # Predict emotions with temporal smoothing
                            emotion, confidence, distributions, inf_latency = self.inference_engine.predict(det["cropped_face"], smooth=True)
                            new_cached_faces.append({
                                "box": det["bbox"],
                                "M": det["M"],
                                "crop_coords": det["crop_coords"],
                                "emotion": emotion,
                                "confidence": confidence,
                                "latency": inf_latency
                            })
                        cached_faces = new_cached_faces
                        
                        if len(detections) == 0:
                            self.inference_engine.reset_history()
                            
                    except Exception as e:
                        logger.error(f"Failed to run detection/classification: {str(e)}")
                            
                elif loop_counter % 2 == 0:
                    # Inference only frame: crop using cached alignment parameters
                    if cached_faces:
                        try:
                            new_cached_faces = []
                            for face in cached_faces:
                                M = face["M"]
                                x_start, y_start, x_end, y_end = face["crop_coords"]
                                
                                # Crop from current frame using cached rotation matrix and crop bounds
                                h_frame, w_frame = resized_frame.shape[:2]
                                aligned_frame = cv2.warpAffine(resized_frame, M, (w_frame, h_frame), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                                cropped_face = aligned_frame[y_start:y_end, x_start:x_end]
                                
                                # Run prediction
                                emotion, confidence, distributions, inf_latency = self.inference_engine.predict(cropped_face, smooth=True)
                                new_cached_faces.append({
                                    "box": face["box"],
                                    "M": M,
                                    "crop_coords": face["crop_coords"],
                                    "emotion": emotion,
                                    "confidence": confidence,
                                    "latency": inf_latency
                                })
                            cached_faces = new_cached_faces
                        except Exception as e:
                            logger.error(f"Failed to run skipped inference: {str(e)}")

                # Increment loop counter
                loop_counter += 1
    
                # --- RENDERING OVERLAY GRAPHICS ---
                for face_data in cached_faces:
                    (x, y, w_face, h_face) = face_data["box"]
                    emotion = face_data["emotion"]
                    confidence = face_data["confidence"]
                    inf_latency = face_data["latency"]
                    
                    color = self.emotion_colors.get(emotion, (255, 255, 255))
                    
                    # 1. Draw bounding box around face
                    cv2.rectangle(resized_frame, (x, y), (x + w_face, y + h_face), color, 2)
                    
                    # 2. Draw semi-transparent background box for text overlay
                    text_label = f"{emotion}: {confidence*100.0:.1f}%"
                    label_size, base_line = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        resized_frame, 
                        (x, y - label_size[1] - 8), 
                        (x + label_size[0] + 10, y), 
                        color, 
                        cv2.FILLED
                    )
                    
                    # 3. Render emotion text label
                    cv2.putText(
                        resized_frame, 
                        text_label, 
                        (x + 5, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 0) if emotion in ["Neutral", "Surprise", "Happy"] else (255, 255, 255), 
                        1, 
                        cv2.LINE_AA
                    )
                    
                    # 4. Display inference execution latency below box
                    cv2.putText(
                        resized_frame, 
                        f"inf: {inf_latency:.1f}ms", 
                        (x, y + h_face + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, 
                        color, 
                        1, 
                        cv2.LINE_AA
                    )
    
                # --- RENDER DASHBOARD METRICS ---
                # Calculate FPS
                curr_time = time.perf_counter()
                elapsed_time = curr_time - prev_time
                if elapsed_time >= 0.5:  # Update FPS value every 0.5s
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    prev_time = curr_time
                    
                # Draw overlay banner for system metrics
                cv2.rectangle(resized_frame, (0, 0), (resized_frame.shape[1], 35), (20, 20, 20), cv2.FILLED)
                cv2.putText(
                    resized_frame, 
                    f"EmotionSense AI System Dashboard  |  FPS: {fps:.1f}  |  Resolution: {resized_frame.shape[1]}x{resized_frame.shape[0]}", 
                    (10, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )
    
                # Show image frame
                cv2.imshow("EmotionSense AI - Real-time Webcam Recognition", resized_frame)
                
                # Listen for close conditions ('q' key or window closed)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User requested termination. Exiting...")
                    break
        finally:
            # Cleanup video stream allocations
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Video streams successfully closed.")

def main():
    # Launch real-time detector
    config = TrainingConfig()
    try:
        detector = RealtimeEmotionDetector(config=config)
        detector.start_detection_loop()
    except Exception as e:
        logger.error(f"Failed to start camera loop: {str(e)}")
        print("Webcam loop failed. Verify that models and cameras are plugged in.")

if __name__ == "__main__":
    main()
