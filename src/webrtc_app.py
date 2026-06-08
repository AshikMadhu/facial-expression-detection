import streamlit as st
import cv2
import numpy as np
import av
import time
import logging
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration, WebRtcMode

# Configure logger
logger = logging.getLogger("WebRTCVideoProcessor")

class EmotionVideoProcessor(VideoProcessorBase):
    """Asynchronous WebRTC Video Processor for real-time face tracking and emotion recognition."""
    
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.frame_counter = 0
        self.cached_faces = []
        self.latest_predictions = None
        
        # Thread-safe prediction history isolated to this processor stream instance
        self.state_dict = {
            "prev_EMA_probs": None,
            "history_window": []
        }
        
        # BGR Color Scheme for overlays matching local webcam
        self.emotion_colors = {
            "Angry": (0, 0, 255),       # Red
            "Disgust": (0, 75, 150),    # Brownish Orange
            "Fear": (255, 0, 255),      # Magenta
            "Happy": (0, 255, 0),       # Green
            "Sad": (255, 0, 0),         # Blue
            "Surprise": (0, 255, 255),   # Yellow
            "Neutral": (200, 200, 200)  # Light Grey
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Processes video frame from browser WebRTC stream and burns telemetry overlays."""
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # Mirror frame horizontally so the display matches standard mirror camera preview
        img = cv2.flip(img, 1)
        
        # --- PERFORMANCE OPTIMIZATION: DYNAMIC SCHEDULING ---
        # Face detection runs every 4 frames (throttling), inference runs every 2 frames (skipping)
        if self.frame_counter % 4 == 0:
            try:
                # Use MediaPipe Face Detection + Alignment
                detections = self.inference_engine.detect_and_align_faces(img)
                new_cached_faces = []
                for det in detections:
                    # Predict with temporal smoothing (passing stream state_dict)
                    emotion, confidence, distributions, latency = self.inference_engine.predict(
                        det["cropped_face"], 
                        smooth=True, 
                        state=self.state_dict
                    )
                    new_cached_faces.append({
                        "box": det["bbox"],
                        "M": det["M"],
                        "crop_coords": det["crop_coords"],
                        "emotion": emotion,
                        "confidence": confidence,
                        "probabilities": distributions,
                        "latency": latency
                    })
                self.cached_faces = new_cached_faces
                
                # Reset history if face is lost
                if not self.cached_faces:
                    self.state_dict["prev_EMA_probs"] = None
                    self.state_dict["history_window"] = []
                    
            except Exception as e:
                logger.error(f"Error in WebRTC face detection: {str(e)}")
                
        elif self.frame_counter % 2 == 0:
            # Inference-only frame: crop using cached alignment parameters from last detection
            if self.cached_faces:
                try:
                    new_cached_faces = []
                    for face in self.cached_faces:
                        M = face["M"]
                        x_start, y_start, x_end, y_end = face["crop_coords"]
                        
                        # Affine crop from current frame
                        aligned_frame = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        cropped_face = aligned_frame[y_start:y_end, x_start:x_end]
                        
                        # Run prediction using the stream state_dict
                        emotion, confidence, distributions, latency = self.inference_engine.predict(
                            cropped_face, 
                            smooth=True, 
                            state=self.state_dict
                        )
                        new_cached_faces.append({
                            "box": face["box"],
                            "M": M,
                            "crop_coords": face["crop_coords"],
                            "emotion": emotion,
                            "confidence": confidence,
                            "probabilities": distributions,
                            "latency": latency
                        })
                    self.cached_faces = new_cached_faces
                except Exception as e:
                    logger.error(f"Error in WebRTC skipped inference: {str(e)}")
                    
        self.frame_counter += 1
        
        # --- RENDERING TELEMETRY OVERLAYS ---
        for face in self.cached_faces:
            (x, y, bw, bh) = face["box"]
            emotion = face["emotion"]
            confidence = face["confidence"]
            inf_latency = face.get("latency", 0.0)
            
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # 1. Bounding Box
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 3)
            
            # 2. Text label background box
            text_label = f"{emotion}: {confidence*100.0:.1f}%"
            label_size, base_line = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                img, 
                (x, y - label_size[1] - 8), 
                (x + label_size[0] + 10, y), 
                color, 
                cv2.FILLED
            )
            
            # 3. Text label
            cv2.putText(
                img, 
                text_label, 
                (x + 5, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0) if emotion in ["Neutral", "Surprise", "Happy"] else (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
            
            # 4. Latency text below box
            cv2.putText(
                img, 
                f"inf: {inf_latency:.1f}ms", 
                (x, y + bh + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                color, 
                1, 
                cv2.LINE_AA
            )
            
        # Update thread-safe prediction cache for the main thread to pull
        if self.cached_faces:
            first_face = self.cached_faces[0]
            self.latest_predictions = {
                "emotion": first_face["emotion"],
                "confidence": first_face["confidence"],
                "probabilities": first_face["probabilities"],
                "timestamp": time.time()
            }
        else:
            self.latest_predictions = {
                "emotion": "Neutral",
                "confidence": 1.0,
                "probabilities": {label: 0.0 for label in self.emotion_colors.keys()},
                "timestamp": time.time()
            }
            self.latest_predictions["probabilities"]["Neutral"] = 1.0
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def render_webrtc_view(inference_engine):
    """Renders the WebRTC streamer component and returns the active connection context."""
    # STUN server configuration for NAT traversal on public network
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    ctx = webrtc_streamer(
        key="emotionsense-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=lambda: EmotionVideoProcessor(inference_engine),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    return ctx
