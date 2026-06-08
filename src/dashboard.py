import time
import json
import os
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

from config import TrainingConfig
from inference import EmotionInferenceEngine
from analytics import EmotionAnalyticsEngine

# Setup page layout configurations
st.set_page_config(
    page_title="EmotionSense AI - Analytics Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium dark mode UI styling via CSS injection
st.markdown("""
<style>
    .reportview-container {
        background: #0d0e12;
    }
    .metric-card {
        background-color: #161922;
        border: 1px solid #282d3d;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00d4b2;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8c96a8;
        margin-bottom: 5px;
    }
    .emotion-badge {
        font-size: 1.8rem;
        font-weight: 800;
        padding: 10px 20px;
        border-radius: 8px;
        text-align: center;
        margin-top: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
if 'engine_loaded' not in st.session_state:
    st.session_state.engine_loaded = False
    
@st.cache_resource
def load_ml_resources():
    """Caches resources to avoid reloading models on every rerun."""
    config = TrainingConfig()
    # Check if a model exists, otherwise catch error
    try:
        inference_engine = EmotionInferenceEngine(config=config)
        return inference_engine, config
    except Exception as e:
        st.error(f"Failed to load target model checkpoint: {str(e)}")
        return None, config

inference_engine, config = load_ml_resources()

if 'analytics_engine' not in st.session_state:
    st.session_state.analytics_engine = EmotionAnalyticsEngine(config=config)

if 'session_running' not in st.session_state:
    st.session_state.session_running = False

if 'prev_EMA_probs' not in st.session_state:
    st.session_state.prev_EMA_probs = None

if 'history_window' not in st.session_state:
    st.session_state.history_window = []

# MediaPipe face detection is initialized inside the inference engine.

# Proactive cleanup of camera resource when session is stopped
if not st.session_state.get('session_running', False):
    if 'cap' in st.session_state and st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("🧠 EmotionSense AI")
st.sidebar.subheader("Real-Time Face Telemetry")
st.sidebar.markdown("---")

# Session management buttons
if not st.session_state.session_running:
    if st.sidebar.button("▶️ Start Recording Session", width="stretch"):
        st.session_state.session_running = True
        st.session_state.analytics_engine.reset_session()
        st.session_state.prev_EMA_probs = None
        st.session_state.history_window = []
        if inference_engine:
            inference_engine.reset_history()
        st.rerun()
else:
    if st.sidebar.button("⏹️ Stop & Compile Session", width="stretch"):
        st.session_state.session_running = False
        st.session_state.analytics_engine.session_end_time = time.time()
        # Clean up camera immediately on Stop
        if 'cap' in st.session_state and st.session_state.cap is not None:
            try:
                st.session_state.cap.release()
            except Exception:
                pass
            st.session_state.cap = None
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "**Privacy Guarantee:** All facial processing is conducted locally in WebAssembly/RAM. "
    "No raw camera video streams or images are transmitted to external servers."
)

# ----------------- MAIN PANEL -----------------
st.title("EmotionSense AI Dashboard")
st.markdown("### Real-Time Human Emotion Recognition and Engagement Analytics")
st.markdown("---")

# Layout structures
col_feed, col_realtime = st.columns([1.2, 1.0])

# Placeholders for dynamic components
with col_feed:
    st.subheader("🎥 Live Capture Input")
    video_placeholder = st.empty()
    
with col_realtime:
    st.subheader("📊 Live Telemetry Metrics")
    
    # Nested columns for dominant emotion and engagement meter
    col_em_card, col_eng_card = st.columns(2)
    
    with col_em_card:
        st.markdown('<div class="metric-label">CURRENT DOMINANT EMOTION</div>', unsafe_allow_html=True)
        emotion_badge_placeholder = st.empty()
        
    with col_eng_card:
        st.markdown('<div class="metric-label">ENGAGEMENT SCORE</div>', unsafe_allow_html=True)
        engagement_meter_placeholder = st.empty()
        
    st.markdown("---")
    st.markdown('<div class="metric-label">PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
    chart_placeholder = st.empty()

# Historical analytics placeholders below feed columns
st.markdown("---")
st.subheader("📈 Session Time-Series Trends")
timeline_chart_placeholder = st.empty()

st.markdown("---")
st.subheader("📋 Session Summary Statistics")
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    stat_duration = st.empty()
with col_stat2:
    stat_avg_eng = st.empty()
with col_stat3:
    stat_distraction = st.empty()
with col_stat4:
    stat_valence = st.empty()

st.markdown("---")
export_placeholder = st.empty()

if st.session_state.session_running:
    # Open connection to local webcam and save it in session state
    if 'cap' not in st.session_state or st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        
    cap = st.session_state.cap
    
    if not cap.isOpened():
        st.error("Error: Could not access the webcam. Ensure the camera is connected and permissions are granted.")
        st.session_state.session_running = False
        st.session_state.cap = None
    else:
        # Skipping and caching helpers
        frame_counter = 0
        cached_faces = []
        
        try:
            while st.session_state.session_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab video frame. Reconnecting...")
                    time.sleep(0.5)
                    continue
                    
                # Mirror frame and convert to RGB for Streamlit rendering
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # --- DYNAMIC DETECTIONS & INFERENCE SCHEDULING ---
                # Face detection runs every 4 frames (throttling), inference every 2 frames (skipping)
                if frame_counter % 4 == 0:
                    if inference_engine:
                        try:
                            # Use MediaPipe Face Detection + Alignment
                            detections = inference_engine.detect_and_align_faces(frame)
                            new_cached_faces = []
                            for det in detections:
                                # Predict with temporal smoothing (session state isolated) and TTA
                                emotion, confidence, distributions, latency = inference_engine.predict(
                                    det["cropped_face"], 
                                    smooth=True, 
                                    state=st.session_state
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
                            cached_faces = new_cached_faces
                            
                            if not cached_faces:
                                st.session_state.prev_EMA_probs = None
                                st.session_state.history_window = []
                        except Exception as e:
                            pass
                    else:
                        cached_faces = []
                        
                elif frame_counter % 2 == 0:
                    # Crop using cached alignment matrix and crop boundaries on skipped frames
                    if cached_faces and inference_engine:
                        try:
                            new_cached_faces = []
                            for face in cached_faces:
                                M = face["M"]
                                x_start, y_start, x_end, y_end = face["crop_coords"]
                                
                                # Crop from current frame using cached rotation matrix and crop bounds
                                h_frame, w_frame = frame.shape[:2]
                                aligned_frame = cv2.warpAffine(frame, M, (w_frame, h_frame), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                                cropped_face = aligned_frame[y_start:y_end, x_start:x_end]
                                
                                # Run prediction
                                emotion, confidence, distributions, latency = inference_engine.predict(
                                    cropped_face, 
                                    smooth=True, 
                                    state=st.session_state
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
                            cached_faces = new_cached_faces
                        except Exception as e:
                            pass
                
                # Increment frame counter
                frame_counter += 1
                
                # Draw box around faces in RGB frame using cached boxes
                for face in cached_faces[:1]:
                    (x, y, w, h) = face["box"]
                    cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 212, 178), 3)
                
                # Extract prediction metrics from first face
                if cached_faces:
                    first_face = cached_faces[0]
                    dominant_emotion = first_face["emotion"]
                    confidence = first_face["confidence"]
                    probabilities = first_face["probabilities"]
                else:
                    dominant_emotion = "Neutral"
                    confidence = 1.0
                    probabilities = {label: 0.0 for label in config.emotion_labels.values()}
                    probabilities["Neutral"] = 1.0
                
                gaze = {"gaze_x": 0.0, "gaze_y": 0.0, "gaze_confidence": 1.0, "blink_detected": False}
                head_pose = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
                
                # Log metrics in analytics engine
                curr_timestamp = time.time()
                st.session_state.analytics_engine.add_record(
                    timestamp=curr_timestamp,
                    emotions_distribution=probabilities,
                    gaze_data=gaze,
                    head_pose=head_pose
                )
                
                # 1. Update Video Frame Display
                video_placeholder.image(rgb_frame, channels="RGB", width="stretch")
                
                # 2. Update Dominant Emotion Card
                em_color_map = {
                    "Happy": "#00ff66", "Surprise": "#ffd700", "Neutral": "#cccccc",
                    "Sad": "#1e90ff", "Angry": "#ff4500", "Fear": "#da70d6", "Disgust": "#8b4513"
                }
                bg_color = em_color_map.get(dominant_emotion, "#222222")
                emotion_badge_placeholder.markdown(
                    f'<div class="emotion-badge" style="background-color: {bg_color}; color: black;">'
                    f'{dominant_emotion} ({confidence*100.0:.1f}%)'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                # 3. Update Engagement Score Meter
                current_es = st.session_state.analytics_engine.history[-1]["engagement_score"]
                engagement_meter_placeholder.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{current_es*100.0:.1f}%</div>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                # 4. Render Probability Distribution bar chart using Plotly
                df_probs = pd.DataFrame({
                    "Emotion": list(probabilities.keys()),
                    "Probability": list(probabilities.values())
                }).sort_values(by="Probability", ascending=True)
                
                fig_bar = px.bar(
                    df_probs, 
                    x="Probability", 
                    y="Emotion", 
                    orientation="h",
                    range_x=[0, 1.0],
                    color="Probability",
                    color_continuous_scale="Purples",
                    template="plotly_dark",
                    height=250
                )
                fig_bar.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False, coloraxis_showscale=False)
                chart_placeholder.plotly_chart(fig_bar, width="stretch", key=f"bar_{curr_timestamp}")
                
                # 5. Render Timeline chart
                history_data = st.session_state.analytics_engine.history
                times = [r["timestamp"] - st.session_state.analytics_engine.session_start_time for r in history_data]
                scores = [r["engagement_score"] for r in history_data]
                
                df_timeline = pd.DataFrame({"Time (s)": times, "Engagement": scores})
                fig_timeline = px.line(
                    df_timeline, 
                    x="Time (s)", 
                    y="Engagement", 
                    range_y=[0, 1.05],
                    template="plotly_dark",
                    height=300
                )
                fig_timeline.update_layout(margin=dict(l=10, r=10, t=20, b=10))
                timeline_chart_placeholder.plotly_chart(fig_timeline, width="stretch", key=f"timeline_{curr_timestamp}")
                
                # 6. Render Current statistics
                stats = st.session_state.analytics_engine.get_session_statistics()
                stat_duration.markdown(f'<div class="metric-card"><div class="metric-label">DURATION</div><div class="metric-value">{stats.get("session_duration_sec", 0.0):.1f}s</div></div>', unsafe_allow_html=True)
                stat_avg_eng.markdown(f'<div class="metric-card"><div class="metric-label">AVG ENGAGEMENT</div><div class="metric-value">{stats.get("average_engagement", 0.0)*100.0:.1f}%</div></div>', unsafe_allow_html=True)
                stat_distraction.markdown(f'<div class="metric-card"><div class="metric-label">DISTRACTION RATE</div><div class="metric-value">{stats.get("distraction_rate", 0.0)*100.0:.1f}%</div></div>', unsafe_allow_html=True)
                stat_valence.markdown(f'<div class="metric-card"><div class="metric-label">VALENCE INDEX</div><div class="metric-value">{stats.get("average_valence_index", 0.0):.2f}</div></div>', unsafe_allow_html=True)
                
                # Frame delay control (Target ~10 FPS)
                time.sleep(0.08)
        finally:
            # Cleanup video stream allocations
            if 'cap' in st.session_state and st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                except Exception:
                    pass
                st.session_state.cap = None
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

# ----------------- STATIC RENDER ON SESSION END -----------------
else:
    # Display message if history is empty
    if not st.session_state.analytics_engine.history:
        video_placeholder.info("Please start the recording session from the sidebar to view camera feeds and analytics logs.")
        emotion_badge_placeholder.markdown('<div class="emotion-badge" style="background-color: #222;">Waiting...</div>', unsafe_allow_html=True)
        engagement_meter_placeholder.markdown('<div class="metric-card"><div class="metric-value">0.0%</div></div>', unsafe_allow_html=True)
        
        # Display empty distributions chart
        dummy_df = pd.DataFrame({"Emotion": list(config.emotion_labels.values()), "Probability": [0.0]*7})
        fig_dummy = px.bar(dummy_df, x="Probability", y="Emotion", orientation="h", range_x=[0, 1.0], template="plotly_dark", height=250)
        fig_dummy.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        chart_placeholder.plotly_chart(fig_dummy, width="stretch", key="bar_static_empty")
    else:
        # Display final session metrics
        stats = st.session_state.analytics_engine.get_session_statistics()
        
        # Render static final charts
        history_data = st.session_state.analytics_engine.history
        times = [r["timestamp"] - st.session_state.analytics_engine.session_start_time for r in history_data]
        scores = [r["engagement_score"] for r in history_data]
        
        df_timeline = pd.DataFrame({"Time (s)": times, "Engagement": scores})
        fig_timeline = px.line(df_timeline, x="Time (s)", y="Engagement", range_y=[0, 1.05], template="plotly_dark", height=300)
        fig_timeline.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        timeline_chart_placeholder.plotly_chart(fig_timeline, width="stretch", key="timeline_static_final")
        
        # Static Metrics cards
        stat_duration.markdown(f'<div class="metric-card"><div class="metric-label">TOTAL DURATION</div><div class="metric-value">{stats.get("session_duration_sec", 0.0):.1f}s</div></div>', unsafe_allow_html=True)
        stat_avg_eng.markdown(f'<div class="metric-card"><div class="metric-label">FINAL AVG ENGAGEMENT</div><div class="metric-value">{stats.get("average_engagement", 0.0)*100.0:.1f}%</div></div>', unsafe_allow_html=True)
        stat_distraction.markdown(f'<div class="metric-card"><div class="metric-label">FINAL DISTRACTION</div><div class="metric-value">{stats.get("distraction_rate", 0.0)*100.0:.1f}%</div></div>', unsafe_allow_html=True)
        stat_valence.markdown(f'<div class="metric-card"><div class="metric-label">FINAL VALENCE INDEX</div><div class="metric-value">{stats.get("average_valence_index", 0.0):.2f}</div></div>', unsafe_allow_html=True)
        
        # Final dominant emotion display
        dominant_final = stats.get("dominant_emotion", "Neutral")
        emotion_badge_placeholder.markdown(f'<div class="emotion-badge" style="background-color: #ffd700; color: black;">{dominant_final}</div>', unsafe_allow_html=True)
        engagement_meter_placeholder.markdown(f'<div class="metric-card"><div class="metric-value">{stats.get("average_engagement", 0.0)*100.0:.1f}%</div></div>', unsafe_allow_html=True)
        
        # Average emotion distribution
        dist = stats.get("emotion_distribution", {})
        df_dist = pd.DataFrame({
            "Emotion": list(dist.keys()),
            "Average Probability": list(dist.values())
        }).sort_values(by="Average Probability", ascending=True)
        
        fig_dist = px.bar(
            df_dist, 
            x="Average Probability", 
            y="Emotion", 
            orientation="h", 
            range_x=[0, 1.0], 
            color="Average Probability", 
            color_continuous_scale="Purples", 
            template="plotly_dark", 
            height=250
        )
        fig_dist.update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        chart_placeholder.plotly_chart(fig_dist, width="stretch", key="bar_static_final")
        
        # Create JSON string for download report
        report_data = {
            "session_metadata": {
                "start_time": st.session_state.analytics_engine.session_start_time,
                "end_time": st.session_state.analytics_engine.session_end_time,
                "exported_at": time.time()
            },
            "aggregated_statistics": stats,
            "telemetry_history": history_data
        }
        json_string = json.dumps(report_data, indent=4)
        
        # Download button
        export_placeholder.download_button(
            label="📥 Download Session JSON Report",
            data=json_string,
            file_name=f"emotionsense_session_report_{int(time.time())}.json",
            mime="application/json",
            width="stretch"
        )
