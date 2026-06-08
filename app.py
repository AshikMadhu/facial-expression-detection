import sys
import os

# Append src/ to the system search path to resolve cross-imports within the src directory
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import time
import json
import streamlit as st
import pandas as pd
import plotly.express as px

from config import TrainingConfig
from inference import EmotionInferenceEngine
from analytics import EmotionAnalyticsEngine
from webrtc_app import render_webrtc_view

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

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("🧠 EmotionSense AI")
st.sidebar.subheader("Real-Time Face Telemetry")
st.sidebar.markdown("---")

# Session reset button
if st.sidebar.button("🔄 Reset Analytics Session", width="stretch"):
    st.session_state.analytics_engine.reset_session()
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
    ctx = render_webrtc_view(inference_engine)
    
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

# ----------------- DYNAMIC EXECUTION CONTROL -----------------
if ctx.state.playing:
    # Stream is active: dynamic polling of WebRTC processor queue
    if not st.session_state.session_running:
        st.session_state.session_running = True
        st.session_state.analytics_engine.reset_session()
        
    if ctx.video_processor:
        latest = ctx.video_processor.latest_predictions
        if latest:
            dominant_emotion = latest["emotion"]
            confidence = latest["confidence"]
            probabilities = latest["probabilities"]
            
            # Log metrics in analytics engine
            curr_timestamp = latest["timestamp"]
            gaze = {"gaze_x": 0.0, "gaze_y": 0.0, "gaze_confidence": 1.0, "blink_detected": False}
            head_pose = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
            st.session_state.analytics_engine.add_record(
                timestamp=curr_timestamp,
                emotions_distribution=probabilities,
                gaze_data=gaze,
                head_pose=head_pose
            )
            
            # 1. Update Dominant Emotion Card
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
            
            # 2. Update Engagement Score Meter
            current_es = st.session_state.analytics_engine.history[-1]["engagement_score"]
            engagement_meter_placeholder.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{current_es*100.0:.1f}%</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
            
            # 3. Render Probability Distribution bar chart using Plotly
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
            
            # 4. Render Timeline chart
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
            
            # 5. Render Current statistics
            stats = st.session_state.analytics_engine.get_session_statistics()
            stat_duration.markdown(f'<div class="metric-card"><div class="metric-label">DURATION</div><div class="metric-value">{stats.get("session_duration_sec", 0.0):.1f}s</div></div>', unsafe_allow_html=True)
            stat_avg_eng.markdown(f'<div class="metric-card"><div class="metric-label">AVG ENGAGEMENT</div><div class="metric-value">{stats.get("average_engagement", 0.0)*100.0:.1f}%</div></div>', unsafe_allow_html=True)
            stat_distraction.markdown(f'<div class="metric-card"><div class="metric-label">DISTRACTION RATE</div><div class="metric-value">{stats.get("distraction_rate", 0.0)*100.0:.1f}%</div></div>', unsafe_allow_html=True)
            stat_valence.markdown(f'<div class="metric-card"><div class="metric-label">VALENCE INDEX</div><div class="metric-value">{stats.get("average_valence_index", 0.0):.2f}</div></div>', unsafe_allow_html=True)
            
    # Rerun the Streamlit app to poll new frames
    time.sleep(0.1)
    st.rerun()

else:
    # Stream is inactive (Stopped or not started yet)
    st.session_state.session_running = False
    
    if not st.session_state.analytics_engine.history:
        # No history: app just loaded
        st.info("Click the 'Start' button above on the camera stream to start recording session analytics.")
        emotion_badge_placeholder.markdown('<div class="emotion-badge" style="background-color: #222; color: #888;">Waiting...</div>', unsafe_allow_html=True)
        engagement_meter_placeholder.markdown('<div class="metric-card"><div class="metric-value">0.0%</div></div>', unsafe_allow_html=True)
        
        # Display empty distributions chart
        dummy_df = pd.DataFrame({"Emotion": list(config.emotion_labels.values()), "Probability": [0.0]*7})
        fig_dummy = px.bar(dummy_df, x="Probability", y="Emotion", orientation="h", range_x=[0, 1.0], template="plotly_dark", height=250)
        fig_dummy.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        chart_placeholder.plotly_chart(fig_dummy, width="stretch", key="bar_static_empty")
    else:
        # History exists: user just stopped the stream. Render final metrics.
        stats = st.session_state.analytics_engine.get_session_statistics()
        
        # Static Timeline chart
        history_data = st.session_state.analytics_engine.history
        times = [r["timestamp"] - st.session_state.analytics_engine.session_start_time for r in history_data]
        scores = [r["engagement_score"] for r in history_data]
        df_timeline = pd.DataFrame({"Time (s)": times, "Engagement": scores})
        fig_timeline = px.line(df_timeline, x="Time (s)", y="Engagement", range_y=[0, 1.05], template="plotly_dark", height=300)
        fig_timeline.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        timeline_chart_placeholder.plotly_chart(fig_timeline, width="stretch", key="timeline_static_final")
        
        # Final dominant emotion display
        dominant_final = stats.get("dominant_emotion", "Neutral")
        emotion_badge_placeholder.markdown(f'<div class="emotion-badge" style="background-color: #ffd700; color: black;">{dominant_final}</div>', unsafe_allow_html=True)
        engagement_meter_placeholder.markdown(f'<div class="metric-card"><div class="metric-value">{stats.get("average_engagement", 0.0)*100.0:.1f}%</div></div>', unsafe_allow_html=True)
        
        # Static Metrics cards
        stat_duration.markdown(f'<div class="metric-card"><div class="metric-label">TOTAL DURATION</div><div class="metric-value">{stats.get("session_duration_sec", 0.0):.1f}s</div></div>', unsafe_allow_html=True)
        stat_avg_eng.markdown(f'<div class="metric-card"><div class="metric-label">FINAL AVG ENGAGEMENT</div><div class="metric-value">{stats.get("average_engagement", 0.0)*100.0:.1f}%</div></div>', unsafe_allow_html=True)
        stat_distraction.markdown(f'<div class="metric-card"><div class="metric-label">FINAL DISTRACTION</div><div class="metric-value">{stats.get("distraction_rate", 0.0)*100.0:.1f}%</div></div>', unsafe_allow_html=True)
        stat_valence.markdown(f'<div class="metric-card"><div class="metric-label">FINAL VALENCE INDEX</div><div class="metric-value">{stats.get("average_valence_index", 0.0):.2f}</div></div>', unsafe_allow_html=True)
        
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
