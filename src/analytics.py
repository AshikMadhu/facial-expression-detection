import time
import json
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

from config import TrainingConfig

# Configure logger
logger = logging.getLogger("EmotionAnalyticsEngine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class EmotionAnalyticsEngine:
    """Processes, stores, and analyzes time-series emotion telemetry and calculates engagement scores."""

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        
        # Session State Containers
        self.history: List[Dict] = []
        self.session_start_time: float = time.time()
        self.session_end_time: Optional[float] = None
        
        # Engagement scoring weights
        self.w_valence = 0.3
        self.w_attention = 0.5
        self.w_distraction = 0.2

    def reset_session(self):
        """Resets the internal session state variables."""
        self.history = []
        self.session_start_time = time.time()
        self.session_end_time = None
        logger.info("Session analytics history reset.")

    def calculate_valence_index(self, emotions: Dict[str, float]) -> float:
        """
        Calculates the valence index mapping from [-1.0 (highly negative) to 1.0 (highly positive)].
        Positive: Happy, Surprise
        Negative: Sad, Angry, Fear, Disgust
        Neutral: 0.0 impact
        """
        pos = emotions.get("Happy", 0.0) + emotions.get("Surprise", 0.0)
        neg = (emotions.get("Sad", 0.0) + 
               emotions.get("Angry", 0.0) + 
               emotions.get("Fear", 0.0) + 
               emotions.get("Disgust", 0.0))
        
        # Valence index
        valence = pos - neg
        # Bound between -1.0 and 1.0
        return float(np.clip(valence, -1.0, 1.0))

    def calculate_attention_index(self, gaze: Optional[Dict], head_pose: Optional[Dict]) -> float:
        """
        Calculates attention level [0.0 to 1.0] based on gaze stability and head rotation.
        A deviation of head pitch/yaw/roll from 0 degrees reduces attention.
        """
        attention = 1.0
        
        # 1. Deduct based on head rotation deviation
        if head_pose:
            pitch = head_pose.get("pitch", 0.0)
            yaw = head_pose.get("yaw", 0.0)
            roll = head_pose.get("roll", 0.0)
            
            # Root-mean-square deviation (RMS)
            deviation = np.sqrt(pitch**2 + yaw**2 + roll**2)
            
            # Assume 40 degrees is max natural tilt; normalize deduction
            tilt_penalty = min(deviation / 40.0, 1.0)
            attention -= (tilt_penalty * 0.5)  # Max 50% deduction for head tilt
            
        # 2. Deduct based on gaze tracking confidence/stability
        if gaze:
            gaze_confidence = gaze.get("gaze_confidence", 1.0)
            # Low tracking confidence indicates blink or eye closures
            attention *= gaze_confidence
            
        return float(np.clip(attention, 0.0, 1.0))

    def calculate_distraction_flag(self, gaze: Optional[Dict]) -> float:
        """
        Determines if the user is distracted [0.0 or 1.0] based on eye-gaze coordinates.
        If gaze coordinates lie outside standard screen boundaries, distraction flag triggers.
        """
        if not gaze:
            return 0.0
            
        # Gaze coordinates typically mapped to [-1.0, 1.0] representing screen bounds
        gaze_x = gaze.get("gaze_x", 0.0)
        gaze_y = gaze.get("gaze_y", 0.0)
        
        # If gaze lies beyond 80% boundary limits, trigger distraction
        if abs(gaze_x) > 0.8 or abs(gaze_y) > 0.8:
            return 1.0
            
        # Check blink state
        if gaze.get("blink_detected", False):
            return 0.5  # Temporary half deduction for blinks
            
        return 0.0

    def compute_engagement_score(self, 
                                 emotions: Dict[str, float], 
                                 gaze: Optional[Dict] = None, 
                                 head_pose: Optional[Dict] = None) -> float:
        """
        Aggregates valence, attention, and distraction indices to compute the final
        Engagement Score (ES) bounded between [0.0, 1.0].
        """
        # Calculate individual indices
        v_index = (self.calculate_valence_index(emotions) + 1.0) / 2.0  # Map [-1, 1] -> [0, 1]
        a_index = self.calculate_attention_index(gaze, head_pose)
        d_index = self.calculate_distraction_flag(gaze)
        
        # Weighted aggregate formula
        es = (self.w_valence * v_index) + (self.w_attention * a_index) - (self.w_distraction * d_index)
        
        return float(np.clip(es, 0.0, 1.0))

    def add_record(self, 
                   timestamp: float, 
                   emotions_distribution: Dict[str, float], 
                   gaze_data: Optional[Dict] = None, 
                   head_pose: Optional[Dict] = None):
        """
        Adds a single raw frame metrics block to the session history.
        Computes engagement score and metadata inline.
        """
        # Extract dominant emotion
        dominant_emotion = max(emotions_distribution, key=emotions_distribution.get)
        confidence = emotions_distribution[dominant_emotion]
        
        # Compute real-time engagement score
        engagement_score = self.compute_engagement_score(emotions_distribution, gaze_data, head_pose)
        
        record = {
            "timestamp": timestamp,
            "emotions": emotions_distribution,
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "gaze": gaze_data,
            "head_pose": head_pose,
            "engagement_score": engagement_score
        }
        
        self.history.append(record)

    def get_emotion_history(self) -> List[Dict]:
        """Returns the chronological logs of all recorded session points."""
        return self.history

    def get_dominant_emotion(self) -> str:
        """Calculates the absolute dominant emotion throughout the entire session duration."""
        if not self.history:
            return "Neutral"
            
        # Tally occurrences of dominant emotions per record
        counts: Dict[str, int] = {}
        for rec in self.history:
            dom = rec["dominant_emotion"]
            counts[dom] = counts.get(dom, 0) + 1
            
        return max(counts, key=counts.get)

    def get_emotion_distribution(self) -> Dict[str, float]:
        """Calculates percentage distributions for all emotions across the session logs."""
        if not self.history:
            return {label: 0.0 for label in self.config.emotion_labels.values()}
            
        total_records = len(self.history)
        tally = {label: 0.0 for label in self.config.emotion_labels.values()}
        
        # Add probability vectors together
        for rec in self.history:
            for emotion, prob in rec["emotions"].items():
                tally[emotion] = tally.get(emotion, 0.0) + prob
                
        # Normalize sum of probabilities to yield average distribution percentage
        distribution = {emotion: float(prob_sum / total_records) for emotion, prob_sum in tally.items()}
        return distribution

    def get_confidence_tracking(self) -> Dict[str, float]:
        """Calculates session statistics concerning model classification confidence rates."""
        if not self.history:
            return {"mean_confidence": 0.0, "std_confidence": 0.0}
            
        confidences = [rec["confidence"] for rec in self.history]
        return {
            "mean_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
            "max_confidence": float(np.max(confidences)),
            "min_confidence": float(np.min(confidences))
        }

    def get_session_statistics(self) -> Dict:
        """
        Compiles structural session statistics summary including durations, peak metrics,
        valence indices, and distraction levels.
        """
        if not self.history:
            return {"status": "No data recorded"}
            
        end_time = self.session_end_time or time.time()
        duration_seconds = end_time - self.session_start_time
        
        # Calculations vectors
        engagements = [rec["engagement_score"] for rec in self.history]
        conf_stats = self.get_confidence_tracking()
        emotions_dist = self.get_emotion_distribution()
        dominant_emotion = self.get_dominant_emotion()
        
        # Peak engagement metric
        peak_idx = int(np.argmax(engagements))
        peak_engagement = float(engagements[peak_idx])
        peak_timestamp = self.history[peak_idx]["timestamp"]
        
        # Calculate total distraction rate
        distractions = [self.calculate_distraction_flag(rec["gaze"]) for rec in self.history]
        distraction_rate = float(np.mean([1.0 if d > 0.0 else 0.0 for d in distractions]))
        
        # Calculate Valence average
        valences = [self.calculate_valence_index(rec["emotions"]) for rec in self.history]
        avg_valence = float(np.mean(valences))
        
        stats = {
            "session_duration_sec": float(duration_seconds),
            "total_records": len(self.history),
            "dominant_emotion": dominant_emotion,
            "average_engagement": float(np.mean(engagements)),
            "peak_engagement": {
                "score": peak_engagement,
                "timestamp": peak_timestamp
            },
            "average_confidence": conf_stats["mean_confidence"],
            "distraction_rate": distraction_rate,
            "average_valence_index": avg_valence,
            "emotion_distribution": emotions_dist
        }
        return stats

    def end_session(self) -> Dict:
        """Marks active session as closed and compiles the final analytics report payload."""
        self.session_end_time = time.time()
        session_report = self.get_session_statistics()
        logger.info(f"Session ended. Duration: {session_report['session_duration_sec']:.2f} seconds.")
        return session_report

    def export_report_to_json(self, filepath: str):
        """Serializes and exports the full session history and aggregated stats to a JSON file."""
        report = {
            "session_metadata": {
                "start_time": self.session_start_time,
                "end_time": self.session_end_time or time.time(),
                "exported_at": time.time()
            },
            "aggregated_statistics": self.get_session_statistics(),
            "telemetry_history": self.history
        }
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Session analytics report exported to: {filepath}")

if __name__ == "__main__":
    # Test execution simulating mock telemetry data
    config = TrainingConfig()
    engine = EmotionAnalyticsEngine(config=config)
    
    # Simulate a 10-second capture session (1 data point per second)
    print("Simulating active metrics collection session...")
    t_start = time.time()
    for i in range(10):
        # Generate random emotions summing to 1.0
        rand_probs = np.random.dirichlet(np.ones(7))
        emotions_map = {config.emotion_labels[idx]: float(prob) for idx, prob in enumerate(rand_probs)}
        
        # Generate random gaze data
        gaze = {
            "gaze_x": float(np.random.uniform(-1.0, 1.0)),
            "gaze_y": float(np.random.uniform(-1.0, 1.0)),
            "gaze_confidence": float(np.random.uniform(0.7, 1.0)),
            "blink_detected": bool(np.random.choice([True, False], p=[0.1, 0.9]))
        }
        
        # Generate random head pose tilt angles
        head = {
            "pitch": float(np.random.uniform(-15.0, 15.0)),
            "yaw": float(np.random.uniform(-15.0, 15.0)),
            "roll": float(np.random.uniform(-5.0, 5.0))
        }
        
        engine.add_record(
            timestamp=t_start + i,
            emotions_distribution=emotions_map,
            gaze_data=gaze,
            head_pose=head
        )
        
    # Compile and print report
    report_data = engine.end_session()
    print("\nSimulated Session Summary statistics:")
    print(f"Duration: {report_data['session_duration_sec']:.2f}s")
    print(f"Dominant Emotion: {report_data['dominant_emotion']}")
    print(f"Avg Engagement Score: {report_data['average_engagement']:.2f}")
    print(f"Avg Valence Index: {report_data['average_valence_index']:.2f}")
    print(f"Distraction Rate: {report_data['distraction_rate'] * 100.0:.1f}%")
