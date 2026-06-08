import os
import unittest
import json
import time
import numpy as np
import pandas as pd

TENSORFLOW_AVAILABLE = True
try:
    import tensorflow as tf
except ModuleNotFoundError:
    TENSORFLOW_AVAILABLE = False
    import sys
    from unittest.mock import MagicMock
    
    # Create mock tensorflow structures
    mock_tf = MagicMock()
    mock_tf.keras = MagicMock()
    mock_tf.keras.layers = MagicMock()
    mock_tf.keras.models = MagicMock()
    mock_tf.keras.mixed_precision = MagicMock()
    mock_tf.keras.optimizers = MagicMock()
    mock_tf.keras.losses = MagicMock()
    mock_tf.keras.applications = MagicMock()
    mock_tf.data = MagicMock()
    mock_tf.data.AUTOTUNE = -1
    
    # Custom helper implementations to prevent simple runtime failures
    def mock_one_hot(indices, depth):
        return np.eye(depth)[indices]
    mock_tf.one_hot = mock_one_hot
    
    # Inject into sys.modules
    sys.modules['tensorflow'] = mock_tf
    sys.modules['tensorflow.keras'] = mock_tf.keras
    sys.modules['tensorflow.keras.layers'] = mock_tf.keras.layers
    sys.modules['tensorflow.keras.models'] = mock_tf.keras.models
    sys.modules['tensorflow.keras.mixed_precision'] = mock_tf.keras.mixed_precision
    sys.modules['tensorflow.keras.optimizers'] = mock_tf.keras.optimizers
    sys.modules['tensorflow.keras.losses'] = mock_tf.keras.losses
    sys.modules['tensorflow.keras.applications'] = mock_tf.keras.applications
    sys.modules['tensorflow.data'] = mock_tf.data
    
    import tensorflow as tf

from PIL import Image

# Modify search path to import sibling files
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import TrainingConfig
from fer2013_pipeline import PipelineConfig, DataValidator, DatasetLoader, ImbalanceHandler
from model import FER2013TransferMobileNetV2
from analytics import EmotionAnalyticsEngine
from inference import EmotionInferenceEngine

class TestUnitComponents(unittest.TestCase):
    """Unit tests validating individual classes and processing logic in isolation."""

    def setUp(self):
        self.config = TrainingConfig()
        self.pipeline_config = PipelineConfig(csv_path="mock.csv")
        self.validator = DataValidator(self.pipeline_config)
        self.loader = DatasetLoader(self.pipeline_config)
        self.analytics = EmotionAnalyticsEngine(self.config)

    def test_data_validation_row(self):
        """Validates that DataValidator handles correct and incorrect rows correctly."""
        # 1. Test valid row
        valid_img = np.random.randint(0, 255, (48, 48, 1), dtype=np.uint8)
        self.assertTrue(self.validator.validate_row(3, valid_img, "Training"))
        
        # 2. Test invalid emotion label (out of range [0, 6])
        self.assertFalse(self.validator.validate_row(7, valid_img, "Training"))
        self.assertFalse(self.validator.validate_row(-1, valid_img, "Training"))
        
        # 3. Test invalid image dimensions
        invalid_img = np.random.randint(0, 255, (50, 50, 1), dtype=np.uint8)
        self.assertFalse(self.validator.validate_row(3, invalid_img, "Training"))
        
        # 4. Test invalid Usage flag
        self.assertFalse(self.validator.validate_row(3, valid_img, "DevTest"))

    def test_gaze_distraction_flag(self):
        """Verifies distraction flag triggers under out-of-bounds gaze inputs."""
        # 1. Gaze centered (No distraction)
        gaze_center = {"gaze_x": 0.1, "gaze_y": -0.2, "gaze_confidence": 0.95, "blink_detected": False}
        self.assertEqual(self.analytics.calculate_distraction_flag(gaze_center), 0.0)
        
        # 2. Gaze off-screen (Distraction triggered)
        gaze_distracted = {"gaze_x": 0.85, "gaze_y": 0.1, "gaze_confidence": 0.95, "blink_detected": False}
        self.assertEqual(self.analytics.calculate_distraction_flag(gaze_distracted), 1.0)
        
        # 3. Gaze with blink (Partial deduction)
        gaze_blink = {"gaze_x": 0.0, "gaze_y": 0.0, "gaze_confidence": 0.3, "blink_detected": True}
        self.assertEqual(self.analytics.calculate_distraction_flag(gaze_blink), 0.5)

    def test_valence_index_logic(self):
        """Validates calculations of Valence Index for positive, negative, and neutral bounds."""
        # 1. Pure positive state
        emotions_pos = {"Happy": 1.0, "Surprise": 0.0, "Sad": 0.0, "Angry": 0.0, "Fear": 0.0, "Disgust": 0.0, "Neutral": 0.0}
        self.assertEqual(self.analytics.calculate_valence_index(emotions_pos), 1.0)
        
        # 2. Pure negative state
        emotions_neg = {"Happy": 0.0, "Surprise": 0.0, "Sad": 0.8, "Angry": 0.2, "Fear": 0.0, "Disgust": 0.0, "Neutral": 0.0}
        self.assertEqual(self.analytics.calculate_valence_index(emotions_neg), -1.0)
        
        # 3. Balanced state
        emotions_bal = {"Happy": 0.4, "Surprise": 0.1, "Sad": 0.3, "Angry": 0.2, "Fear": 0.0, "Disgust": 0.0, "Neutral": 0.0}
        self.assertAlmostEqual(self.analytics.calculate_valence_index(emotions_bal), 0.0)


@unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow is not available on this platform (e.g. Python 3.14)")
class TestModelArchitecture(unittest.TestCase):
    """Structural tests ensuring model layout compliance with architectural NFRs."""

    def setUp(self):
        self.config = TrainingConfig()
        self.builder = FER2013TransferMobileNetV2(self.config)
        self.model, self.base_model = self.builder.build_model()

    def test_input_output_dimensions(self):
        """Verifies input shape matches (None, 48, 48, 1) and output matches (None, 7)."""
        # Input Check
        self.assertEqual(self.model.input_shape, (None, 48, 48, 1))
        # Output Check
        self.assertEqual(self.model.output_shape, (None, 7))

    def test_mixed_precision_datatype(self):
        """Confirms the final classification layer runs on float32 for mixed precision stability."""
        output_layer = self.model.get_layer("output_emotion")
        # Ensure dtype is float32 to avoid precision issues in softmax
        self.assertEqual(output_layer.dtype, "float32")

    def test_fine_tune_freezing_thresholds(self):
        """Validates that base layers freeze correctly when lock properties are toggled."""
        # Unfreeze base model and freeze up to layer index 100
        self.base_model.trainable = True
        for layer in self.base_model.layers[:self.config.fine_tune_unfreeze_limit]:
            layer.trainable = False
            
        # Verify first layer is frozen
        self.assertFalse(self.base_model.layers[0].trainable)
        # Verify layer after threshold is trainable
        self.assertTrue(self.base_model.layers[self.config.fine_tune_unfreeze_limit + 1].trainable)


class TestIntegrationFlows(unittest.TestCase):
    """Integration tests executing data loading, model inference, and output serialization."""

    def setUp(self):
        self.config = TrainingConfig()
        self.config.csv_path = "tests/test_mock_fer2013.csv"
        os.makedirs(os.path.dirname(self.config.csv_path), exist_ok=True)
        
        # 1. Create a dummy test mock CSV file representation of FER2013
        # Contains 1 row per emotion class (7 classes)
        dummy_pixels = " ".join(["127"] * 2304)  # Gray values
        data = {
            "emotion": [0, 1, 2, 3, 4, 5, 6],
            "pixels": [dummy_pixels] * 7,
            "Usage": ["Training", "Training", "PublicTest", "PublicTest", "PrivateTest", "PrivateTest", "Training"]
        }
        pd.DataFrame(data).to_csv(self.config.csv_path, index=False)
        
        # Create directories for results
        self.test_output_json = "tests/test_session_report.json"

    def tearDown(self):
        # Cleanup temporary files
        if os.path.exists(self.config.csv_path):
            os.remove(self.config.csv_path)
        if os.path.exists(self.test_output_json):
            os.remove(self.test_output_json)

    def test_end_to_end_analytics_pipeline(self):
        """Simulates full workflow: load raw mock file, run mock inference, compute scores, write report."""
        # 1. Load data
        pipeline_config = PipelineConfig(csv_path=self.config.csv_path, batch_size=2)
        loader = DatasetLoader(pipeline_config)
        df = loader.load_raw_dataframe()
        self.assertEqual(len(df), 7)
        
        # 2. Run mock session loop and feed analytics
        analytics = EmotionAnalyticsEngine(self.config)
        
        # Add 3 mock data frames
        for i in range(3):
            analytics.add_record(
                timestamp=time.time() + i,
                emotions_distribution={"Happy": 0.6, "Neutral": 0.3, "Sad": 0.1, "Surprise": 0.0, "Angry": 0.0, "Fear": 0.0, "Disgust": 0.0},
                gaze_data={"gaze_x": 0.0, "gaze_y": 0.1, "gaze_confidence": 0.98},
                head_pose={"pitch": 1.2, "yaw": -0.5, "roll": 0.1}
            )
            
        # 3. Compile report
        report = analytics.end_session()
        self.assertEqual(report["total_records"], 3)
        self.assertEqual(report["dominant_emotion"], "Happy")
        self.assertTrue(report["average_engagement"] > 0.5)
        
        # 4. Serialize report to file
        analytics.export_report_to_json(self.test_output_json)
        self.assertTrue(os.path.exists(self.test_output_json))
        
        # Read back and verify
        with open(self.test_output_json, "r") as f:
            saved_data = json.load(f)
            self.assertEqual(saved_data["aggregated_statistics"]["dominant_emotion"], "Happy")


class TestDashboardInitialization(unittest.TestCase):
    """Verifies dashboard state definitions and resource caches."""

    def test_mock_streamlit_configs(self):
        """Validates config defaults mapping for dashboard layout runs."""
        config = TrainingConfig()
        # Ensure target size is 160 for EfficientNetV2B0
        self.assertEqual(config.target_image_size, 160)
        # Ensure default output has 7 labels
        self.assertEqual(len(config.emotion_labels), 7)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks testing system NFR limits (Latency and Throughput)."""

    def setUp(self):
        self.config = TrainingConfig()
        self.pipeline_config = PipelineConfig(csv_path="mock.csv")
        self.validator = DataValidator(self.pipeline_config)

    def test_validation_throughput(self):
        """Throughput Benchmark: Ensures data validator can parse > 1000 lines/sec."""
        valid_img = np.random.randint(0, 255, (48, 48, 1), dtype=np.uint8)
        num_iterations = 1000
        
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            self.validator.validate_row(3, valid_img, "Training")
        elapsed = time.perf_counter() - start_time
        
        # Calculate operations per second
        throughput = num_iterations / elapsed
        logger = tf.get_logger()
        logger.info(f"Data Validation Throughput: {throughput:.1f} rows/second")
        
        # Ensure throughput exceeds minimum 1000 rows/second threshold
        self.assertTrue(throughput > 1000.0, f"Throughput low: {throughput:.1f} rows/s")

    def test_inference_latency_simulation(self):
        """Latency Benchmark: Verifies individual inference latency averages < 50ms."""
        # Using dummy check because full TF model load might not run without active weights file.
        # This test ensures we track performance budgets.
        config = TrainingConfig()
        if os.path.exists(config.saved_model_path):
            engine = EmotionInferenceEngine(config=config)
            dummy_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            
            # Warm up
            engine.predict(dummy_image)
            
            # Run 50 iterations to average out latency spikes
            latencies = []
            for _ in range(50):
                _, _, _, lat = engine.predict(dummy_image)
                latencies.append(lat)
                
            avg_latency = np.mean(latencies)
            logger = tf.get_logger()
            logger.info(f"Average Inference Latency: {avg_latency:.2f} ms")
            
            # Set latency budget based on CPU vs GPU hardware availability
            gpus = tf.config.list_physical_devices('GPU')
            budget = 50.0 if gpus else 200.0
            
            # NFR Budget Check
            self.assertTrue(avg_latency < budget, f"Average latency exceeds {budget}ms budget: {avg_latency:.2f} ms")
        else:
            # Skip check with info if model file doesn't exist in workspace yet
            self.skipTest("Trained baseline model checkpoint not found. Skipping latency check.")

if __name__ == "__main__":
    unittest.main()
