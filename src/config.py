from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TrainingConfig:
    """Configuration class for the MobileNetV2 Transfer Learning pipeline."""
    # File Paths using pathlib.Path
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    csv_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "fer2013.csv")
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "checkpoints")
    log_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "logs")
    saved_model_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models" / "best_model.h5")
    
    # Image Parameters
    raw_image_size: int = 48
    target_image_size: int = 160  # Resize to 160x160 for optimal feature resolution
    num_channels: int = 1
    num_classes: int = 7
    
    # Mixed Precision Execution
    use_mixed_precision: bool = False
    
    # Phase 1: Feature Extraction Hyperparameters
    batch_size: int = 128  # Larger batch size enabled by mixed precision
    extraction_epochs: int = 10
    extraction_learning_rate: float = 1e-3
    
    # Phase 2: Fine-Tuning Hyperparameters
    fine_tune_epochs: int = 25
    fine_tune_learning_rate: float = 1e-4
    fine_tune_unfreeze_limit: int = 135  # Unfreeze from layer 135 (top blocks of EfficientNetV2B0)
    
    # Decoupled weight decay for AdamW
    weight_decay: float = 1e-3
    
    # Loss Parameters
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    
    # Regularization
    dropout_rate: float = 0.5
    l2_regularization: float = 1e-4
    shuffle_buffer_size: int = 10000
    balance_classes: bool = True  # If false, computes dynamic class weights
    
    # Callback Configurations
    early_stopping_patience: int = 8
    
    # Labels
    emotion_labels: Dict[int, str] = field(default_factory=lambda: {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral"
    })

    def __post_init__(self):
        """Creates necessary directories if they do not exist and validates configuration."""
        # Validate critical parameters
        if self.raw_image_size <= 0 or self.target_image_size <= 0:
            raise ValueError("Image sizes must be positive integers.")
        if self.num_channels <= 0:
            raise ValueError("Number of channels must be a positive integer.")
        if self.num_classes <= 0:
            raise ValueError("Number of classes must be a positive integer.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if self.extraction_epochs < 0 or self.fine_tune_epochs < 0:
            raise ValueError("Epoch counts cannot be negative.")
            
        # Create directories using pathlib
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saved_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
