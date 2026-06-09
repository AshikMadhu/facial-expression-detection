from pathlib import Path
import logging
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers, losses, mixed_precision

from config import TrainingConfig
from model import FER2013TransferEfficientNetV2
from fer2013_pipeline import PipelineConfig, FER2013PipelineManager

# Configure logger
logger = logging.getLogger("FER2013Trainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class CategoricalFocalLoss(tf.keras.losses.Loss):
    """
    Categorical Focal Loss for multi-class classification.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.0, name="categorical_focal_loss", **kwargs):
        super(CategoricalFocalLoss, self).__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        
        # Apply label smoothing if set
        if self.label_smoothing > 0.0:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)
            
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = self.alpha * tf.math.pow(1.0 - y_pred, self.gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super(CategoricalFocalLoss, self).get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing
        })
        return config

def enable_mixed_precision(config: TrainingConfig):
    """Enables float16 mixed precision execution for accelerated training."""
    if config.use_mixed_precision:
        try:
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision globally enabled with 'mixed_float16' policy.")
        except Exception as e:
            logger.warning(f"Could not enable mixed precision: {str(e)}. Defaulting to float32.")
    else:
        logger.info("Mixed precision disabled. Running on standard float32.")

def run_training_pipeline(config: TrainingConfig):
    """Orchestrates two-stage transfer learning and fine-tuning with EfficientNetV2B0."""
    
    # 1. Enable Mixed Precision Policy
    enable_mixed_precision(config)
    
    # 2. Initialize and load datasets
    pipeline_config = PipelineConfig(
        csv_path=config.csv_path,
        batch_size=config.batch_size,
        image_size=config.raw_image_size,
        num_channels=config.num_channels,
        num_classes=config.num_classes,
        random_seed=config.shuffle_buffer_size,
        balance_classes=config.balance_classes
    )
    
    pipeline_manager = FER2013PipelineManager(pipeline_config)
    try:
        train_ds, val_ds, test_ds, class_weights = pipeline_manager.build_pipelines()
    except Exception as e:
        logger.error(f"Failed to build training pipelines: {str(e)}")
        logger.warning("Aborting run. Verify dataset path and structure.")
        return

    # 3. Build Model Components
    model_builder = FER2013TransferEfficientNetV2(config)
    model, base_model = model_builder.build_model()
    
    # --- PHASE 1: FEATURE EXTRACTION (FREEZE BASE MODEL) ---
    logger.info("--- PHASE 1: STARTING FEATURE EXTRACTION ---")
    base_model.trainable = False  # Freeze all pre-trained convolutional blocks
    
    # Verify parameter statuses
    logger.info(f"Model Total parameters: {model.count_params()}")
    
    # Compile model for extraction phase
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=config.extraction_learning_rate, weight_decay=config.weight_decay),
        loss=CategoricalFocalLoss(gamma=config.focal_gamma, label_smoothing=config.label_smoothing),
        metrics=["accuracy"]
    )
    
    # Phase 1 Callbacks
    early_stopping_p1 = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    
    tensorboard_p1 = callbacks.TensorBoard(
        log_dir=str(config.log_dir / "phase1_extraction"),
        update_freq="epoch"
    )
    
    # Calculate steps per epoch based on training dataset size (28,709 samples in FER2013)
    steps_per_epoch = 28709 // config.batch_size

    # Fit classifier layers
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.extraction_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[early_stopping_p1, tensorboard_p1],
        class_weight=class_weights if not config.balance_classes else None,
        verbose=1
    )
    logger.info("--- PHASE 1: FEATURE EXTRACTION COMPLETE ---")

    # --- PHASE 2: FINE-TUNING (UNFREEZE TOP LAYERS) ---
    logger.info("--- PHASE 2: STARTING FINE-TUNING PROCESS ---")
    base_model.trainable = True  # Make base model parameters trainable
    
    # Freeze bottom convolutional blocks, unfreeze from unfreeze_limit onwards
    for layer in base_model.layers[:config.fine_tune_unfreeze_limit]:
        layer.trainable = False
        
    logger.info(f"Unfrozen Base Model from layer index {config.fine_tune_unfreeze_limit}.")
    
    # Re-compile model with low learning rate and Cosine Decay schedule (One-Cycle warmup/decay)
    warmup_steps = 3 * steps_per_epoch
    decay_steps = config.fine_tune_epochs * steps_per_epoch
    
    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-6,
        decay_steps=decay_steps,
        alpha=0.01,
        warmup_target=config.fine_tune_learning_rate,
        warmup_steps=warmup_steps
    )
    
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=lr_schedule, weight_decay=config.weight_decay),
        loss=CategoricalFocalLoss(gamma=config.focal_gamma, label_smoothing=config.label_smoothing),
        metrics=["accuracy"]
    )
    
    # Phase 2 Callbacks
    early_stopping_p2 = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Checkpoint both weights and full model to saved_model_path when val_loss improves
    checkpoint_p2 = callbacks.ModelCheckpoint(
        filepath=str(config.saved_model_path),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode="min"
    )
    
    tensorboard_p2 = callbacks.TensorBoard(
        log_dir=str(config.log_dir / "phase2_finetuning"),
        update_freq="epoch",
        histogram_freq=1
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.fine_tune_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[early_stopping_p2, checkpoint_p2, tensorboard_p2],
        class_weight=class_weights if not config.balance_classes else None,
        verbose=1
    )
    logger.info("--- PHASE 2: FINE-TUNING COMPLETE ---")

    # 4. Final Evaluation on isolated test set
    logger.info("Executing final evaluation on test partition...")
    # Load the best checkpoint before evaluating on test set, to guarantee we evaluate the best epoch
    if config.saved_model_path.exists():
        logger.info("Loading best model from checkpoint for evaluation and final save...")
        # Since we load to evaluate, we can load with CategoricalFocalLoss custom object
        model = tf.keras.models.load_model(
            str(config.saved_model_path),
            custom_objects={"CategoricalFocalLoss": CategoricalFocalLoss}
        )
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    logger.info(f"Final Test Evaluation -> Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

    # 5. Save final fine-tuned model (this is redundant since ModelCheckpoint saves it, but ensures we write the restored best weights)
    logger.info(f"Saving final fine-tuned EfficientNetV2 model to: {config.saved_model_path}")
    model.save(str(config.saved_model_path))
    logger.info("Training pipeline complete.")

def main():
    # Load default training configurations and trigger pipeline execution
    config = TrainingConfig()
    run_training_pipeline(config)

if __name__ == "__main__":
    main()
