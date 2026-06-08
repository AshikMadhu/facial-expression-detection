import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetV2B0
from config import TrainingConfig

class FER2013TransferEfficientNetV2:
    """Builds an EfficientNetV2B0-based transfer learning model for facial emotion recognition."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config

    def build_model(self) -> tuple:
        """
        Constructs the hybrid transfer learning model. Handles grayscale to RGB
        conversion, resolution resizing, base model loading, and classification head.
        """
        l2_reg = regularizers.l2(self.config.l2_regularization)
        
        # 1. Grayscale Input Layer (48x48x1)
        inputs = layers.Input(shape=(self.config.raw_image_size, self.config.raw_image_size, self.config.num_channels), name="raw_input")
        
        # 2. Convert Grayscale (1 Channel) to RGB (3 Channels) via Concatenation
        x = layers.Concatenate(axis=-1, name="grayscale_to_rgb")([inputs, inputs, inputs])
        
        # 3. Resize Image to target_image_size using bicubic interpolation to prevent aliasing
        x = layers.Resizing(self.config.target_image_size, self.config.target_image_size, interpolation="bicubic", name="resize_layer")(x)
        
        # 4. Map [0.0, 1.0] pixels back to [0.0, 255.0] as expected by EfficientNetV2B0
        x = layers.Rescaling(scale=255.0, name="efficientnetv2_preprocess")(x)
        
        # 5. Load EfficientNetV2B0 base weights trained on ImageNet
        base_model = EfficientNetV2B0(
            input_shape=(self.config.target_image_size, self.config.target_image_size, 3),
            include_top=False,
            weights="imagenet"
        )
        
        # Connect base model
        x = base_model(x, training=False)  # Ensure BatchNormalization runs in inference mode during frozen phases
        
        # 6. Classification Head
        x = layers.GlobalAveragePooling2D(name="global_pooling")(x)
        
        # Dense representation layer
        x = layers.Dense(256, kernel_regularizer=l2_reg, name="fc1")(x)
        x = layers.LayerNormalization(name="ln_fc1")(x)
        x = layers.Activation("swish", name="swish_fc1")(x)
        x = layers.Dropout(self.config.dropout_rate, name="dropout_fc1")(x)
        
        # Output layer - Force float32 activation for mixed precision stability
        outputs = layers.Dense(
            self.config.num_classes, 
            activation="softmax", 
            dtype="float32", 
            name="output_emotion"
        )(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name="FER2013_EfficientNetV2_Transfer")
        return model, base_model

# Maintain backward compatibility alias
FER2013TransferMobileNetV2 = FER2013TransferEfficientNetV2

if __name__ == "__main__":
    # Test model build execution and summary
    config = TrainingConfig()
    builder = FER2013TransferEfficientNetV2(config)
    model, base_model = builder.build_model()
    model.summary()
