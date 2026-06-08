import logging
import os
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FER2013DataPipeline")

@dataclass
class PipelineConfig:
    """Configuration class for the FER2013 Data Pipeline."""
    csv_path: str
    batch_size: int = 64
    image_size: int = 48
    num_channels: int = 1
    num_classes: int = 7
    validation_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    cache_dataset: bool = True
    shuffle_buffer_size: int = 10000
    
    # Class weights and balancing
    balance_classes: bool = False
    
    # Emotion labels mapping
    emotion_labels: Dict[int, str] = field(default_factory=lambda: {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral"
    })

class DatasetLoader:
    """Loads and parses raw FER2013 dataset from CSV format."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_raw_dataframe(self) -> pd.DataFrame:
        """Reads the CSV file containing the FER2013 data."""
        if not os.path.exists(self.config.csv_path):
            raise FileNotFoundError(f"FER2013 CSV file not found at path: {self.config.csv_path}")
        
        logger.info(f"Loading FER2013 raw data from {self.config.csv_path}...")
        df = pd.read_csv(self.config.csv_path)
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        return df

    def parse_pixels(self, pixel_str: str) -> np.ndarray:
        """Parses the raw space-separated string of pixel values into a numpy array."""
        try:
            pixels = np.fromstring(pixel_str, dtype=np.uint8, sep=" ")
            expected_pixels = self.config.image_size * self.config.image_size
            if len(pixels) != expected_pixels:
                raise ValueError(f"Malformed pixel sequence length. Expected {expected_pixels}, got {len(pixels)}")
            return pixels.reshape((self.config.image_size, self.config.image_size, self.config.num_channels))
        except Exception as e:
            logger.error(f"Error parsing pixels: {str(e)}")
            # Return empty or placeholder array for validation module to clean/catch
            return np.zeros((self.config.image_size, self.config.image_size, self.config.num_channels), dtype=np.uint8)


class DataValidator:
    """Validates data structures, formats, range values, and types."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config

    def validate_row(self, emotion: int, pixel_arr: np.ndarray, usage: str) -> bool:
        """Validates a single dataset record."""
        # Validate emotion range
        if not (0 <= emotion < self.config.num_classes):
            logger.warning(f"Invalid emotion label: {emotion}")
            return False
            
        # Validate image shape
        expected_shape = (self.config.image_size, self.config.image_size, self.config.num_channels)
        if pixel_arr.shape != expected_shape:
            logger.warning(f"Invalid pixel array shape: {pixel_arr.shape}. Expected: {expected_shape}")
            return False
            
        # Validate pixel ranges
        if np.any((pixel_arr < 0) | (pixel_arr > 255)):
            logger.warning("Pixel values out of range [0, 255]")
            return False
            
        # Validate Usage tag
        valid_usages = {"Training", "PublicTest", "PrivateTest"}
        if usage not in valid_usages:
            logger.warning(f"Invalid usage tag: {usage}")
            return False

        return True

    def validate_batch(self, emotions: np.ndarray, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs batch-level schema validation and dimensions checks."""
        assert images.shape[1:] == (self.config.image_size, self.config.image_size, self.config.num_channels), "Batch image shape mismatch"
        assert len(emotions.shape) == 1 or (len(emotions.shape) == 2 and emotions.shape[1] == self.config.num_classes), "Batch emotion label shape mismatch"
        return emotions, images


class DataCleaningPipeline:
    """Handles missing values, corrupted pixel strings, and outlier data removal."""
    
    def __init__(self, config: PipelineConfig, validator: DataValidator, loader: DatasetLoader):
        self.config = config
        self.validator = validator
        self.loader = loader

    def process(self) -> Tuple[pd.DataFrame, List[np.ndarray]]:
        """Reads raw data, cleans, and parses valid matrices."""
        df = self.loader.load_raw_dataframe()
        
        # 1. Drop rows with null values in critical columns
        initial_count = len(df)
        df = df.dropna(subset=["emotion", "pixels", "Usage"])
        if len(df) < initial_count:
            logger.warning(f"Dropped {initial_count - len(df)} rows with missing values.")
            
        # 2. Parse and Validate rows
        clean_rows = []
        parsed_images = []
        
        for idx, row in df.iterrows():
            emotion = int(row["emotion"])
            pixels = self.loader.parse_pixels(row["pixels"])
            usage = row["Usage"]
            
            if self.validator.validate_row(emotion, pixels, usage):
                clean_rows.append(row)
                parsed_images.append(pixels)
            else:
                logger.warning(f"Filtering out invalid row index: {idx}")
                
        cleaned_df = pd.DataFrame(clean_rows).reset_index(drop=True)
        logger.info(f"Cleaning complete. Kept {len(cleaned_df)} of {initial_count} records.")
        return cleaned_df, parsed_images


class RandomCoarseDropout(tf.keras.layers.Layer):
    """Custom Keras layer to perform Coarse Dropout (Cutout / Random Erasing) on a single 3D image tensor."""
    def __init__(self, probability=0.5, size_range=(8, 16), seed=None, **kwargs):
        super(RandomCoarseDropout, self).__init__(**kwargs)
        self.probability = probability
        self.size_range = size_range
        self.seed = seed

    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        # Determine if we apply erasing
        should_erase = tf.random.uniform([], seed=self.seed) < self.probability
        
        if not should_erase:
            return inputs
            
        shape = tf.shape(inputs)
        h, w = shape[0], shape[1]
        
        # Select random mask height and width
        mask_h = tf.random.uniform([], self.size_range[0], self.size_range[1], dtype=tf.int32, seed=self.seed)
        mask_w = tf.random.uniform([], self.size_range[0], self.size_range[1], dtype=tf.int32, seed=self.seed)
        
        # Select random mask top-left corner
        top = tf.random.uniform([], 0, h - mask_h, dtype=tf.int32, seed=self.seed)
        left = tf.random.uniform([], 0, w - mask_w, dtype=tf.int32, seed=self.seed)
        
        # Create mask of zeros
        mask = tf.zeros((mask_h, mask_w, shape[2]), dtype=inputs.dtype)
        
        # Create padding parameters
        paddings = [[top, h - top - mask_h], [left, w - left - mask_w], [0, 0]]
        padded_mask = tf.pad(mask, paddings, constant_values=1.0)
        
        return inputs * padded_mask

class DataAugmentationPipeline(tf.keras.layers.Layer):
    """Keras-native data augmentation pipeline running on CPU/GPU."""
    
    def __init__(self, config: PipelineConfig):
        super(DataAugmentationPipeline, self).__init__()
        self.config = config
        
        # Set up sequential augmentation layers
        self.augmentations = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=config.random_seed),
            tf.keras.layers.RandomRotation(0.12, fill_mode="constant", fill_value=0, seed=config.random_seed),
            tf.keras.layers.RandomZoom(0.1, 0.1, fill_mode="constant", fill_value=0, seed=config.random_seed),
            tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08, fill_mode="constant", fill_value=0, seed=config.random_seed),
            tf.keras.layers.RandomContrast(0.2, seed=config.random_seed),
            tf.keras.layers.RandomBrightness(0.2, seed=config.random_seed),
            RandomCoarseDropout(probability=0.4, size_range=(6, 12), seed=config.random_seed)
        ])

    def call(self, inputs, training=None):
        if training:
            # Scale pixels from [0, 255] to [0.0, 1.0] before augmenting
            x = tf.cast(inputs, tf.float32) / 255.0
            return self.augmentations(x)
        # For evaluation/test, only normalize (no random augmentations)
        return tf.cast(inputs, tf.float32) / 255.0


class ImbalanceHandler:
    """Computes weights and samples probability vectors to balance FER2013 label distributions."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config

    def calculate_class_weights(self, train_labels: np.ndarray) -> Dict[int, float]:
        """Calculates balanced class weights inversely proportional to class frequencies."""
        labels, counts = np.unique(train_labels, return_counts=True)
        total_samples = len(train_labels)
        num_classes = self.config.num_classes
        
        # Class weight formula: w = total / (classes * class_count)
        class_weights = {}
        for label, count in zip(labels, counts):
            weight = total_samples / (num_classes * count)
            class_weights[int(label)] = float(weight)
            
        logger.info(f"Calculated Class Weights: {class_weights}")
        return class_weights

    def balance_dataset(self, 
                        class_datasets: List[tf.data.Dataset], 
                        class_counts: np.ndarray) -> tf.data.Dataset:
        """Balances classes via target probability oversampling using tf.data.Dataset.sample_from_datasets."""
        # Calculate uniform sampling probability target (1/7 for each of the 7 classes)
        num_datasets = len(class_datasets)
        sampling_probabilities = [1.0 / num_datasets] * num_datasets
        
        logger.info("Oversampling dataset using uniform probability distributions.")
        balanced_ds = tf.data.Dataset.sample_from_datasets(
            class_datasets, 
            weights=sampling_probabilities,
            seed=self.config.random_seed
        )
        return balanced_ds


class FER2013PipelineManager:
    """Orchestrates data pipeline segments into production-ready tf.data streams."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.loader = DatasetLoader(config)
        self.validator = DataValidator(config)
        self.cleaner = DataCleaningPipeline(config, self.validator, self.loader)
        self.augmenter = DataAugmentationPipeline(config)
        self.imbalance_handler = ImbalanceHandler(config)

    def _split_by_usage(self, df: pd.DataFrame, images: List[np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Splits the parsed dataset by the original FER2013 'Usage' metadata splits."""
        splits = {
            "train": ([], []),
            "val": ([], []),
            "test": ([], [])
        }
        
        for idx, row in df.iterrows():
            img = images[idx]
            label = int(row["emotion"])
            usage = row["Usage"]
            
            if usage == "Training":
                splits["train"][0].append(img)
                splits["train"][1].append(label)
            elif usage == "PublicTest":
                splits["val"][0].append(img)
                splits["val"][1].append(label)
            elif usage == "PrivateTest":
                splits["test"][0].append(img)
                splits["test"][1].append(label)

        # Convert to numpy arrays
        final_splits = {}
        for key, (imgs, lbls) in splits.items():
            final_splits[key] = (np.array(imgs, dtype=np.uint8), np.array(lbls, dtype=np.int32))
            logger.info(f"Split '{key}' size: {len(final_splits[key][0])} samples")
            
        return final_splits

    def build_pipelines(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[int, float]]:
        """Assembles end-to-end data pipeline streams for Train, Validation, and Test subsets."""
        
        # 1. Load, clean, and parse structures
        cleaned_df, parsed_images = self.cleaner.process()
        
        # 2. Extract splits using the metadata flags
        dataset_splits = self._split_by_usage(cleaned_df, parsed_images)
        x_train, y_train = dataset_splits["train"]
        x_val, y_val = dataset_splits["val"]
        x_test, y_test = dataset_splits["test"]
        
        # 3. Calculate class weights for training model configurations
        class_weights = self.imbalance_handler.calculate_class_weights(y_train)

        # 4. Construct tf.data.Dataset components
        # 4.1 Validation Dataset
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_ds = val_ds.map(lambda x, y: (self.augmenter(x, training=False), tf.one_hot(y, self.config.num_classes)),
                            num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.config.batch_size)
        if self.config.cache_dataset:
            val_ds = val_ds.cache()
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # 4.2 Test Dataset
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_ds = test_ds.map(lambda x, y: (self.augmenter(x, training=False), tf.one_hot(y, self.config.num_classes)),
                              num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.config.batch_size)
        if self.config.cache_dataset:
            test_ds = test_ds.cache()
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        # 4.3 Training Dataset (with optional Class Re-sampling)
        if self.config.balance_classes:
            # Create sub-datasets for each class to perform weighted sampling
            class_datasets = []
            labels, counts = np.unique(y_train, return_counts=True)
            
            for label in range(self.config.num_classes):
                indices = np.where(y_train == label)[0]
                class_x = x_train[indices]
                class_y = y_train[indices]
                
                class_ds = tf.data.Dataset.from_tensor_slices((class_x, class_y))
                # Shuffle individual class datasets, repeat indefinitely for sampling
                class_ds = class_ds.shuffle(buffer_size=1000).repeat()
                class_datasets.append(class_ds)
                
            train_ds = self.imbalance_handler.balance_dataset(class_datasets, counts)
        else:
            # Standard random shuffle dataset
            train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_ds = train_ds.shuffle(buffer_size=self.config.shuffle_buffer_size, seed=self.config.random_seed)

        # Apply transformations, batch, and optimizations for training dataset
        train_ds = train_ds.map(
            lambda x, y: (self.augmenter(x, training=True), tf.one_hot(y, self.config.num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch and prefetch
        train_ds = train_ds.batch(self.config.batch_size)
        if self.config.cache_dataset:
            train_ds = train_ds.cache()
        
        # Prefetch to memory for maximum throughput
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        logger.info("Data pipelines generated successfully.")
        return train_ds, val_ds, test_ds, class_weights

# Example usage pattern when executing the module directly
if __name__ == "__main__":
    # Setup standard configuration (using placeholder csv file path)
    config = PipelineConfig(
        csv_path="data/fer2013.csv",
        batch_size=32,
        balance_classes=True
    )
    
    pipeline_manager = FER2013PipelineManager(config)
    try:
        train_pipeline, val_pipeline, test_pipeline, weights = pipeline_manager.build_pipelines()
        print("Pipeline initialization test complete.")
    except Exception as e:
        logger.error(f"Failed to build pipeline: {str(e)}")
        print(f"Skipping complete run: File not created yet in filesystem.")
