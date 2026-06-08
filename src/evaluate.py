import os
import logging
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from config import TrainingConfig
from fer2013_pipeline import PipelineConfig, FER2013PipelineManager

# Configure logger
logger = logging.getLogger("FER2013Evaluator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class FER2013Evaluator:
    """Evaluates trained Keras models and generates reports, dashboards, and error analysis logs."""

    def __init__(self, config: TrainingConfig, output_dir: str = None):
        self.config = config
        self.output_dir = output_dir or os.path.join(config.project_root, "evaluation_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load label mapping from config
        self.labels = [config.emotion_labels[i] for i in sorted(config.emotion_labels.keys())]

    def load_model(self) -> tf.keras.Model:
        """Loads the saved Keras model."""
        if not os.path.exists(self.config.saved_model_path):
            raise FileNotFoundError(f"Trained model not found at path: {self.config.saved_model_path}")
        logger.info(f"Loading trained model from {self.config.saved_model_path}...")
        return tf.keras.models.load_model(self.config.saved_model_path, compile=False)

    def extract_ground_truth_and_predictions(self, model: tf.keras.Model, test_ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Runs batch predictions to extract inputs, true labels, predicted classes, and probabilities."""
        logger.info("Running inference on test dataset...")
        
        all_images = []
        all_true_labels = []
        all_pred_probs = []
        
        # Iterate over batches to collect data (handles custom prefetching/batches)
        for images, labels in test_ds:
            probs = model.predict(images, verbose=0)
            
            all_images.append(images.numpy())
            # Convert one-hot back to integer labels
            all_true_labels.append(np.argmax(labels.numpy(), axis=-1))
            all_pred_probs.append(probs)
            
        # Concatenate batches
        images_arr = np.concatenate(all_images, axis=0)
        true_labels = np.concatenate(all_true_labels, axis=0)
        pred_probs = np.concatenate(all_pred_probs, axis=0)
        pred_labels = np.argmax(pred_probs, axis=-1)
        
        return images_arr, true_labels, pred_labels, pred_probs

    def compute_metrics(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict:
        """Calculates exact Confusion Matrix, Precision, Recall, F1, and Classification Report."""
        logger.info("Computing metrics...")
        
        # Classification report (dict structure)
        clf_dict = classification_report(
            true_labels, 
            pred_labels, 
            target_names=self.labels, 
            output_dict=True,
            zero_division=0
        )
        
        # Standard classification text report
        clf_text = classification_report(
            true_labels, 
            pred_labels, 
            target_names=self.labels, 
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Per class stats
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, 
            pred_labels, 
            labels=list(range(self.config.num_classes)),
            zero_division=0
        )
        
        # Save text report to file
        report_path = os.path.join(self.output_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write("=== EMOTIONSENSE AI: CLASSIFICATION REPORT ===\n\n")
            f.write(clf_text)
        logger.info(f"Saved text report to: {report_path}")
        
        return {
            "classification_report_dict": clf_dict,
            "confusion_matrix": cm,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support
        }

    def run_error_analysis(self, 
                           images: np.ndarray, 
                           true_labels: np.ndarray, 
                           pred_labels: np.ndarray, 
                           pred_probs: np.ndarray) -> List[Dict]:
        """
        Analyzes prediction errors. Identifies top misclassifications and locates high-confidence
        errors where the model made an incorrect prediction with high confidence.
        """
        logger.info("Running error analysis...")
        errors = []
        
        for idx in range(len(true_labels)):
            true_l = true_labels[idx]
            pred_l = pred_labels[idx]
            
            if true_l != pred_l:
                pred_confidence = float(pred_probs[idx][pred_l])
                true_confidence = float(pred_probs[idx][true_l])
                
                errors.append({
                    "index": idx,
                    "true_label": self.config.emotion_labels[true_l],
                    "predicted_label": self.config.emotion_labels[pred_l],
                    "pred_confidence": pred_confidence,
                    "true_confidence": true_confidence,
                    "image": images[idx].tolist()  # Save pixel values as list for JSON serialization
                })
                
        # Sort errors by confidence (highest confidence errors first)
        errors.sort(key=lambda x: x["pred_confidence"], reverse=True)
        
        # Log summary stats
        total_errors = len(errors)
        logger.info(f"Total misclassifications analyzed: {total_errors} (out of {len(true_labels)} samples)")
        
        # Save a summary log to JSON (exclude image list to keep size small)
        json_log = []
        for err in errors[:100]:  # Top 100 severe errors
            log_item = err.copy()
            log_item.pop("image")  # Remove raw pixels
            json_log.append(log_item)
            
        log_path = os.path.join(self.output_dir, "error_analysis_log.json")
        with open(log_path, "w") as f:
            json.dump(json_log, f, indent=4)
        logger.info(f"Saved top 100 error logs to: {log_path}")
        
        return errors

    def generate_visualization_dashboard(self, 
                                         metrics: Dict, 
                                         images: np.ndarray, 
                                         errors: List[Dict]):
        """Generates a high-resolution, multi-panel PDF/PNG report visualization dashboard."""
        logger.info("Generating visualization dashboard...")
        
        # Create subplots grid
        sns.set_theme(style="dark")
        fig = plt.figure(figsize=(20, 16))
        grid = GridSpec(2, 2, wspace=0.3, hspace=0.3)
        
        # --- PANEL A: CONFUSION MATRIX ---
        ax_cm = fig.add_subplot(grid[0, 0])
        cm = metrics["confusion_matrix"]
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_norm, 
            annot=cm, 
            fmt="d", 
            cmap="Purples", 
            xticklabels=self.labels, 
            yticklabels=self.labels,
            cbar=True,
            ax=ax_cm,
            annot_kws={"size": 11}
        )
        ax_cm.set_title("Normalized Confusion Matrix (Percentages & Raw Counts)", fontsize=14, fontweight="bold", pad=10)
        ax_cm.set_xlabel("Predicted Emotion Label", fontsize=12)
        ax_cm.set_ylabel("True Emotion Label", fontsize=12)
        
        # --- PANEL B: PER-CLASS PERFORMANCE BAR CHART ---
        ax_bars = fig.add_subplot(grid[0, 1])
        x = np.arange(len(self.labels))
        width = 0.25
        
        ax_bars.bar(x - width, metrics["precision"], width, label="Precision", color="#3f51b5")
        ax_bars.bar(x, metrics["recall"], width, label="Recall", color="#e91e63")
        ax_bars.bar(x + width, metrics["f1_score"], width, label="F1-Score", color="#009688")
        
        ax_bars.set_title("Per-Class Metric Profiles (Precision, Recall, F1)", fontsize=14, fontweight="bold", pad=10)
        ax_bars.set_xticks(x)
        ax_bars.set_xticklabels(self.labels, fontsize=10)
        ax_bars.set_ylim(0, 1.05)
        ax_bars.set_ylabel("Metric Value Score", fontsize=12)
        ax_bars.grid(axis='y', linestyle='--', alpha=0.7)
        ax_bars.legend(loc="upper right", frameon=True)
        
        # --- PANEL C: COMPACT CLASSIFICATION STATS TABLE ---
        ax_table = fig.add_subplot(grid[1, 0])
        ax_table.axis('off')
        
        # Extract dynamic summary parameters
        table_data = []
        for i, label in enumerate(self.labels):
            table_data.append([
                label,
                f"{metrics['precision'][i]:.3f}",
                f"{metrics['recall'][i]:.3f}",
                f"{metrics['f1_score'][i]:.3f}",
                str(metrics["support"][i])
            ])
            
        # Add summary rows
        clf_dict = metrics["classification_report_dict"]
        table_data.append(["", "", "", "", ""]) # Empty spacer
        table_data.append(["Accuracy", "", "", f"{clf_dict['accuracy']:.3f}", str(clf_dict["macro avg"]["support"])])
        table_data.append(["Macro Avg", f"{clf_dict['macro avg']['precision']:.3f}", f"{clf_dict['macro avg']['recall']:.3f}", f"{clf_dict['macro avg']['f1-score']:.3f}", str(clf_dict["macro avg"]["support"])])
        table_data.append(["Weighted Avg", f"{clf_dict['weighted avg']['precision']:.3f}", f"{clf_dict['weighted avg']['recall']:.3f}", f"{clf_dict['weighted avg']['f1-score']:.3f}", str(clf_dict["weighted avg"]["support"])])
        
        column_headers = ["Emotion Class", "Precision", "Recall", "F1-Score", "Support Count"]
        
        table = ax_table.table(
            cellText=table_data,
            colLabels=column_headers,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.8)
        # Bold headers and summary lines
        for key, cell in table.get_celld().items():
            row, col = key
            if row == 0 or row > len(self.labels):
                cell.set_text_props(weight='bold')
                
        ax_table.set_title("Detailed Performance Summary Metrics", fontsize=14, fontweight="bold", pad=10)

        # --- PANEL D: SAMPLE HIGH-CONFIDENCE ERRORS ---
        ax_errors = fig.add_subplot(grid[1, 1])
        ax_errors.axis('off')
        
        # Render a 2x3 grid of top errors
        num_err_to_show = min(6, len(errors))
        if num_err_to_show > 0:
            err_grid = GridSpecFromSubplotSpec(2, 3, subplot_spec=grid[1, 1], wspace=0.3, hspace=0.4)
            for i in range(num_err_to_show):
                err_data = errors[i]
                row = i // 3
                col = i % 3
                
                ax_img = fig.add_subplot(err_grid[row, col])
                img_data = np.array(err_data["image"], dtype=np.float32)
                
                # Check dimensions and shape for visualization
                if img_data.shape[-1] == 1:
                    img_data = np.squeeze(img_data, axis=-1)
                
                ax_img.imshow(img_data, cmap="gray")
                ax_img.axis('off')
                
                title_str = (
                    f"True: {err_data['true_label']}\n"
                    f"Pred: {err_data['predicted_label']}\n"
                    f"Conf: {err_data['pred_confidence']:.2f}"
                )
                ax_img.set_title(title_str, fontsize=10, color="darkred", fontweight="semibold")
            
            ax_errors.set_title("Most Severe Prediction Failures (High-Confidence Errors)", fontsize=14, fontweight="bold", pad=15)
        else:
            ax_errors.text(0.5, 0.5, "No classification errors detected.", fontsize=14, ha='center')

        # Save dashboard image file
        dashboard_path = os.path.join(self.output_dir, "evaluation_dashboard.png")
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved visualization dashboard to: {dashboard_path}")

    def run_eval(self):
        """Orchestrates the evaluation flow."""
        logger.info("Starting baseline evaluation framework sequence...")
        
        # 1. Setup configs
        pipeline_config = PipelineConfig(
            csv_path=self.config.csv_path,
            batch_size=self.config.batch_size,
            image_size=self.config.raw_image_size,
            num_channels=self.config.num_channels,
            num_classes=self.config.num_classes,
            random_seed=self.config.shuffle_buffer_size,
            balance_classes=False
        )
        
        # 2. Get Test Pipeline
        pipeline_manager = FER2013PipelineManager(pipeline_config)
        try:
            _, _, test_ds, _ = pipeline_manager.build_pipelines()
        except Exception as e:
            logger.error(f"Failed to load test dataset pipeline: {str(e)}")
            return
            
        # 3. Load Model
        try:
            model = self.load_model()
        except Exception as e:
            logger.error(f"Failed to load saved model: {str(e)}")
            return
            
        # 4. Extract target data and predictions
        images, true_labels, pred_labels, pred_probs = self.extract_ground_truth_and_predictions(model, test_ds)
        
        # 5. Calculate Metrics
        metrics = self.compute_metrics(true_labels, pred_labels)
        
        # 6. Run Error Analysis
        errors = self.run_error_analysis(images, true_labels, pred_labels, pred_probs)
        
        # 7. Render Visualization Dashboard
        self.generate_visualization_dashboard(metrics, images, errors)
        
        logger.info("Evaluation framework completed evaluation run successfully.")

def main():
    # Initialize defaults and execute
    config = TrainingConfig()
    evaluator = FER2013Evaluator(config)
    evaluator.run_eval()

if __name__ == "__main__":
    main()
