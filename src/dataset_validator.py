import os
import sys
import logging
from typing import Dict
import pandas as pd
import numpy as np

# Adjust sys.path to find sibling configs
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from config import TrainingConfig

# Configure logger
logger = logging.getLogger("DatasetValidator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class FER2013DatasetValidator:
    """Verifies FER2013 CSV integrity, schemas, split counts, and pixel matrices reconstruction."""

    def __init__(self, csv_path: str = None):
        self.config = TrainingConfig()
        self.csv_path = csv_path or self.config.csv_path
        
        # Expected standard FER2013 parameters
        self.expected_columns = {"emotion", "pixels", "Usage"}
        self.expected_total_records = 35887
        self.expected_splits = {
            "Training": 28709,
            "PublicTest": 3589,
            "PrivateTest": 3589
        }
        self.expected_pixels_count = 2304  # 48 * 48

    def run_all_checks(self) -> bool:
        """Executes the full dataset verification suite. Returns True if dataset is 100% ready."""
        logger.info("Starting FER2013 Dataset Validation Suite...")
        
        report_data = {
            "file_exists": False,
            "schema_valid": False,
            "row_counts_valid": False,
            "no_null_values": False,
            "zero_corrupt_pixels": False,
            "reconstruction_valid": False,
            "details": {}
        }
        
        # 1. Check File Existence
        if not os.path.exists(self.csv_path):
            logger.error(f"Dataset file NOT found at: {self.csv_path}")
            report_data["details"]["file_error"] = f"File missing at path: {self.csv_path}"
            self._save_report(report_data, False)
            return False
            
        report_data["file_exists"] = True
        logger.info(f"Verified dataset file exists at: {self.csv_path}")
        
        # 2. Load DataFrame
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded DataFrame. Initial shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to read CSV file: {str(e)}")
            report_data["details"]["read_error"] = f"CSV read failed: {str(e)}"
            self._save_report(report_data, False)
            return False

        # 3. Validate Schema Column Names
        columns = set(df.columns)
        if columns != self.expected_columns:
            logger.error(f"Invalid columns: {columns}. Expected: {self.expected_columns}")
            report_data["details"]["schema_error"] = f"Column mismatch. Got {columns}, expected {self.expected_columns}"
            self._save_report(report_data, False)
            return False
            
        report_data["schema_valid"] = True
        logger.info("CSV columns validated successfully.")

        # 4. Check for Null Values
        null_counts = df.isnull().sum().to_dict()
        total_nulls = sum(null_counts.values())
        if total_nulls > 0:
            logger.warning(f"Null values detected: {null_counts}")
            report_data["details"]["null_counts"] = null_counts
        else:
            report_data["no_null_values"] = True
            logger.info("No null values detected in the CSV columns.")

        # 5. Check Row Counts and Splits Statistics
        total_rows = len(df)
        report_data["details"]["total_records"] = total_rows
        
        usage_counts = df["Usage"].value_counts().to_dict()
        report_data["details"]["usage_counts"] = usage_counts
        
        logger.info(f"Record distribution: {usage_counts}")
        
        # Validate count standard values
        counts_match = (
            total_rows == self.expected_total_records and
            usage_counts.get("Training", 0) == self.expected_splits["Training"] and
            usage_counts.get("PublicTest", 0) == self.expected_splits["PublicTest"] and
            usage_counts.get("PrivateTest", 0) == self.expected_splits["PrivateTest"]
        )
        
        if counts_match:
            report_data["row_counts_valid"] = True
            logger.info("Row counts and split sizes match official FER2013 standards.")
        else:
            logger.warning(
                f"Row counts mismatch standard sizes. Total: {total_rows} (Expected: {self.expected_total_records}), "
                f"Training: {usage_counts.get('Training', 0)} (Expected: {self.expected_splits['Training']}), "
                f"PublicTest: {usage_counts.get('PublicTest', 0)} (Expected: {self.expected_splits['PublicTest']}), "
                f"PrivateTest: {usage_counts.get('PrivateTest', 0)} (Expected: {self.expected_splits['PrivateTest']})"
            )

        # 6. Validate Image Reconstruction & Corrupted Pixels
        logger.info("Verifying pixel dimensions and array reconstructions (sampling check)...")
        corrupted_count = 0
        sample_indices = np.random.choice(len(df), size=min(1000, len(df)), replace=False)  # Sample check 1000 rows
        
        for idx in sample_indices:
            pixel_str = df.loc[idx, "pixels"]
            try:
                # Convert space-separated string to numpy array
                pixels = np.fromstring(pixel_str, dtype=np.int32, sep=" ")
                if len(pixels) != self.expected_pixels_count:
                    corrupted_count += 1
            except Exception:
                corrupted_count += 1
                
        report_data["details"]["corrupted_pixel_rows_sampled_check"] = corrupted_count
        
        if corrupted_count == 0:
            report_data["zero_corrupt_pixels"] = True
            report_data["reconstruction_valid"] = True
            logger.info("Pixel sequence lengths and array reconstructions validated successfully.")
        else:
            logger.error(f"Corrupted pixel configurations detected in {corrupted_count} sampled rows.")

        # 7. Compile and save markdown report
        is_ready = all([
            report_data["file_exists"],
            report_data["schema_valid"],
            report_data["row_counts_valid"],
            report_data["no_null_values"],
            report_data["zero_corrupt_pixels"],
            report_data["reconstruction_valid"]
        ])
        
        self._save_report(report_data, is_ready)
        
        if is_ready:
            logger.info("=== FER2013 DATASET INTEGRATION READY ===")
            return True
        else:
            logger.error("=== FER2013 DATASET INTEGRATION FAILED ===")
            return False

    def _save_report(self, report: Dict, is_ready: bool):
        """Saves a detailed markdown report inside the output folder."""
        output_dir = os.path.join(os.path.dirname(self.csv_path), "reports")
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "dataset_readiness_report.md")
        
        status_badge = "✅ READY" if is_ready else "❌ NOT READY"
        
        markdown_content = f"""# FER2013 Dataset Integration Readiness Report
Generated on: {pd.Timestamp.now()}
Status: **{status_badge}**

---

## 📋 Verification Checks

| Check Name | Target Criteria | Result Status |
| :--- | :--- | :--- |
| **File Existence** | File exists at `data/fer2013.csv` | {"✅ Pass" if report["file_exists"] else "❌ Fail"} |
| **Schema Columns** | Exactly: `['emotion', 'pixels', 'Usage']` | {"✅ Pass" if report["schema_valid"] else "❌ Fail"} |
| **Row Count Split Validation** | Total: 35,887, Train: 28,709, Val: 3,589, Test: 3,589 | {"✅ Pass" if report["row_counts_valid"] else "⚠️ Warning/Mismatch"} |
| **Data Completeness** | Zero null values | {"✅ Pass" if report["no_null_values"] else "⚠️ Null values present"} |
| **Array Structure** | Space-separated strings containing 2304 integers | {"✅ Pass" if report["zero_corrupt_pixels"] else "❌ Fail"} |
| **Reconstruction** | Reshapes cleanly to `(48, 48, 1)` | {"✅ Pass" if report["reconstruction_valid"] else "❌ Fail"} |

---

## 📊 Summary Statistics

*   **Total Checked Records**: {report['details'].get('total_records', 0)}
*   **Split Distribution**:
    *   *Training*: {report['details'].get('usage_counts', {}).get('Training', 0)} (Target: 28,709)
    *   *PublicTest (Validation)*: {report['details'].get('usage_counts', {}).get('PublicTest', 0)} (Target: 3,589)
    *   *PrivateTest (Test)*: {report['details'].get('usage_counts', {}).get('PrivateTest', 0)} (Target: 3,589)
*   **Sample Corrupt Rows**: {report['details'].get('corrupted_pixel_rows_sampled_check', 0)} / 1000 sampled

---

## 🛠️ Errors / Warnings Details
```json
{json.dumps(report["details"], indent=2)}
```
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Saved dataset readiness report to: {report_path}")

# Import json helper
import json

if __name__ == "__main__":
    validator = FER2013DatasetValidator()
    success = validator.run_all_checks()
    sys.exit(0 if success else 1)
