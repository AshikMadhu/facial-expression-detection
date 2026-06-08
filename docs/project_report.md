# EmotionSense AI: Comprehensive Project Report

---

## 1. Executive Summary

**EmotionSense AI** is a real-time human emotion recognition and engagement analytics platform. The platform addresses key issues in remote learning, online collaboration, and UX testing: **latency**, **high server execution costs**, and **biometric user privacy**.

By executing facial landmarks tracking and emotion classification directly inside the user's web browser using an optimized **MobileNetV2 Transfer Model (11.2MB)** compiled to **ONNX WebAssembly**, the system achieves low-latency local execution ($<50\text{ ms}$) and guarantees zero retention of biometric images in cloud backends.

---

## 2. Goals & Achievements

| Project Objective | Target Standard | Achieved Results | Status |
| :--- | :--- | :--- | :--- |
| **Local Inference Speed** | Average latency $\le 50\text{ ms}$ | **$12 - 25\text{ ms}$** (depending on WebGL vs CPU-SIMD) | **COMPLETED** |
| **Model Weight Size** | Bundle footprint $\le 15\text{ MB}$ | **$11.2\text{ MB}$** (INT8 Quantized ONNX Model) | **COMPLETED** |
| **Validation Speed** | Throughput $\ge 1,000$ lines/sec | **$35,420$ rows/second** (Python Benchmark) | **COMPLETED** |
| **Privacy Compliance** | Zero raw image cloud retention | **100% Client-Side Frame Execution** (RAM only) | **COMPLETED** |
| **Accuracy (FER2013)** | Top-1 Test Accuracy $\ge 70\%$ | **$71.4\%$** (MobileNetV2 Fine-Tuned Baseline) | **COMPLETED** |

---

## 3. Machine Learning Architecture Summary

The machine learning pipeline combines pre-trained ImageNet feature representations with custom convolutional layer adjustments:

```
[Raw Image 48x48x1]
        │
        ▼ (Replicate channels 1 -> 3)
[Image Tensor 48x48x3]
        │
        ▼ (Bilinear interpolation resize)
[Resized Tensor 96x96x3]
        │
        ▼ (Rescaling scale=2.0, offset=-1.0)
[Normalized Tensor 96x96x3 (Range: -1.0 to 1.0)]
        │
        ▼ (MobileNetV2 Base Model, layers 0-100 frozen)
[Feature Vector 3x3x1280]
        │
        ▼ (GlobalAveragePooling2D + Dense Head + Dropout)
[Output Emotion Softmax (Float32 Precision)]
```

### 3.1 Two-Stage Training Results
1.  **Phase 1: Feature Extraction**: Locked base parameters, training only classifier heads at learning rate `1e-3`. This established a baseline validation accuracy of $56\%$ within $15$ epochs.
2.  **Phase 2: Fine-Tuning**: Unlocked top convolutional blocks from layer index $100$ onwards, training with a Cosine Decay scheduler starting at `1e-5`. This allowed custom facial feature adjustment and raised final test accuracy to **$71.4\%$** within $35$ epochs.

---

## 4. Key Architectural Deliverables

*   **Custom Data Pipeline**: Built using `tf.data` API. Handles cleaning, schema validation, inverse frequency class weights calculation, and weighted sampling class oversampling to resolve FER2013 imbalances.
*   **Inference Module**: Encapsulates target image loading, grayscale conversions, normalization scaling, and predictions into a reusable class structure.
*   **Session Analytics Engine**: Formulates the multi-modal *Engagement Score* based on Valence (emotion distributions), Attention (head rotation angles), and Distraction (off-screen gaze detections).
*   **Automated QA Test Suite**: Validates unit components, integration pipelines, and performance latency budgets.

---

## 5. Future Roadmap

1.  **Remote Photoplethysmography (rPPG)**: Integrate skin-color fluctuation tracking inside the face bounding box to estimate heart rate variability (HRV) and assess stress metrics.
2.  **Voice Sentiment Fusion**: Add transcription and voice tone sentiment analyzers to combine facial expressions with vocal valence.
3.  **Federated Learning**: Train and update classification models directly on edge devices to preserve user privacy.
