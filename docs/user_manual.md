# EmotionSense AI: User Manual

This manual explains how to operate the EmotionSense AI real-time user interface, interpret active telemetry metrics, and download session reports.

---

## 1. Launching the Interface

To open the Streamlit user application:
1. Ensure your virtual environment is activated.
2. Run the start command from the terminal:
   ```bash
   streamlit run packages/ml-models/src/dashboard.py
   ```
3. A browser tab will automatically open pointing to: `http://localhost:8501`.

---

## 2. Navigating the UI Dashboard

The interface is divided into two primary zones: the **Sidebar Controls** and the **Active Workspace**.

```
+-------------------------------------------------------------------+
| 🧠 EmotionSense AI          EmotionSense AI Dashboard             |
|                                                                   |
| [ ▶️ Start Session ]        [ 🎥 Live Video ]   [ 📊 Metrics ]    |
|                           |   (Camera feed)   |   Dominant Em.    |
|                           |                   |   Engage. Score   |
|                           +-------------------+                   |
|                           | Probability Chart (Horizontal bar)    |
|                           +---------------------------------------+
|                           | 📈 Session Timeline (Real-time Line)  |
|                           +---------------------------------------+
|                           | 📋 Summary Statistics (Valence, etc.) |
+-------------------------------------------------------------------+
```

### 2.1 Sidebar Panel
*   **▶️ Start Recording Session**: Opens your webcam capture interface, resets any prior metrics history, and starts recording active metrics.
*   **⏹️ Stop & Compile Session**: Safely disconnects the camera stream, computes cumulative statistics, and offers a download button for the session report.
*   **Local badge**: Displays privacy assurances indicating that all frames are processed locally.

### 2.2 Live Video Canvas
*   Displays the mirrored camera feed.
*   Features a cyan box overlay surrounding the user's face, displaying the current emotion label and model classification confidence score.

### 2.3 Live Telemetry Panel
*   **Current Dominant Emotion Badge**: Displays the predicted class. The color changes dynamically based on the emotion (e.g., Green for Happy, Red for Angry).
*   **Engagement Score Metric**: Displays your computed engagement score as a percentage.
*   **Probability Chart**: Renders a horizontal bar chart showing the active confidence levels across all seven emotion classes.

---

## 3. Interpreting Telemetry Metrics

EmotionSense AI translates physical facial landmarks into four distinct metrics:

1.  **Dominant Emotion**: The emotion category containing the highest probability value. Bounded to the seven FER2013 classes.
2.  **Valence Index (Range: -1.0 to 1.0)**:
    *   *Positive values* ($>0.0$) imply positive reactions (Happy, Surprise).
    *   *Negative values* ($<0.0$) imply negative reactions (Frustration, Confusion, Sadness).
3.  **Attention Level (Range: 0.0% to 100.0%)**: Measures user focus. Focus drops if the user tilts or turns their head away from the camera, or closes their eyes for extended periods.
4.  **Engagement Score (Range: 0.0% to 100.0%)**: The aggregate index. Computed using:
    $$\text{Engagement} = 30\% \cdot \text{Valence} + 50\% \cdot \text{Attention} - 20\% \cdot \text{Distraction}$$

---

## 4. Exporting Session Reports

When you click **⏹️ Stop & Compile Session**:
1. The video stream halts.
2. The UI renders the final session metrics and average emotion distributions.
3. Click the **📥 Download Session JSON Report** button to save the session data to your local machine.

The exported JSON file contains:
*   `session_metadata`: Timestamp records tracking start and stop execution times.
*   `aggregated_statistics`: Session averages for engagement, distraction rate, valence index, and dominant emotion.
*   `telemetry_history`: Chronological logs of every frame processed during the session (10 entries per second).
