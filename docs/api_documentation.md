# EmotionSense AI: API Specifications

This document defines the REST API endpoints, time-series telemetry payloads, and schema definitions for the EmotionSense AI backend ingestion and analytics service.

---

## 1. Authentication & Security
All outbound API requests from clients to the cloud gateway must include a standard JSON Web Token (JWT) in the headers to guarantee tenant isolation and authorization:

```http
Authorization: Bearer <JWT_TOKEN>
```

---

## 2. Ingestion Telemetry Stream API

Submits batch telemetry frames from the client edge application to the time-series datastore.

*   **URL**: `/api/v1/sessions/{session_id}/telemetry`
*   **Method**: `POST`
*   **Headers**:
    *   `Content-Type: application/json`
    *   `Authorization: Bearer <JWT_TOKEN>`

### 2.1 Path Parameters
*   `session_id` (string, UUIDv4): The unique identifier for the active tracking session.

### 2.2 Request Payload Schema
```json
{
  "client_timestamp": "string (ISO-8601)",
  "sequence_number": "integer",
  "metrics": [
    {
      "timestamp_offset_ms": "integer",
      "emotions": {
        "happy": "float [0.0 - 1.0]",
        "sad": "float [0.0 - 1.0]",
        "angry": "float [0.0 - 1.0]",
        "surprise": "float [0.0 - 1.0]",
        "fear": "float [0.0 - 1.0]",
        "disgust": "float [0.0 - 1.0]",
        "neutral": "float [0.0 - 1.0]"
      },
      "gaze": {
        "gaze_x": "float [-1.0 - 1.0]",
        "gaze_y": "float [-1.0 - 1.0]",
        "gaze_confidence": "float [0.0 - 1.0]",
        "blink_detected": "boolean"
      },
      "head_pose": {
        "pitch": "float (degrees)",
        "yaw": "float (degrees)",
        "roll": "float (degrees)"
      },
      "engagement_score": "float [0.0 - 1.0]"
    }
  ]
}
```

### 2.3 Response Codes
*   **`202 Accepted`**: Ingestion payload successfully validated and queued for time-series aggregation.
    ```json
    {
      "status": "success",
      "message": "Metrics batch queued.",
      "records_received": 1
    }
    ```
*   **`400 Bad Request`**: Validation failure due to incorrect types, missing keys, or timestamp values.
*   **`401 Unauthorized`**: Token validation failure or expired credentials.
*   **`429 Too Many Requests`**: Ingestion rate limit exceeded (Max: 1 request per 3 seconds per client IP).

---

## 3. Historical Summary Query API

Retrieves aggregated session analytics for the administrative dashboard.

*   **URL**: `/api/v1/analytics/sessions/{session_id}/summary`
*   **Method**: `GET`
*   **Headers**:
    *   `Authorization: Bearer <JWT_TOKEN>`

### 3.1 Path Parameters
*   `session_id` (string, UUIDv4): Target session identification.

### 3.2 Response Payload
*   **`200 OK`**:
    ```json
    {
      "session_id": "d3b07384-d113-4c9f-b7a4-8461011d8d97",
      "duration_seconds": 1240.5,
      "total_records": 12405,
      "dominant_emotion": "Neutral",
      "averages": {
        "engagement": 0.764,
        "confidence": 0.812,
        "distraction_rate": 0.12,
        "valence_index": 0.18
      },
      "peak_engagement": {
        "score": 0.942,
        "timestamp_offset_seconds": 345.0
      },
      "emotion_distribution": {
        "Happy": 0.23,
        "Neutral": 0.54,
        "Sad": 0.08,
        "Surprise": 0.05,
        "Angry": 0.06,
        "Fear": 0.03,
        "Disgust": 0.01
      }
    }
    ```
*   **`404 Not Found`**: Session ID not found.
