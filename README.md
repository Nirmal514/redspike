# Appliance Health Monitor Using Sound & Vibration

A production-grade, real-time edge diagnostic system that monitors domestic and industrial appliances (**Refrigerator**, **Washing Machine**, **Water Pump**) by processing continuous acoustic and tri-axial mechanical vibration telemetry.

---

## 🌟 Key Capabilities & System Architecture

```
[Phone Mic / Acoustic Sensor (16kHz)] ──> [STFT / Mel-Spectrogram (128x16)] ──┐
                                                                              ├──> [Dual-Stream Fusion CNN] ──> [Anomaly Score, Fault Type, Severity]
[3-Axis Accelerometer (X,Y,Z 500Hz)]  ──> [Time/Freq Statistical Vector (36d)] ┘                                            │
                                                                                                                           ▼
                                                                                                              [Temporal Safety State Machine]
                                                                                                                           │
                                                                                 ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
                                                                                 ▼                                         ▼                                         ▼
                                                                      [MQTT / Smart Plug Cutoff]                [Local SQLite Audit Log]                [Energy Autopilot REST API]
```

1. **Synthetic Sensor Data Generator & Dynamic Streamer (`data_simulator/`)**:
   - Continuous audio synthesis (16 kHz) simulating motor harmonics (50Hz/60Hz), pink noise floor, and rotational dynamics.
   - 3-axis accelerometer synthesis (500 Hz) simulating rotational dynamics and unbalanced orbits.
   - Dynamic real-time fault injector supporting:
     - **Refrigerator:** Compressor Bearing Wear, Fan Blade Friction, Relay Chatter.
     - **Washing Machine:** Out-of-Balance Tub, Belt Slip, Drain Pump Cavitation.
     - **Water Pump:** Impeller Cavitation, Dry Running, Motor Bearing Failure, Pipe Water Hammer.

2. **Digital Signal Processing (DSP) Pipeline (`dsp/`)**:
   - 128-band Mel-Spectrogram extraction (`(128, 16)` tensors for 256ms sliding windows).
   - Time-domain vibration statistics: RMS, Crest Factor, Kurtosis, Skewness, Peak-to-Peak, Variance.
   - Frequency-domain vibration statistics: Dominant frequency bins, sub-synchronous energy, rotational energy, high-frequency defect band energy.
   - Butterworth filtering & envelope demodulation.

3. **Dual-Stream Edge Neural Network (`ml_engine/`)**:
   - **Stream 1 (Acoustic):** 2D-CNN with depthwise separable conv blocks processing 128-band Mel-Spectrogram frames.
   - **Stream 2 (Vibration):** 1D-Dense statistical feature extraction network processing 36-dimensional vectors.
   - **Multi-task Heads:**
     - `Anomaly Score` (0.0 to 1.0)
     - `Fault Classification` (7 classes: Normal, Bearing Wear, Cavitation, Unbalance, Friction, Dry Running, Chatter/Hammer)
     - `Severity Score` (Low, Medium, Critical)
   - PyTorch and ONNX Runtime support.

4. **Edge Safety Controller & Rule Engine (`edge_controller/`)**:
   - Temporal state machine with persistence thresholds (e.g. 2.0s continuous critical fault required before emergency trip to prevent false positives).
   - Actuation simulated via MQTT (`smartplug/{appliance_id}/command -> OFF`) and simulated GPIO relay.
   - SQLite event logging for historical audit trails.
   - Energy Autopilot Gateway (`/api/v1/appliance-health`) providing health factors to pause scheduled high-energy cycles on degraded hardware.

5. **Real-Time Monitoring Dashboard (`dashboard/`)**:
   - Glassmorphic dark-mode web interface.
   - 60 FPS Canvas Oscilloscopes (Acoustic Waveform + 3-Axis Vibration).
   - Real-time rolling 128-band Mel-Spectrogram heatmap.
   - Dynamic health gauge, active fault cards, interactive synthetic fault injection panel, and live event audit stream.

---

## 🚀 Quickstart Guide

### Option 1: Docker (Recommended)
```bash
docker-compose up --build
```
Open your browser at `http://localhost:8000`.

---

### Option 2: Local Python Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train & export the edge ML model:**
   ```bash
   python ml_engine/train.py
   ```

3. **Launch the dashboard server:**
   ```bash
   python dashboard/app.py
   ```
   Or using uvicorn:
   ```bash
   uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
   ```

---

## 📡 API Reference

### 1. Energy Autopilot Health Endpoint
- **URL:** `GET /api/v1/appliance-health?appliance_id=appliance_waterpump_01`
- **Response:**
  ```json
  {
    "appliance_id": "appliance_waterpump_01",
    "status": "HEALTHY",
    "allow_operation": true,
    "health_score": 98.5,
    "degradation_factor": 0.015,
    "active_fault": "normal",
    "severity": "low",
    "recommended_action": "Optimal efficiency. Unrestricted autopilot scheduling allowed.",
    "timestamp": 1724601234.56
  }
  ```

### 2. Inject Synthetic Fault
- **URL:** `POST /api/fault/inject`
- **Payload:**
  ```json
  {
    "fault_type": "impeller_cavitation",
    "intensity": 0.85,
    "duration_sec": 5.0
  }
  ```

### 3. Reset Emergency Relay
- **URL:** `POST /api/relay/reset`
- **Response:** `{"status": "success", "relay_state": "CLOSED"}`

### 4. Real-Time Telemetry WebSocket
- **URL:** `ws://localhost:8000/ws`
- **Streams:** Real-time audio waveform, 3-axis vibration samples, 128-mel spectrogram frames, ML predictions, safety relay status, and autopilot flags.
