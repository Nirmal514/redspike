import os
import sys
import asyncio
import json
import time
import yaml
from typing import Dict, Any, List
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from data_simulator.audio_synth import AudioSynthesizer
from data_simulator.vibration_synth import VibrationSynthesizer
from data_simulator.fault_injector import FaultInjector, FaultType
from dsp.feature_extractor import DSPFeatureExtractor
from ml_engine.inference import EdgeInferenceEngine
from edge_controller.rule_engine import EdgeSafetyRuleEngine
from edge_controller.autopilot_integration import autopilot_manager, autopilot_router

# Load Configuration
config_path = os.path.join(BASE_DIR, "config", "default_config.yaml")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}

app = FastAPI(title="Appliance Health Monitor", version="1.0.0")

# Mount Static & Templates
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Include Autopilot API Router
app.include_router(autopilot_router)

# Global Processing Singletons
current_appliance_type = "water_pump"
current_appliance_id = "appliance_waterpump_01"

fault_injector = FaultInjector()
audio_synth = AudioSynthesizer(sample_rate=16000, appliance_type=current_appliance_type)
vib_synth = VibrationSynthesizer(sample_rate=500, appliance_type=current_appliance_type)
dsp_extractor = DSPFeatureExtractor()
ml_engine = EdgeInferenceEngine()
rule_engine = EdgeSafetyRuleEngine()

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        dead_sockets = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead_sockets.append(ws)
        for dead in dead_sockets:
            self.disconnect(dead)

manager = ConnectionManager()

# Background Processing & Streaming Task
async def simulation_loop():
    global audio_synth, vib_synth, current_appliance_type, current_appliance_id
    
    # 256ms audio frame: 4096 samples at 16kHz
    # 256ms vibration frame: 128 samples at 500Hz
    AUDIO_SAMPLES_PER_FRAME = 4096
    VIB_SAMPLES_PER_FRAME = 128

    while True:
        try:
            # Check if relay is tripped (Power cut off)
            relay_state = rule_engine._relay_state.get(current_appliance_id, "CLOSED")
            is_powered_on = (relay_state == "CLOSED")

            if is_powered_on:
                # 1. Synthesize Audio & Vibration
                audio_chunk = audio_synth.generate_chunk(AUDIO_SAMPLES_PER_FRAME, fault_injector=fault_injector)
                vib_chunk = vib_synth.generate_chunk(VIB_SAMPLES_PER_FRAME, fault_injector=fault_injector)
            else:
                # Machine turned off by emergency trip -> low ambient background noise
                audio_chunk = np.random.normal(0, 0.005, AUDIO_SAMPLES_PER_FRAME).astype(np.float32)
                vib_chunk = np.random.normal(0, 0.002, (3, VIB_SAMPLES_PER_FRAME)).astype(np.float32)

            # 2. Extract DSP Features
            mel_spec, vib_vector, dsp_telemetry = dsp_extractor.process_frame(audio_chunk, vib_chunk)

            # 3. Edge ML Inference
            if is_powered_on:
                prediction = ml_engine.predict(mel_spec, vib_vector)
            else:
                prediction = {
                    "anomaly_score": 0.0,
                    "is_anomaly": False,
                    "health_score": 0.0,
                    "fault_class": "normal (POWER_OFF)",
                    "fault_confidence": 1.0,
                    "severity": "low",
                    "severity_confidence": 1.0,
                    "fault_probabilities": {"normal": 1.0}
                }

            # 4. Edge Safety State Machine Evaluation
            safety_eval = rule_engine.evaluate(current_appliance_id, prediction)

            # 5. Update Autopilot Manager
            autopilot_manager.update_status(current_appliance_id, prediction, safety_eval)

            # Downsample waveforms for lightweight UI streaming
            audio_downsampled = audio_chunk[::16].tolist()  # 256 points for oscilloscope
            vib_downsampled_x = vib_chunk[0][::2].tolist()   # 64 points
            vib_downsampled_y = vib_chunk[1][::2].tolist()
            vib_downsampled_z = vib_chunk[2][::2].tolist()

            # Spectrogram frame: 128 mel bins by 16 time slices
            # Quantize mel_spec to float with 2 decimal places for lightweight JSON payload
            mel_slice = np.clip((mel_spec + 1.0) * 50.0, 0, 100).astype(int).tolist()

            # Broadcast package
            telemetry_packet = {
                "timestamp": time.time(),
                "appliance_id": current_appliance_id,
                "appliance_type": current_appliance_type,
                "prediction": prediction,
                "safety": safety_eval,
                "fault_injector": fault_injector.get_state(),
                "autopilot": autopilot_manager.get_health(current_appliance_id),
                "dsp": dsp_telemetry,
                "waveform": {
                    "audio": audio_downsampled,
                    "vib_x": vib_downsampled_x,
                    "vib_y": vib_downsampled_y,
                    "vib_z": vib_downsampled_z
                },
                "spectrogram": mel_slice
            }

            await manager.broadcast(telemetry_packet)
        except Exception as e:
            print(f"[SimLoop Error] {e}")

        # Target ~ 4-5 updates per second (250ms interval)
        await asyncio.sleep(0.25)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulation_loop())

# Request Models
class FaultInjectRequest(BaseModel):
    fault_type: str
    intensity: float = 0.8
    duration_sec: float = 0.0  # 0 = indefinite

class ApplianceSelectRequest(BaseModel):
    appliance_type: str  # "refrigerator", "washing_machine", "water_pump"

# Web Routes
@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "appliance_type": current_appliance_type,
        "appliance_id": current_appliance_id
    })

@app.post("/api/fault/inject")
async def inject_fault(req: FaultInjectRequest):
    dur = req.duration_sec if req.duration_sec > 0 else None
    fault_injector.inject_fault(
        fault=req.fault_type,
        intensity=req.intensity,
        duration_sec=dur,
        appliance_id=current_appliance_id
    )
    return {"status": "success", "state": fault_injector.get_state()}

@app.post("/api/fault/clear")
async def clear_fault():
    fault_injector.clear_fault()
    return {"status": "success", "state": fault_injector.get_state()}

@app.post("/api/relay/reset")
async def reset_safety_relay():
    rule_engine.reset_relay(current_appliance_id)
    fault_injector.clear_fault()
    return {"status": "success", "relay_state": "CLOSED"}

@app.post("/api/appliance/select")
async def select_appliance(req: ApplianceSelectRequest):
    global current_appliance_type, current_appliance_id, audio_synth, vib_synth
    valid_types = {
        "refrigerator": "appliance_refrigerator_01",
        "washing_machine": "appliance_washer_01",
        "water_pump": "appliance_waterpump_01"
    }
    if req.appliance_type not in valid_types:
        return JSONResponse(status_code=400, content={"error": "Invalid appliance type"})

    current_appliance_type = req.appliance_type
    current_appliance_id = valid_types[req.appliance_type]
    audio_synth = AudioSynthesizer(sample_rate=16000, appliance_type=current_appliance_type)
    vib_synth = VibrationSynthesizer(sample_rate=500, appliance_type=current_appliance_type)
    fault_injector.clear_fault()
    rule_engine.reset_relay(current_appliance_id)

    return {
        "status": "success",
        "appliance_type": current_appliance_type,
        "appliance_id": current_appliance_id
    }

@app.get("/api/events")
async def get_events(limit: int = 20):
    return {"events": rule_engine.get_recent_events(limit=limit)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep-alive listen
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
