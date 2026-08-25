import time
import json
import os
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Safe SQLite import
SQLITE_AVAILABLE = False
try:
    import sqlite3
    SQLITE_AVAILABLE = True
except (ImportError, Exception):
    SQLITE_AVAILABLE = False

class SystemAction(str, Enum):
    NO_ACTION = "NO_ACTION"
    WARNING_ALERT = "WARNING_ALERT"
    SCHEDULE_MAINTENANCE = "SCHEDULE_MAINTENANCE"
    EMERGENCY_POWER_CUTOFF = "EMERGENCY_POWER_CUTOFF"

@dataclass
class SystemEvent:
    timestamp: float
    appliance_id: str
    action: SystemAction
    fault_type: str
    severity: str
    anomaly_score: float
    description: str
    relay_state: str  # "CLOSED" (Power ON) or "OPEN" (Power OFF)

class EdgeSafetyRuleEngine:
    """
    Temporal State Machine Rule Engine:
    - Analyzes incoming sliding-window predictions.
    - Demands temporal persistence before actuating physical shutoffs to prevent false positives.
    - Logs incidents to persistent storage (SQLite or JSONL fallback).
    - Simulates MQTT and GPIO relay actuation.
    """
    def __init__(
        self,
        db_path: str = "appliance_health.db",
        trip_persistence_seconds: float = 2.0,
        warning_persistence_seconds: float = 1.0
    ):
        self.trip_persistence_seconds = trip_persistence_seconds
        self.warning_persistence_seconds = warning_persistence_seconds
        self.db_path = db_path
        self.jsonl_path = "appliance_health_events.jsonl"

        # State tracking per appliance
        self._fault_start_times: Dict[str, float] = {}
        self._current_fault: Dict[str, str] = {}
        self._relay_state: Dict[str, str] = {}  # "CLOSED" (ON) or "OPEN" (OFF/TRIPPED)
        self._active_events: List[SystemEvent] = []

        self._init_db()

    def _init_db(self):
        if SQLITE_AVAILABLE:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS health_events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp REAL,
                            appliance_id TEXT,
                            action TEXT,
                            fault_type TEXT,
                            severity TEXT,
                            anomaly_score REAL,
                            description TEXT,
                            relay_state TEXT
                        )
                    """)
                    conn.commit()
            except Exception as e:
                print(f"[RuleEngine] SQLite note: {e}")

    def log_event(self, event: SystemEvent):
        self._active_events.append(event)
        if len(self._active_events) > 100:
            self._active_events.pop(0)

        # Write to SQLite if available
        if SQLITE_AVAILABLE:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO health_events
                        (timestamp, appliance_id, action, fault_type, severity, anomaly_score, description, relay_state)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.timestamp,
                        event.appliance_id,
                        event.action.value,
                        event.fault_type,
                        event.severity,
                        event.anomaly_score,
                        event.description,
                        event.relay_state
                    ))
                    conn.commit()
                return
            except Exception:
                pass

        # Fallback JSONL audit logging
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                event_dict = asdict(event)
                event_dict["action"] = event.action.value
                f.write(json.dumps(event_dict) + "\n")
        except Exception as e:
            print(f"[RuleEngine] Error writing event log: {e}")

    def evaluate(self, appliance_id: str, ml_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates real-time frame inference result against safety policies.
        """
        now = time.time()
        fault = ml_prediction.get("fault_class", "normal")
        severity = ml_prediction.get("severity", "low")
        anomaly_score = ml_prediction.get("anomaly_score", 0.0)
        is_anomaly = ml_prediction.get("is_anomaly", False)

        if appliance_id not in self._relay_state:
            self._relay_state[appliance_id] = "CLOSED"

        if self._relay_state[appliance_id] == "OPEN":
            return {
                "action": SystemAction.NO_ACTION.value,
                "relay_state": "OPEN",
                "status": "POWER_DISCONNECTED",
                "persistence_sec": 0.0,
                "message": "Power cut off. Manual reset required."
            }

        if is_anomaly and fault != "normal":
            if self._current_fault.get(appliance_id) == fault:
                start_time = self._fault_start_times.get(appliance_id, now)
            else:
                self._current_fault[appliance_id] = fault
                self._fault_start_times[appliance_id] = now
                start_time = now
            duration = now - start_time
        else:
            self._current_fault[appliance_id] = "normal"
            self._fault_start_times[appliance_id] = now
            duration = 0.0

        action = SystemAction.NO_ACTION
        description = "Nominal operating conditions."

        if severity == "critical" and duration >= self.trip_persistence_seconds:
            action = SystemAction.EMERGENCY_POWER_CUTOFF
            self._relay_state[appliance_id] = "OPEN"
            description = (
                f"CRITICAL SAFETY TRIP: Persistent {fault.upper()} detected for {duration:.1f}s. "
                f"Emergency power cutoff initiated to protect motor/impeller."
            )
            self._emit_mqtt_trip(appliance_id, "OFF")
            self._simulate_gpio_trip(appliance_id, 18, "OPEN")

        elif severity in ("medium", "critical") and duration >= self.warning_persistence_seconds:
            action = SystemAction.WARNING_ALERT
            description = (
                f"MAINTENANCE WARNING: {fault.replace('_', ' ').title()} active for {duration:.1f}s. "
                f"Scheduled service dispatch recommended."
            )
        elif is_anomaly:
            description = f"Transient anomaly detected ({fault}). Monitoring persistence ({duration:.1f}s / {self.trip_persistence_seconds:.1f}s)..."

        if action != SystemAction.NO_ACTION:
            event = SystemEvent(
                timestamp=now,
                appliance_id=appliance_id,
                action=action,
                fault_type=fault,
                severity=severity,
                anomaly_score=anomaly_score,
                description=description,
                relay_state=self._relay_state[appliance_id]
            )
            self.log_event(event)

        return {
            "action": action.value,
            "relay_state": self._relay_state[appliance_id],
            "persistence_sec": round(duration, 2),
            "description": description,
            "status": "TRIPPED" if self._relay_state[appliance_id] == "OPEN" else ("WARNING" if action == SystemAction.WARNING_ALERT else "NORMAL")
        }

    def reset_relay(self, appliance_id: str):
        self._relay_state[appliance_id] = "CLOSED"
        self._fault_start_times[appliance_id] = time.time()
        self._current_fault[appliance_id] = "normal"
        self._emit_mqtt_trip(appliance_id, "ON")

    def _emit_mqtt_trip(self, appliance_id: str, command: str):
        topic = f"smartplug/{appliance_id}/command"
        print(f"[MQTT Simulated] Published topic='{topic}' payload='{command}'")

    def _simulate_gpio_trip(self, appliance_id: str, pin: int, state: str):
        print(f"[GPIO Simulated] Relay PIN {pin} set to {state} for {appliance_id}")

    def get_recent_events(self, limit: int = 15) -> List[Dict[str, Any]]:
        return [asdict(e) for e in reversed(self._active_events[-limit:])]
