import time
from enum import Enum
from typing import Dict, Any, Optional

class FaultType(str, Enum):
    NORMAL = "normal"
    # Refrigerator faults
    COMPRESSOR_BEARING_WEAR = "compressor_bearing_wear"
    FAN_BLADE_FRICTION = "fan_blade_friction"
    RELAY_CHATTER = "relay_chatter"
    # Washing machine faults
    UNBALANCE_TUB = "unbalance_tub"
    BELT_SLIP = "belt_slip"
    DRAIN_CAVITATION = "drain_cavitation"
    # Water pump faults
    IMPELLER_CAVITATION = "impeller_cavitation"
    DRY_RUNNING = "dry_running"
    MOTOR_BEARING_WEAR = "motor_bearing_wear"
    PIPE_WATER_HAMMER = "pipe_water_hammer"

class FaultInjector:
    """
    Manages active faults, intensity (0.0 to 1.0), and duration for target appliances.
    Thread-safe state controller for dynamic runtime fault generation.
    """
    def __init__(self):
        self._active_fault: FaultType = FaultType.NORMAL
        self._intensity: float = 0.0  # 0.0 (None) to 1.0 (Severe)
        self._start_time: Optional[float] = None
        self._duration_sec: Optional[float] = None
        self._appliance_id: str = "appliance_refrigerator_01"

    def inject_fault(self, fault: str, intensity: float = 0.8, duration_sec: Optional[float] = None, appliance_id: Optional[str] = None):
        try:
            self._active_fault = FaultType(fault)
        except ValueError:
            self._active_fault = FaultType.NORMAL
        self._intensity = max(0.0, min(1.0, float(intensity)))
        self._start_time = time.time()
        self._duration_sec = duration_sec
        if appliance_id:
            self._appliance_id = appliance_id

    def clear_fault(self):
        self._active_fault = FaultType.NORMAL
        self._intensity = 0.0
        self._start_time = None
        self._duration_sec = None

    def get_state(self) -> Dict[str, Any]:
        # Check for auto-expiry if duration is set
        if self._active_fault != FaultType.NORMAL and self._duration_sec and self._start_time:
            if time.time() - self._start_time > self._duration_sec:
                self.clear_fault()

        return {
            "fault": self._active_fault.value,
            "intensity": self._intensity,
            "is_anomalous": self._active_fault != FaultType.NORMAL,
            "appliance_id": self._appliance_id,
            "elapsed_sec": (time.time() - self._start_time) if self._start_time else 0.0
        }

    @property
    def active_fault(self) -> FaultType:
        return self._active_fault

    @property
    def intensity(self) -> float:
        return self._intensity
