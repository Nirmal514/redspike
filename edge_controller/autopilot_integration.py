import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

class ApplianceHealthResponse(BaseModel):
    appliance_id: str
    status: str  # "HEALTHY", "DEGRADED", "CRITICAL_SHUTDOWN"
    allow_operation: bool
    health_score: float  # 0.0 to 100.0%
    degradation_factor: float  # 0.0 (pristine) to 1.0 (failed)
    active_fault: str
    severity: str
    recommended_action: str
    timestamp: float

class AutopilotManager:
    """
    Coordinates with Smart Home Energy Autopilots (e.g. dynamic grid pricing dispatchers).
    Prevents scheduled high-efficiency runs on damaged machinery to avert catastrophic failures.
    """
    def __init__(self):
        self._appliance_status: Dict[str, Dict[str, Any]] = {}

    def update_status(self, appliance_id: str, ml_pred: Dict[str, Any], rule_eval: Dict[str, Any]):
        health_score = ml_pred.get("health_score", 100.0)
        anomaly_score = ml_pred.get("anomaly_score", 0.0)
        fault_class = ml_pred.get("fault_class", "normal")
        severity = ml_pred.get("severity", "low")
        relay_state = rule_eval.get("relay_state", "CLOSED")

        if relay_state == "OPEN":
            status = "CRITICAL_SHUTDOWN"
            allow_op = False
            rec_action = "EMERGENCY: Power disconnected due to active critical fault. Inspect hardware."
        elif health_score < 60.0 or severity in ("medium", "critical"):
            status = "DEGRADED"
            allow_op = False
            rec_action = f"HOLD OPERATION: Maintenance required for {fault_class.replace('_', ' ')}. Do not schedule."
        elif health_score < 85.0:
            status = "MONITORING"
            allow_op = True
            rec_action = "Sub-optimal acoustic profile. Safe for normal low-load cycles."
        else:
            status = "HEALTHY"
            allow_op = True
            rec_action = "Optimal efficiency. Unrestricted autopilot scheduling allowed."

        self._appliance_status[appliance_id] = {
            "appliance_id": appliance_id,
            "status": status,
            "allow_operation": allow_op,
            "health_score": health_score,
            "degradation_factor": round(anomaly_score, 4),
            "active_fault": fault_class,
            "severity": severity,
            "recommended_action": rec_action,
            "timestamp": time.time()
        }

    def get_health(self, appliance_id: str) -> Optional[Dict[str, Any]]:
        return self._appliance_status.get(appliance_id)

    def get_all_appliances(self) -> Dict[str, Dict[str, Any]]:
        return self._appliance_status

# Global Autopilot Instance
autopilot_manager = AutopilotManager()
autopilot_router = APIRouter(prefix="/api/v1", tags=["Energy Autopilot Integration"])

@autopilot_router.get("/appliance-health", response_model=ApplianceHealthResponse)
def get_appliance_health(appliance_id: str = Query("appliance_refrigerator_01", description="ID of appliance")):
    health = autopilot_manager.get_health(appliance_id)
    if not health:
        # Default nominal state
        return ApplianceHealthResponse(
            appliance_id=appliance_id,
            status="HEALTHY",
            allow_operation=True,
            health_score=100.0,
            degradation_factor=0.0,
            active_fault="normal",
            severity="low",
            recommended_action="Appliance initialized in nominal mode.",
            timestamp=time.time()
        )
    return ApplianceHealthResponse(**health)

@autopilot_router.get("/appliances")
def get_all_appliances():
    return {"appliances": autopilot_manager.get_all_appliances()}
