from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class SensorReadingIn(BaseModel):
    """One raw sensor reading. A prediction request sends a short window of these."""
    timestamp: datetime
    cycle: int
    vibration_mm_s: float
    temperature_c: float
    pressure_bar: float
    rpm: float
    current_amp: float


class PredictionRequest(BaseModel):
    machine_id: str
    machine_type: str
    readings: List[SensorReadingIn] = Field(
        ..., min_length=1,
        description="Trailing window of readings for this machine, oldest first. "
                    "At least 60 readings recommended so rolling/FFT features are fully populated."
    )


class FeatureDriver(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    machine_id: str
    timestamp: datetime
    predicted_rul: float
    failure_probability: float
    anomaly_score: float
    is_anomaly: bool
    risk_state: str
    risk_reason: str
    top_drivers: List[FeatureDriver]
    model_version: str


class MachineHistoryPoint(BaseModel):
    timestamp: datetime
    predicted_rul: float
    failure_probability: float
    risk_state: str


class AlertOut(BaseModel):
    id: int
    machine_id: str
    triggered_at: datetime
    risk_state: str
    predicted_rul: Optional[float]
    failure_probability: Optional[float]
    message: Optional[str]
    acknowledged: bool