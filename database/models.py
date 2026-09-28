"""SQLAlchemy models for the predictive maintenance schema."""
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Machine(Base):
    __tablename__ = "machines"

    machine_id = Column(String, primary_key=True)
    machine_type = Column(String, nullable=False)
    installed_at = Column(DateTime, nullable=True)


class SensorReading(Base):
    """Raw ingested time series — one row per machine per timestamp."""
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String, ForeignKey("machines.machine_id"), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    cycle = Column(Integer, nullable=False)
    vibration_mm_s = Column(Float)
    temperature_c = Column(Float)
    pressure_bar = Column(Float)
    rpm = Column(Float)
    current_amp = Column(Float)


class Prediction(Base):
    """One prediction per (machine, timestamp) — RUL, failure risk, anomaly score."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String, ForeignKey("machines.machine_id"), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    predicted_rul = Column(Float, nullable=False)
    failure_probability = Column(Float, nullable=False)   # P(fail within FAILURE_WINDOW_CYCLES)
    anomaly_score = Column(Float, nullable=False)          # higher = more anomalous
    is_anomaly = Column(Boolean, default=False)
    risk_state = Column(String, nullable=False)             # OK / WATCH / WARNING / CRITICAL
    top_drivers = Column(Text)                              # JSON-encoded SHAP top features
    model_version = Column(String, nullable=True)


class Alert(Base):
    """Alert log — created whenever a prediction crosses a configured threshold."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String, ForeignKey("machines.machine_id"), index=True, nullable=False)
    triggered_at = Column(DateTime, nullable=False)
    risk_state = Column(String, nullable=False)
    predicted_rul = Column(Float)
    failure_probability = Column(Float)
    message = Column(Text)
    acknowledged = Column(Boolean, default=False)