import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.alerts.thresholds import assess_risk
from src.api.schemas import (
    PredictionRequest, PredictionResponse, FeatureDriver,
    MachineHistoryPoint, AlertOut,
)
from src.config import settings, SENSOR_COLUMNS
from src.database.db import init_db, get_session
from src.database.models import Prediction, Alert, Machine
from src.explainability.explain import FailureExplainer
from src.features.engineering import _rolling_features, _fft_features

MODEL_DIR = Path(settings.model_dir)
MODEL_VERSION = "xgb-v1"

app = FastAPI(
    title="Predictive Maintenance API",
    description="Real-time RUL, failure-window, and anomaly predictions for industrial equipment.",
    version="1.0.0",
)

# Loaded once at startup, reused across requests
_state = {}


@app.on_event("startup")
def load_artifacts():
    init_db()
    try:
        _state["reg"] = joblib.load(MODEL_DIR / "rul_regressor.joblib")
        _state["clf"] = joblib.load(MODEL_DIR / "failure_classifier.joblib")
        _state["iso"] = joblib.load(MODEL_DIR / "isolation_forest.joblib")
        _state["feature_cols"] = joblib.load(MODEL_DIR / "feature_columns.joblib")
        _state["explainer"] = FailureExplainer()
        _state["ready"] = True
    except FileNotFoundError:
        # API can still start (e.g. for /health checks in CI) before training has run
        _state["ready"] = False


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _state.get("ready", False)}


def _build_feature_row(request: PredictionRequest) -> pd.DataFrame:
    """Runs the same rolling/FFT feature logic used in training on a request window."""
    readings = sorted(request.readings, key=lambda r: r.cycle)
    df = pd.DataFrame([r.model_dump() for r in readings])
    df["machine_id"] = request.machine_id

    rolling = _rolling_features(df)
    fft = _fft_features(df)
    combined = pd.concat([rolling, fft], axis=1)
    return combined.iloc[[-1]]  # most recent reading's features


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Models not loaded — run training first.")

    feature_cols = _state["feature_cols"]
    feature_row_df = _build_feature_row(request)

    missing = [c for c in feature_cols if c not in feature_row_df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Could not compute {len(missing)} required feature(s): {missing[:5]} ... "
                   f"— check that the readings window is long enough and sensors are complete.",
        )
    X = feature_row_df[feature_cols].to_numpy()

    predicted_rul = float(_state["reg"].predict(X)[0])
    failure_probability = float(_state["clf"].predict_proba(X)[0, 1])

    iso_pred = _state["iso"].predict(X)[0]           # -1 = anomaly, 1 = normal
    anomaly_score = float(-_state["iso"].decision_function(X)[0])  # higher = more anomalous
    is_anomaly = bool(iso_pred == -1)

    risk = assess_risk(predicted_rul, failure_probability, is_anomaly)

    drivers = _state["explainer"].top_drivers(X[0], top_n=5)
    top_drivers = [FeatureDriver(feature=f, shap_value=float(v)) for f, v in drivers]

    # Stamp the prediction with the latest sensor reading's time, not wall-clock
    # time, so history charts follow the machine's real timeline
    now = max(r.timestamp for r in request.readings)

    with get_session() as session:
        if not session.get(Machine, request.machine_id):
            session.add(Machine(machine_id=request.machine_id, machine_type=request.machine_type))

        session.add(Prediction(
            machine_id=request.machine_id,
            timestamp=now,
            predicted_rul=predicted_rul,
            failure_probability=failure_probability,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            risk_state=risk.risk_state,
            top_drivers=json.dumps([d.model_dump() for d in top_drivers]),
            model_version=MODEL_VERSION,
        ))

        if risk.risk_state in ("WARNING", "CRITICAL"):
            session.add(Alert(
                machine_id=request.machine_id,
                triggered_at=now,
                risk_state=risk.risk_state,
                predicted_rul=predicted_rul,
                failure_probability=failure_probability,
                message=risk.reason,
            ))

    return PredictionResponse(
        machine_id=request.machine_id,
        timestamp=now,
        predicted_rul=predicted_rul,
        failure_probability=failure_probability,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        risk_state=risk.risk_state,
        risk_reason=risk.reason,
        top_drivers=top_drivers,
        model_version=MODEL_VERSION,
    )


@app.get("/machines/{machine_id}/history", response_model=List[MachineHistoryPoint])
def machine_history(machine_id: str, limit: int = Query(100, le=1000)):
    with get_session() as session:
        rows = (
            session.query(Prediction)
            .filter(Prediction.machine_id == machine_id)
            .order_by(Prediction.timestamp.desc())
            .limit(limit)
            .all()
        )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No prediction history for machine '{machine_id}'")

    return [
        MachineHistoryPoint(
            timestamp=r.timestamp,
            predicted_rul=r.predicted_rul,
            failure_probability=r.failure_probability,
            risk_state=r.risk_state,
        )
        for r in reversed(rows)
    ]


@app.get("/alerts", response_model=List[AlertOut])
def list_alerts(
    machine_id: Optional[str] = None,
    risk_state: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    limit: int = Query(200, le=1000),
):
    with get_session() as session:
        q = session.query(Alert)
        if machine_id:
            q = q.filter(Alert.machine_id == machine_id)
        if risk_state:
            q = q.filter(Alert.risk_state == risk_state)
        if acknowledged is not None:
            q = q.filter(Alert.acknowledged == acknowledged)
        rows = q.order_by(Alert.triggered_at.desc()).limit(limit).all()

    return [
        AlertOut(
            id=r.id, machine_id=r.machine_id, triggered_at=r.triggered_at,
            risk_state=r.risk_state, predicted_rul=r.predicted_rul,
            failure_probability=r.failure_probability, message=r.message,
            acknowledged=r.acknowledged,
        )
        for r in rows
    ]
