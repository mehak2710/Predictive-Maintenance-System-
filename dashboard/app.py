import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sqlalchemy import text

from src.config import settings, SENSOR_COLUMNS
from src.database.db import engine

API_BASE = f"http://{settings.api_host if settings.api_host != '0.0.0.0' else 'localhost'}:{settings.api_port}"

st.set_page_config(page_title="Predictive Maintenance — Fleet Ops", layout="wide")

RISK_COLORS = {"OK": "#2ecc71", "WATCH": "#f1c40f", "WARNING": "#e67e22", "CRITICAL": "#e74c3c"}


@st.cache_data(ttl=30)
def load_latest_predictions() -> pd.DataFrame:
    query = text("""
        SELECT DISTINCT ON (machine_id)
            machine_id, timestamp, predicted_rul, failure_probability,
            anomaly_score, is_anomaly, risk_state
        FROM predictions
        ORDER BY machine_id, timestamp DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=30)
def load_machine_history(machine_id: str, limit: int = 300) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, predicted_rul, failure_probability, risk_state
        FROM predictions
        WHERE machine_id = :mid
        ORDER BY timestamp DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"mid": machine_id, "limit": limit})
    return df.sort_values("timestamp")


@st.cache_data(ttl=30)
def load_sensor_history(machine_id: str, limit: int = 500) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, cycle, vibration_mm_s, temperature_c, pressure_bar, rpm, current_amp
        FROM sensor_readings
        WHERE machine_id = :mid
        ORDER BY cycle DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"mid": machine_id, "limit": limit})
    return df.sort_values("cycle")


@st.cache_data(ttl=30)
def load_alerts(machine_id: str = None) -> pd.DataFrame:
    if machine_id:
        query = text("""
            SELECT id, machine_id, triggered_at, risk_state, predicted_rul,
                   failure_probability, message, acknowledged
            FROM alerts WHERE machine_id = :mid ORDER BY triggered_at DESC LIMIT 100
        """)
        params = {"mid": machine_id}
    else:
        query = text("""
            SELECT id, machine_id, triggered_at, risk_state, predicted_rul,
                   failure_probability, message, acknowledged
            FROM alerts ORDER BY triggered_at DESC LIMIT 100
        """)
        params = {}
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


st.title("🛠️ Predictive Maintenance — Fleet Operations")

try:
    latest = load_latest_predictions()
except Exception as e:
    st.error(
        "Could not read from the database. Make sure Postgres is running, "
        "ingestion + training have been run, and the API has served at least "
        f"one /predict call so predictions exist.\n\nDetails: {e}"
    )
    st.stop()

if latest.empty:
    st.info("No predictions yet. Run the ingestion + training pipeline, then call "
            "POST /predict at least once (or write a small backfill loop) to populate this view.")
    st.stop()

# --- Fleet overview ----------------------------------------------------------
st.subheader("Fleet risk overview")
counts = latest["risk_state"].value_counts().reindex(["OK", "WATCH", "WARNING", "CRITICAL"]).fillna(0)
cols = st.columns(4)
for col, state in zip(cols, ["OK", "WATCH", "WARNING", "CRITICAL"]):
    col.metric(state, int(counts[state]))

fig_overview = px.bar(
    latest.sort_values("failure_probability", ascending=False),
    x="machine_id", y="failure_probability", color="risk_state",
    color_discrete_map=RISK_COLORS,
    title="Failure probability by machine (latest prediction)",
    labels={"failure_probability": "P(failure within window)", "machine_id": "Machine"},
)
st.plotly_chart(fig_overview, use_container_width=True)

st.dataframe(
    latest.sort_values("risk_state", key=lambda s: s.map({"CRITICAL": 0, "WARNING": 1, "WATCH": 2, "OK": 3}))
          .rename(columns={"predicted_rul": "RUL (cycles)", "failure_probability": "Failure prob."}),
    use_container_width=True,
)

# --- Machine drill-down --------------------------------------------------
st.subheader("Machine drill-down")
machine_id = st.selectbox("Select a machine", sorted(latest["machine_id"].unique()))

sensor_df = load_sensor_history(machine_id)
history_df = load_machine_history(machine_id)

tab_sensors, tab_risk, tab_alerts = st.tabs(["Sensor trends", "Risk history", "Alerts"])

with tab_sensors:
    if sensor_df.empty:
        st.info("No raw sensor readings loaded for this machine yet.")
    else:
        for sensor in SENSOR_COLUMNS:
            fig = px.line(sensor_df, x="cycle", y=sensor, title=sensor)
            st.plotly_chart(fig, use_container_width=True)

with tab_risk:
    if history_df.empty:
        st.info("No prediction history for this machine yet — call /predict to generate some.")
    else:
        fig_rul = px.line(history_df, x="timestamp", y="predicted_rul", title="Predicted RUL over time")
        st.plotly_chart(fig_rul, use_container_width=True)
        fig_prob = px.line(history_df, x="timestamp", y="failure_probability",
                            title="Failure probability over time")
        st.plotly_chart(fig_prob, use_container_width=True)

with tab_alerts:
    alerts_df = load_alerts(machine_id)
    if alerts_df.empty:
        st.info("No alerts for this machine.")
    else:
        st.dataframe(alerts_df, use_container_width=True)

# --- Fleet-wide alert log --------------------------------------------------
st.subheader("Recent fleet-wide alerts")
all_alerts = load_alerts()
if all_alerts.empty:
    st.info("No alerts logged yet.")
else:
    st.dataframe(all_alerts, use_container_width=True)

st.caption(f"API base: {API_BASE} · Data refreshes every 30s (cached).")