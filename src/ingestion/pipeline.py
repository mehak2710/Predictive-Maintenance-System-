import sys
from pathlib import Path

import pandas as pd

from src.config import settings, SENSOR_COLUMNS
from src.database.db import init_db, get_session
from src.database.models import Machine, SensorReading

RAW_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "sensor_readings.csv"

REQUIRED_COLUMNS = {"machine_id", "machine_type", "cycle", "timestamp", *SENSOR_COLUMNS}


class SchemaValidationError(Exception):
    pass


def validate(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaValidationError(f"Missing required columns: {sorted(missing)}")

    null_counts = df[list(REQUIRED_COLUMNS)].isnull().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise SchemaValidationError(f"Null values found in required columns:\n{bad}")

    for sensor in SENSOR_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[sensor]):
            raise SchemaValidationError(f"Sensor column '{sensor}' is not numeric")


def load(df: pd.DataFrame, chunk_size: int = 5000) -> None:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    machines = df[["machine_id", "machine_type"]].drop_duplicates()
    with get_session() as session:
        existing = {m.machine_id for m in session.query(Machine.machine_id).all()}
        for _, row in machines.iterrows():
            if row["machine_id"] not in existing:
                session.add(Machine(machine_id=row["machine_id"], machine_type=row["machine_type"]))

    records = df[["machine_id", "timestamp", "cycle", *SENSOR_COLUMNS]].to_dict(orient="records")
    with get_session() as session:
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            session.bulk_insert_mappings(SensorReading, chunk)
            print(f"  inserted rows {i}-{i + len(chunk)} / {len(records)}")


def main():
    if not RAW_PATH.exists():
        print(f"Raw data not found at {RAW_PATH}. Run data/generate_synthetic_data.py first.")
        sys.exit(1)

    print(f"Reading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)

    print("Validating schema ...")
    validate(df)

    print("Initializing database (create tables if needed) ...")
    init_db()

    print(f"Loading {len(df):,} readings for {df['machine_id'].nunique()} machines into Postgres ...")
    load(df)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()