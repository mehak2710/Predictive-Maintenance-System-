import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests

from src.config import settings, SENSOR_COLUMNS
from src.database.db import engine

API_URL = f"http://localhost:{settings.api_port}/predict"
MIN_WINDOW = 60      # matches FFT_WINDOW so features are fully populated
STEP = 15            # take a snapshot every N cycles per machine
CUTOFF_RANGE = (0.35, 0.98)   # fraction of each machine's life to replay
SEED = 7             # fixed so the demo fleet looks the same every run


def main():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM sensor_readings ORDER BY machine_id, cycle", conn)
        machines = pd.read_sql("SELECT machine_id, machine_type FROM machines", conn)

    machine_types = dict(zip(machines["machine_id"], machines["machine_type"]))
    rng = np.random.default_rng(SEED)

    for machine_id, g in df.groupby("machine_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        if len(g) < MIN_WINDOW + STEP:
            continue

        # Stop at a random point in this machine's life
        cutoff = max(int(len(g) * rng.uniform(*CUTOFF_RANGE)), MIN_WINDOW)
        g = g.iloc[:cutoff]

        ends = list(range(MIN_WINDOW, len(g) + 1, STEP))
        print(f"{machine_id}: {len(ends)} snapshots (replaying {cutoff} cycles)")

        for end in ends:
            window = g.iloc[end - MIN_WINDOW:end]
            readings = [
                {
                    "timestamp": row["timestamp"].isoformat(),
                    "cycle": int(row["cycle"]),
                    **{s: float(row[s]) for s in SENSOR_COLUMNS},
                }
                for _, row in window.iterrows()
            ]
            payload = {
                "machine_id": machine_id,
                "machine_type": machine_types.get(machine_id, "unknown"),
                "readings": readings,
            }
            try:
                resp = requests.post(API_URL, json=payload, timeout=10)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"  failed at cycle {end}: {e}")
                continue

        time.sleep(0.05)  # be gentle on the API during backfill

    print("Backfill complete. Open the Streamlit dashboard to view results.")


if __name__ == "__main__":
    main()