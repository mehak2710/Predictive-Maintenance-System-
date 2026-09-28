import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT_DIR = Path(__file__).resolve().parent / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_MACHINES = 40
MIN_LIFETIME, MAX_LIFETIME = 150, 400   # cycles per machine
N_ATYPICAL = 4                            # machines with an unusual failure shape

MACHINE_TYPES = ["pump", "compressor", "motor", "conveyor"]

# Healthy baseline (mean, std) per sensor
BASELINE = {
    "vibration_mm_s": (2.0, 0.15),
    "temperature_c": (55.0, 1.5),
    "pressure_bar": (8.0, 0.2),
    "rpm": (1500.0, 15.0),
    "current_amp": (12.0, 0.4),
}

# How much each sensor drifts by end-of-life, as a multiple of baseline mean
DEGRADATION_FACTOR = {
    "vibration_mm_s": 3.2,   # vibration rises sharply near failure
    "temperature_c": 1.35,
    "pressure_bar": 0.55,    # pressure tends to drop
    "rpm": 0.85,             # rpm sags under load
    "current_amp": 1.6,      # current draw climbs
}


def degradation_curve(n_cycles: int, atypical: bool) -> np.ndarray:
    """Returns a 0->1 'wear' curve. Mostly flat, then accelerates."""
    t = np.linspace(0, 1, n_cycles)
    if atypical:
        # sudden-onset failure: flat, then a sharp late ramp with a plateau blip
        wear = np.where(t < 0.75, 0.05 * t, (t - 0.75) / 0.25) ** 1.8
        wear += 0.05 * np.sin(t * 40) * (t > 0.75)  # odd oscillation near failure
    else:
        # gradual exponential-ish wear, typical bathtub-tail shape
        wear = t ** 2.5
    wear = np.clip(wear, 0, 1)
    return wear


def simulate_machine(machine_id: str, machine_type: str, atypical: bool) -> pd.DataFrame:
    n_cycles = RNG.integers(MIN_LIFETIME, MAX_LIFETIME)
    wear = degradation_curve(n_cycles, atypical)
    rows = {"machine_id": machine_id, "machine_type": machine_type,
            "cycle": np.arange(n_cycles),
            "rul": (n_cycles - 1) - np.arange(n_cycles)}  # remaining useful life

    for sensor, (mean, std) in BASELINE.items():
        factor = DEGRADATION_FACTOR[sensor]
        # direction of drift: some sensors rise, some fall as they degrade
        direction = -1 if factor < 1 else 1
        drift = mean * (factor - 1) * wear * direction
        # noise grows as the machine wears down (less stable near failure)
        noise_scale = std * (1 + 2.5 * wear)
        noise = RNG.normal(0, 1, n_cycles) * noise_scale
        rows[sensor] = mean + drift + noise

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.date_range("2025-01-01", periods=n_cycles, freq="h")
    df["is_atypical_unit"] = atypical  # ground-truth flag, not used in training
    return df


def main():
    frames = []
    atypical_ids = set(RNG.choice(N_MACHINES, size=N_ATYPICAL, replace=False))
    for i in range(N_MACHINES):
        machine_id = f"M{i:03d}"
        machine_type = MACHINE_TYPES[i % len(MACHINE_TYPES)]
        frames.append(simulate_machine(machine_id, machine_type, i in atypical_ids))

    full = pd.concat(frames, ignore_index=True)
    out_path = OUT_DIR / "sensor_readings.csv"
    full.to_csv(out_path, index=False)
    print(f"Wrote {len(full):,} readings across {N_MACHINES} machines -> {out_path}")
    print(f"Atypical-degradation machines: {sorted(f'M{i:03d}' for i in atypical_ids)}")


if __name__ == "__main__":
    main()
