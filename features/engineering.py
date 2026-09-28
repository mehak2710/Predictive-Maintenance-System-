from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SENSOR_COLUMNS, ROLLING_WINDOWS, FFT_WINDOW, FAILURE_WINDOW_CYCLES

RAW_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "sensor_readings.csv"
FEATURES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "features.parquet"


def _rolling_features(g: pd.DataFrame) -> pd.DataFrame:
    """Rolling stats + rate-of-change for one machine's sensor columns, in cycle order."""
    g = g.sort_values("cycle").reset_index(drop=True)
    out = {}

    for sensor in SENSOR_COLUMNS:
        series = g[sensor]

        # Rate of change
        out[f"{sensor}_diff1"] = series.diff().fillna(0)

        for w in ROLLING_WINDOWS:
            roll = series.rolling(window=w, min_periods=1)
            out[f"{sensor}_roll{w}_mean"] = roll.mean()
            out[f"{sensor}_roll{w}_std"] = roll.std().fillna(0)
            out[f"{sensor}_roll{w}_min"] = roll.min()
            out[f"{sensor}_roll{w}_max"] = roll.max()
            # rolling slope: change over the window, normalized by window size
            out[f"{sensor}_roll{w}_slope"] = (series - series.shift(w)).fillna(0) / w

    return pd.DataFrame(out, index=g.index)


def _fft_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    Sliding-window FFT features per sensor: dominant frequency magnitude and
    spectral energy, computed over the trailing FFT_WINDOW readings.
    Early cycles (before a full window exists) are backfilled with zeros.
    """
    g = g.sort_values("cycle").reset_index(drop=True)
    n = len(g)
    out = {f"{sensor}_fft_dom_mag": np.zeros(n) for sensor in SENSOR_COLUMNS}
    out.update({f"{sensor}_fft_energy": np.zeros(n) for sensor in SENSOR_COLUMNS})

    for sensor in SENSOR_COLUMNS:
        values = g[sensor].to_numpy()
        for end in range(FFT_WINDOW, n + 1):
            window = values[end - FFT_WINDOW:end]
            window = window - window.mean()  # remove DC component
            spectrum = np.abs(np.fft.rfft(window))
            spectrum = spectrum[1:]  # drop the (now near-zero) DC bin
            out[f"{sensor}_fft_dom_mag"][end - 1] = spectrum.max() if len(spectrum) else 0.0
            out[f"{sensor}_fft_energy"][end - 1] = float(np.sum(spectrum ** 2))

    return pd.DataFrame(out, index=g.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_frames = []
    for machine_id, g in df.groupby("machine_id", sort=False):
        rolling = _rolling_features(g)
        fft = _fft_features(g)
        base = g.sort_values("cycle").reset_index(drop=True)[
            ["machine_id", "machine_type", "cycle", "timestamp", "rul"]
        ]
        combined = pd.concat([base, rolling, fft], axis=1)
        feature_frames.append(combined)

    features = pd.concat(feature_frames, ignore_index=True)

    # Labels derived from RUL, used by the two model heads
    features["will_fail_soon"] = (features["rul"] <= FAILURE_WINDOW_CYCLES).astype(int)

    return features


def main():
    print(f"Reading raw readings from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)

    print("Building rolling / rate-of-change / FFT features ...")
    features = build_features(df)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)
    print(f"Wrote {features.shape[0]:,} rows x {features.shape[1]} columns -> {FEATURES_PATH}")


if __name__ == "__main__":
    main()