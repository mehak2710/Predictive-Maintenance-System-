from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import settings

FEATURES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "features.parquet"
MODEL_DIR = Path(settings.model_dir)


def main():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("predictive-maintenance")

    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")

    # Anchor "normal" on each machine's early life
    df["life_fraction"] = 1 - (df["rul"] / df.groupby("machine_id")["rul"].transform("max"))
    healthy = df[df["life_fraction"] <= 0.6]

    with mlflow.start_run(run_name="isolation_forest_anomaly"):
        contamination = 0.03
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_healthy_rows", len(healthy))

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )
        model.fit(healthy[feature_cols])

        # sanity check: what fraction of held-out (later-life) rows get flagged
        late_life = df[df["life_fraction"] > 0.6]
        preds = model.predict(late_life[feature_cols])
        flagged_rate = (preds == -1).mean()
        mlflow.log_metric("late_life_flagged_rate", flagged_rate)
        print(f"Anomaly rate among later-life readings: {flagged_rate:.1%} "
              f"(expected to be higher than {contamination:.0%} baseline, since "
              f"degrading machines look less 'normal')")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "isolation_forest.joblib")
        mlflow.log_artifact(str(MODEL_DIR / "isolation_forest.joblib"))
        print(f"Saved anomaly model to {MODEL_DIR / 'isolation_forest.joblib'}")


if __name__ == "__main__":
    main()