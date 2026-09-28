from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error, roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import GroupShuffleSplit

from src.config import settings

FEATURES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "features.parquet"
MODEL_DIR = Path(settings.model_dir)
NON_FEATURE_COLS = {"machine_id", "machine_type", "cycle", "timestamp", "rul", "will_fail_soon"}


def load_features():
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df, feature_cols


def split_by_machine(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["machine_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def train_regressor(X_train, y_train, X_test, y_test) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def train_classifier(X_train, y_train, X_test, y_test) -> xgb.XGBClassifier:
    # class imbalance: failure-window rows are a minority
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def main():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("predictive-maintenance")

    df, feature_cols = load_features()
    train_df, test_df = split_by_machine(df)

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    y_rul_train, y_rul_test = train_df["rul"], test_df["rul"]
    y_clf_train, y_clf_test = train_df["will_fail_soon"], test_df["will_fail_soon"]

    with mlflow.start_run(run_name="xgb_rul_and_failure_window"):
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train_rows", len(train_df))
        mlflow.log_param("n_test_rows", len(test_df))
        mlflow.log_param("n_train_machines", train_df["machine_id"].nunique())
        mlflow.log_param("n_test_machines", test_df["machine_id"].nunique())

        print("Training RUL regressor ...")
        reg = train_regressor(X_train, y_rul_train, X_test, y_rul_test)
        rul_pred = reg.predict(X_test)
        mae = mean_absolute_error(y_rul_test, rul_pred)
        mlflow.log_metric("rul_mae_cycles", mae)
        print(f"  RUL MAE: {mae:.2f} cycles")

        print("Training failure-window classifier ...")
        clf = train_classifier(X_train, y_clf_train, X_test, y_clf_test)
        clf_proba = clf.predict_proba(X_test)[:, 1]
        clf_pred = (clf_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_clf_test, clf_proba)
        precision = precision_score(y_clf_test, clf_pred, zero_division=0)
        recall = recall_score(y_clf_test, clf_pred, zero_division=0)
        f1 = f1_score(y_clf_test, clf_pred, zero_division=0)

        mlflow.log_metric("failure_window_auc", auc)
        mlflow.log_metric("failure_window_precision", precision)
        mlflow.log_metric("failure_window_recall", recall)
        mlflow.log_metric("failure_window_f1", f1)
        print(f"  Failure-window AUC: {auc:.3f} | precision: {precision:.3f} | recall: {recall:.3f}")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(reg, MODEL_DIR / "rul_regressor.joblib")
        joblib.dump(clf, MODEL_DIR / "failure_classifier.joblib")
        joblib.dump(feature_cols, MODEL_DIR / "feature_columns.joblib")

        mlflow.log_artifact(str(MODEL_DIR / "rul_regressor.joblib"))
        mlflow.log_artifact(str(MODEL_DIR / "failure_classifier.joblib"))
        mlflow.log_artifact(str(MODEL_DIR / "feature_columns.joblib"))

        print(f"Saved models to {MODEL_DIR}")


if __name__ == "__main__":
    main()