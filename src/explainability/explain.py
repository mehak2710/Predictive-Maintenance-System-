from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import shap

from src.config import settings

MODEL_DIR = Path(settings.model_dir)


class FailureExplainer:
    def __init__(self):
        self.classifier = joblib.load(MODEL_DIR / "failure_classifier.joblib")
        self.feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
        self._explainer = shap.TreeExplainer(self.classifier)

    def top_drivers(self, feature_row: np.ndarray, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        feature_row: 1D array matching self.feature_cols order.
        Returns [(feature_name, shap_value), ...] sorted by |impact|, largest first.
        Positive shap_value = pushes failure probability up.
        """
        shap_values = self._explainer.shap_values(feature_row.reshape(1, -1))
        values = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        pairs = list(zip(self.feature_cols, values))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs[:top_n]