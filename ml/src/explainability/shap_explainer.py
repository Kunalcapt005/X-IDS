from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import shap


@dataclass
class Explanation:
    predicted_label: str
    confidence: float
    top_features: list[dict[str, Any]]
    text: str


class TreeShapExplainer:
    def __init__(self, model, feature_names: list[str], class_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X_row: np.ndarray, predicted_index: int) -> Explanation:
        prediction = self.model.predict_proba(X_row)[0]
        shap_values = self.explainer.shap_values(X_row)
        values = self._extract_class_values(shap_values, predicted_index)

        top = np.argsort(np.abs(values))[::-1][:10]
        top_features = [
            {
                "feature": self.feature_names[index],
                "shap_value": float(values[index]),
                "direction": "toward" if values[index] >= 0 else "away",
                "magnitude": float(abs(values[index])),
            }
            for index in top
        ]
        label = self.class_names[predicted_index]
        confidence = float(prediction[predicted_index])
        text = self._render_text(label, confidence, top_features)
        return Explanation(label, confidence, top_features, text)

    @staticmethod
    def _extract_class_values(shap_values, class_index: int) -> np.ndarray:
        if isinstance(shap_values, list):
            return np.asarray(shap_values[class_index])[0]
        values = np.asarray(shap_values)
        if values.ndim == 3:
            # Common SHAP layouts: (samples, features, classes) or (samples, classes, features)
            if values.shape[2] > class_index and values.shape[1] != values.shape[2]:
                return values[0, :, class_index]
            return values[0, class_index, :]
        if values.ndim == 2:
            return values[0]
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}")

    @staticmethod
    def _render_text(label: str, confidence: float, top_features: list[dict[str, Any]]) -> str:
        drivers = [item["feature"] for item in top_features[:3]]
        driver_text = ", ".join(drivers)
        if driver_text:
            return (
                f"Classified as {label} ({confidence:.1%} confidence). "
                f"Top attribution features: {driver_text}."
            )
        return f"Classified as {label} ({confidence:.1%} confidence)."
