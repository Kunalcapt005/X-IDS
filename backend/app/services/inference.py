from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import shap

from .model_registry import ModelRegistry


SEVERITY = {
    "Normal": "low",
    "Fuzzers": "medium",
    "Analysis": "medium",
    "Backdoors": "critical",
    "DoS": "high",
    "Exploits": "critical",
    "Generic": "medium",
    "Reconnaissance": "medium",
    "Shellcode": "critical",
    "Worms": "critical",
}


class InferenceService:
    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()

    @lru_cache(maxsize=8)
    def _explainer(self, model_file: str):
        model = self.registry.load_bundle(model_file)["model"]
        return shap.TreeExplainer(model)

    @lru_cache(maxsize=1)
    def _feature_columns(self) -> tuple[str, ...]:
        # The training pipeline stores the raw feature schema separately in
        # feature_metadata.json. Older model bundles do not embed it inside
        # metadata, so inference must not depend on that optional field.
        metadata_path = self.registry.artifact_dir / "feature_metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Feature metadata not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        columns = metadata.get("feature_columns")
        if not columns:
            raise ValueError("feature_metadata.json does not contain feature_columns")
        return tuple(str(column) for column in columns)

    def predict(self, model_file: str, feature_payload: dict[str, Any]) -> dict[str, Any]:
        bundle = self.registry.load_bundle(model_file)
        preprocessor = bundle["preprocessor"]
        label_encoder = bundle["label_encoder"]
        model = bundle["model"]

        bundle_metadata = bundle.get("metadata", {})
        raw_columns = list(bundle_metadata.get("feature_columns") or self._feature_columns())

        missing = [column for column in raw_columns if column not in feature_payload]
        if missing:
            preview = ", ".join(missing[:10])
            suffix = "..." if len(missing) > 10 else ""
            raise ValueError(f"Missing required feature(s): {preview}{suffix}")

        # ColumnTransformer was fitted against named DataFrame columns.
        # Preserve those names and the original schema at inference time.
        row = {column: feature_payload[column] for column in raw_columns}
        X_raw = pd.DataFrame([row], columns=raw_columns)
        X = preprocessor.transform(X_raw)

        probabilities = np.asarray(model.predict_proba(X))[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_label = str(label_encoder.inverse_transform([predicted_index])[0])
        confidence = float(probabilities[predicted_index])

        feature_names = preprocessor.get_feature_names_out().tolist()
        X_for_shap = X if isinstance(X, np.ndarray) else X.toarray()
        explainer = self._explainer(model_file)
        shap_values = explainer.shap_values(X_for_shap)
        attribution_values = self._extract_class_values(shap_values, predicted_index)

        top_indices = np.argsort(np.abs(attribution_values))[::-1][:10]
        top_features = [
            {
                "feature": feature_names[index],
                "shap_value": float(attribution_values[index]),
                "direction": "toward" if attribution_values[index] >= 0 else "away",
                "magnitude": float(abs(attribution_values[index])),
            }
            for index in top_indices
        ]

        drivers = ", ".join(item["feature"] for item in top_features[:3])
        explanation = (
            f"Classified as {predicted_label} ({confidence:.1%} confidence). "
            f"Top attribution features: {drivers}."
            if drivers
            else f"Classified as {predicted_label} ({confidence:.1%} confidence)."
        )

        return {
            "model_file": model_file,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probability_distribution": {
                str(label): float(probability)
                for label, probability in zip(label_encoder.classes_, probabilities)
            },
            "severity": SEVERITY.get(predicted_label, "medium"),
            "explanation": explanation,
            "top_features": top_features,
        }

    @staticmethod
    def _extract_class_values(shap_values: Any, class_index: int) -> np.ndarray:
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[class_index])
            return values[0] if values.ndim > 1 else values

        values = np.asarray(shap_values)
        if values.ndim == 3:
            # SHAP versions can expose either (samples, features, classes) or
            # (samples, classes, features) for multiclass tree models.
            if values.shape[2] > class_index and values.shape[1] != values.shape[2]:
                return values[0, :, class_index]
            if values.shape[1] > class_index:
                return values[0, class_index, :]
        if values.ndim == 2:
            return values[0]
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}")
