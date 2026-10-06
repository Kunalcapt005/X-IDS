from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy.orm import Session

from ..core.config import settings
from ..models.core import ModelVersion


class ModelRegistry:
    """Loads immutable model bundles from the X-IDS artifact directory."""

    def __init__(self, artifact_dir: str | None = None):
        configured = Path(artifact_dir or settings.ml_artifact_dir)
        if not configured.is_absolute():
            configured = Path(__file__).resolve().parents[3] / configured
        self.artifact_dir = configured.resolve()
        self._metadata_path = self.artifact_dir / "feature_metadata.json"
        self._comparison_path = self.artifact_dir / "training_comparison.json"

    @lru_cache(maxsize=8)
    def load_bundle(self, model_file: str) -> dict[str, Any]:
        self._validate_model_name(model_file)
        path = self.artifact_dir / model_file
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        return joblib.load(path)

    def feature_columns(self) -> list[str]:
        if not self._metadata_path.exists():
            raise FileNotFoundError(f"Feature metadata not found: {self._metadata_path}")
        data = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        return list(data.get("feature_columns", []))

    def available_models(self) -> list[str]:
        if not self.artifact_dir.exists():
            return []
        names = {
            path.name
            for path in self.artifact_dir.glob("*.joblib")
            if path.is_file() and ("__" in path.name or path.name == "selected_model.joblib")
        }
        return sorted(names)

    def default_model(self) -> str:
        preferred = self.artifact_dir / "selected_model.joblib"
        if preferred.exists():
            return preferred.name
        models = self.available_models()
        if not models:
            raise FileNotFoundError(f"No model artifacts found in {self.artifact_dir}")
        return models[0]

    def model_info(self, model_file: str) -> dict[str, Any]:
        bundle = self.load_bundle(model_file)
        metadata = bundle.get("metadata", {})
        name = model_file.removesuffix(".joblib")
        algorithm = str(metadata.get("model", name.split("__", 1)[0]))
        strategy = str(metadata.get("imbalance_strategy", name.split("__", 1)[1] if "__" in name else "selected"))
        return {
            "model_file": model_file,
            "name": f"{algorithm} + {strategy}",
            "algorithm": algorithm,
            "strategy": strategy,
            "feature_count": metadata.get("feature_count"),
            "evaluation_metrics": self._metrics_for(model_file),
        }

    def sync_model_versions(self, db: Session) -> dict[str, int]:
        """Ensure every local joblib has a matching database model version."""
        mapping: dict[str, int] = {}
        feature_list = self.feature_columns()
        for model_file in self.available_models():
            info = self.model_info(model_file)
            row = db.query(ModelVersion).filter(ModelVersion.model_artifact_path == model_file).first()
            if row is None:
                row = ModelVersion(
                    name=info["name"],
                    algorithm=info["algorithm"],
                    feature_list=feature_list,
                    model_artifact_path=model_file,
                    scaler_artifact_path="preprocessor.joblib",
                    evaluation_metrics=info["evaluation_metrics"],
                )
                db.add(row)
                db.flush()
            mapping[model_file] = row.id
        db.commit()
        return mapping

    def model_file_from_version(self, db: Session, model_version_id: int) -> str:
        row = db.get(ModelVersion, model_version_id)
        if row is None or not row.model_artifact_path:
            raise ValueError(f"Model version {model_version_id} was not found")
        return row.model_artifact_path

    def _metrics_for(self, model_file: str) -> dict[str, Any]:
        if not self._comparison_path.exists():
            return {}
        try:
            data = json.loads(self._comparison_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        name = model_file.removesuffix(".joblib")
        if model_file == "selected_model.joblib":
            selected = data.get("selected") or {}
            return {
                "validation": selected.get("validation", {}),
                "test": selected.get("test", {}),
                "selection_key": selected.get("selection_key"),
            }
        model, _, strategy = name.partition("__")
        for row in data.get("runs", []):
            if row.get("model") == model and row.get("strategy") == strategy:
                return {
                    "validation": row.get("validation", {}),
                    "test": row.get("test", {}),
                    "selection_key": row.get("selection_key"),
                }
        return {}

    @staticmethod
    def _validate_model_name(model_file: str) -> None:
        # Normalize both Windows and POSIX separators before validating so a
        # filename cannot escape the artifact directory on either platform.
        normalized = model_file.replace("\\", "/")
        name = Path(normalized).name
        if normalized != name or not name.endswith(".joblib"):
            raise ValueError("model_file must be a .joblib filename from the model artifact directory")
