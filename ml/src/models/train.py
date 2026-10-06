from __future__ import annotations

from pathlib import Path

import joblib

from src.imbalance.strategies import balanced_sample_weights, apply_resampling


def fit_model(estimator, X_train, y_train, strategy: str, random_state: int = 42):
    X_fit, y_fit, _ = apply_resampling(X_train, y_train, strategy, random_state)

    fit_kwargs = {}
    if strategy == "class_weight":
        fit_kwargs["sample_weight"] = balanced_sample_weights(y_fit)

    estimator.fit(X_fit, y_fit, **fit_kwargs)
    return estimator


def save_model_bundle(path: str | Path, preprocessor, label_encoder, estimator, metadata: dict):
    bundle = {
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "model": estimator,
        "metadata": metadata,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
