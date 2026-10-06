from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


@dataclass
class ModelSpec:
    name: str
    estimator_factory: Any


def build_models(config: dict) -> list[ModelSpec]:
    lr_cfg = config["models"]["logistic_regression"]
    rf_cfg = config["models"]["random_forest"]
    xgb_cfg = config["models"]["xgboost"]

    return [
        ModelSpec(
            "logistic_regression",
            lambda: LogisticRegression(
                max_iter=lr_cfg.get("max_iter", 2000),
                C=lr_cfg.get("C", 1.0),
            ),
        ),
        ModelSpec(
            "random_forest",
            lambda: RandomForestClassifier(
                n_estimators=rf_cfg.get("n_estimators", 350),
                random_state=rf_cfg.get("random_state", 42),
                n_jobs=rf_cfg.get("n_jobs", -1),
                class_weight=None,
            ),
        ),
        ModelSpec(
            "xgboost",
            lambda: XGBClassifier(
                n_estimators=xgb_cfg.get("n_estimators", 400),
                max_depth=xgb_cfg.get("max_depth", 8),
                learning_rate=xgb_cfg.get("learning_rate", 0.08),
                subsample=xgb_cfg.get("subsample", 0.9),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.9),
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=xgb_cfg.get("random_state", 42),
                n_jobs=xgb_cfg.get("n_jobs", -1),
                tree_method="hist",
            ),
        ),
    ]
