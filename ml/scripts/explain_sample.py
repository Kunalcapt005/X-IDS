from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.explainability.shap_explainer import TreeShapExplainer


def main():
    parser = argparse.ArgumentParser(description="Generate a SHAP explanation for one transformed test row")
    parser.add_argument("--model", default="xgboost__baseline.joblib")
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()

    bundle = joblib.load(ML_ROOT / "artifacts" / args.model)
    X_test = joblib.load(ML_ROOT / "artifacts" / "X_test_transformed.joblib")
    feature_names = bundle["preprocessor"].get_feature_names_out().tolist()
    labels = bundle["label_encoder"].classes_.tolist()
    model = bundle["model"]

    if not isinstance(X_test, np.ndarray):
        X_test = X_test.toarray()
    X_row = X_test[args.row : args.row + 1]
    predicted_index = int(np.argmax(model.predict_proba(X_row)[0]))

    explainer = TreeShapExplainer(model, feature_names, labels)
    explanation = explainer.explain(X_row, predicted_index)

    print(explanation.text)
    for feature in explanation.top_features:
        print(
            f"{feature['feature']}: {feature['shap_value']:+.6f} ({feature['direction']})"
        )


if __name__ == "__main__":
    main()
