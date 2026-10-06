from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.common import ensure_dirs, load_config, save_json
from src.evaluation.metrics import evaluate_classifier, selection_key
from src.evaluation.report import save_confusion_matrix_png, save_evaluation_report
from src.features.build import build_preprocessor, infer_feature_schema
from src.features.selection import prune_redundant_features
from src.models.catalog import build_models
from src.models.train import fit_model, save_model_bundle
from sklearn.preprocessing import LabelEncoder
from src.preprocessing.clean import clean_frame
from src.preprocessing.load_data import load_unsw_nb15
from src.split import split_training_data


def load_frames(config):
    root = ML_ROOT
    train_path = root / config["paths"]["train_csv"]
    test_path = root / config["paths"]["test_csv"]
    train, test = load_unsw_nb15(train_path, test_path)
    return clean_frame(train), clean_frame(test)


def preprocess(config):
    artifact_dir, report_dir = ensure_dirs(config)
    train, test = load_frames(config)

    target = config["columns"]["target"]
    drop_columns = config["columns"]["drop"]
    forced_cats = config["columns"]["categorical"]

    X_raw = train.drop(columns=[target]).copy()
    y_raw = train[target].copy()
    X_test_raw = test.drop(columns=[target]).copy()
    y_test_raw = test[target].copy()

    X_fit_raw, X_val_raw, y_fit_raw, y_val_raw = split_training_data(
        X_raw,
        y_raw,
        validation_size=config["training"]["validation_size"],
        random_state=config["training"]["random_state"],
    )

    # Feature pruning is fit only on the training subset so validation/test information
    # cannot influence the selected feature set.
    fit_for_selection, selected_pruned = prune_redundant_features(
        X_fit_raw,
        protected_columns=forced_cats,
    )
    selected_columns = list(fit_for_selection.columns)

    X_fit_raw = X_fit_raw[selected_columns]
    X_val_raw = X_val_raw[selected_columns]
    X_test_raw = X_test_raw[selected_columns]

    schema = infer_feature_schema(
        X_fit_raw.assign(**{target: y_fit_raw.to_numpy()}),
        target,
        drop_columns,
        forced_cats,
    )
    preprocessor = build_preprocessor(schema)

    X_fit_t = preprocessor.fit_transform(X_fit_raw[schema.feature_columns])
    X_val_t = preprocessor.transform(X_val_raw[schema.feature_columns])
    X_test_t = preprocessor.transform(X_test_raw[schema.feature_columns])

    label_encoder = LabelEncoder()
    y_fit = label_encoder.fit_transform(y_fit_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)
    feature_names = list(preprocessor.get_feature_names_out())

    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")
    joblib.dump(label_encoder, artifact_dir / "label_encoder.joblib")
    joblib.dump(X_fit_t, artifact_dir / "X_fit_transformed.joblib")
    joblib.dump(X_val_t, artifact_dir / "X_val_transformed.joblib")
    joblib.dump(X_test_t, artifact_dir / "X_test_transformed.joblib")
    pd.Series(y_fit).to_csv(artifact_dir / "y_fit.csv", index=False)
    pd.Series(y_val).to_csv(artifact_dir / "y_val.csv", index=False)
    pd.Series(y_test).to_csv(artifact_dir / "y_test.csv", index=False)

    metadata = {
        "feature_columns": schema.feature_columns,
        "numeric_columns": schema.numeric_columns,
        "categorical_columns": schema.categorical_columns,
        "transformed_feature_names": feature_names,
        "removed_features": selected_pruned,
        "classes": label_encoder.classes_.tolist(),
        "train_rows": len(y_fit),
        "validation_rows": len(y_val),
        "test_rows": len(y_test),
    }
    save_json(artifact_dir / "feature_metadata.json", metadata)
    save_json(report_dir / "dataset_summary.json", {
        "train_raw_rows": len(train),
        "test_raw_rows": len(test),
        "fit_rows": len(y_fit),
        "validation_rows": len(y_val),
        "test_rows": len(y_test),
        "classes": label_encoder.classes_.tolist(),
        "removed_features": selected_pruned,
    })

    print(f"Preprocessing complete. Fit={X_fit_t.shape}, Val={X_val_t.shape}, Test={X_test_t.shape}")
    print(f"Classes: {label_encoder.classes_.tolist()}")
    print(f"Artifacts: {artifact_dir}")


def train(config):
    artifact_dir, report_dir = ensure_dirs(config)
    X_fit = joblib.load(artifact_dir / "X_fit_transformed.joblib")
    X_val = joblib.load(artifact_dir / "X_val_transformed.joblib")
    X_test = joblib.load(artifact_dir / "X_test_transformed.joblib")
    y_fit = pd.read_csv(artifact_dir / "y_fit.csv").iloc[:, 0]
    y_val = pd.read_csv(artifact_dir / "y_val.csv").iloc[:, 0]
    y_test = pd.read_csv(artifact_dir / "y_test.csv").iloc[:, 0]
    preprocessor = joblib.load(artifact_dir / "preprocessor.joblib")
    label_encoder = joblib.load(artifact_dir / "label_encoder.joblib")

    rare_classes = config["training"]["rare_classes"]
    results = []
    best = None
    best_estimator = None
    best_spec = None

    # IMPORTANT: the official test partition is not consulted during model/strategy
    # selection. All 12 candidates are ranked using validation metrics only.
    for model_spec in build_models(config):
        for strategy in config["imbalance"]["strategies"]:
            estimator = model_spec.estimator_factory()
            fit_model(estimator, X_fit, y_fit, strategy, config["training"]["random_state"])
            val_metrics = evaluate_classifier(estimator, X_val, y_val, label_encoder)
            row = {
                "model": model_spec.name,
                "strategy": strategy,
                "validation": val_metrics,
                "selection_key": selection_key(val_metrics, rare_classes),
            }
            results.append(row)

            if best is None or row["selection_key"] > best["selection_key"]:
                best = row
                best_estimator = estimator
                best_spec = model_spec

            filename = f"{model_spec.name}__{strategy}.joblib"
            save_model_bundle(
                artifact_dir / filename,
                preprocessor,
                label_encoder,
                estimator,
                {
                    "model": model_spec.name,
                    "imbalance_strategy": strategy,
                    "labels": label_encoder.classes_.tolist(),
                    "feature_count": X_fit.shape[1],
                    "validation_selection_key": row["selection_key"],
                    "evaluation_role": "validation_candidate",
                },
            )
            print(model_spec.name, strategy, "validation macro-F1=", round(val_metrics["macro_f1"], 4))

    if best is None or best_estimator is None or best_spec is None:
        raise RuntimeError("No model/strategy candidates were trained.")

    # The test partition is evaluated exactly once, after selection, to preserve
    # a strict held-out final evaluation.
    final_test_metrics = evaluate_classifier(best_estimator, X_test, y_test, label_encoder)
    best["test"] = final_test_metrics
    best["evaluation_role"] = "selected_final_model"

    selected_filename = "selected_model.joblib"
    save_model_bundle(
        artifact_dir / selected_filename,
        preprocessor,
        label_encoder,
        best_estimator,
        {
            "model": best_spec.name,
            "imbalance_strategy": best["strategy"],
            "labels": label_encoder.classes_.tolist(),
            "feature_count": X_fit.shape[1],
            "validation_selection_key": best["selection_key"],
            "test_metrics": final_test_metrics,
            "evaluation_role": "selected_final_model",
        },
    )

    save_confusion_matrix_png(
        report_dir,
        "selected_model__test_confusion",
        final_test_metrics["confusion_matrix"],
        final_test_metrics["labels"],
    )

    save_json(
        artifact_dir / "training_comparison.json",
        {
            "runs": results,
            "selected": best,
            "test_evaluation_note": "Test metrics are reported only for the validation-selected final model.",
        },
    )
    print(
        "Selected run by validation criteria: "
        f"{best['model']} + {best['strategy']}"
    )
    print(
        "Final held-out test macro-F1=",
        round(final_test_metrics["macro_f1"], 4),
    )


def evaluate(config):
    artifact_dir, report_dir = ensure_dirs(config)
    comparison_path = artifact_dir / "training_comparison.json"
    if not comparison_path.exists():
        raise FileNotFoundError("Run the train stage before evaluate.")
    comparison = joblib.load(comparison_path) if comparison_path.suffix == ".joblib" else None
    if comparison is None:
        import json
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    save_evaluation_report(report_dir, "model_comparison", comparison)
    print(f"Evaluation report copied to {report_dir / 'model_comparison.json'}")


def main():
    parser = argparse.ArgumentParser(description="X-IDS ML pipeline")
    parser.add_argument("--stage", choices=["preprocess", "train", "evaluate"], required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)

    if args.stage == "preprocess":
        preprocess(config)
    elif args.stage == "train":
        train(config)
    else:
        evaluate(config)


if __name__ == "__main__":
    main()
