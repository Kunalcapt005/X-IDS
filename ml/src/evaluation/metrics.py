from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_classifier(model, X, y_encoded, label_encoder) -> dict:
    predictions_encoded = model.predict(X)
    probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    predictions = label_encoder.inverse_transform(predictions_encoded.astype(int))
    y_true = label_encoder.inverse_transform(np.asarray(y_encoded).astype(int))
    labels = list(label_encoder.classes_)

    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, predictions, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        "labels": labels,
    }

    if probabilities is not None:
        try:
            result["roc_auc_ovr"] = float(
                roc_auc_score(y_encoded, probabilities, multi_class="ovr", labels=np.arange(len(labels)))
            )
        except ValueError:
            result["roc_auc_ovr"] = None
    else:
        result["roc_auc_ovr"] = None

    return result


def rare_class_recall(metrics: dict, rare_classes: list[str]) -> float:
    report = metrics["classification_report"]
    recalls = [report[name]["recall"] for name in rare_classes if name in report]
    return float(np.mean(recalls)) if recalls else 0.0


def selection_key(metrics: dict, rare_classes: list[str]):
    return metrics["macro_f1"], rare_class_recall(metrics, rare_classes)
