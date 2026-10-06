from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common import save_json


def save_evaluation_report(report_dir: str | Path, name: str, payload: dict):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    save_json(report_dir / f"{name}.json", payload)


def save_confusion_matrix_png(report_dir: str | Path, name: str, matrix, labels: list[str]):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(normalized, aspect="auto")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(report_dir / f"{name}.png", dpi=160)
    plt.close(fig)
