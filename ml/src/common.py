from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else project_root() / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    root = project_root()
    artifact_dir = root / config["paths"]["artifact_dir"]
    report_dir = root / config["paths"]["report_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir, report_dir


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def normalize_attack_labels(frame, target_col: str, binary_label_col: str):
    frame = frame.copy()
    if target_col not in frame.columns:
        if binary_label_col not in frame.columns:
            raise ValueError(
                f"Dataset must contain '{target_col}' or '{binary_label_col}'."
            )
        frame[target_col] = frame[binary_label_col].map({0: "Normal", 1: "UnknownAttack"})

    labels = frame[target_col].astype("string").str.strip()
    is_missing = labels.isna() | labels.eq("") | labels.eq("<NA>")
    if binary_label_col in frame.columns:
        labels = labels.mask(is_missing & frame[binary_label_col].eq(0), "Normal")

    frame[target_col] = labels.fillna("UnknownAttack").str.title()
    frame[target_col] = frame[target_col].replace(
        {
            "Dos": "DoS",
            "Reconnaissance": "Reconnaissance",
            "Backdoor": "Backdoors",
            "Web Attack": "WebAttack",
        }
    )
    return frame
