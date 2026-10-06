from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import normalize_attack_labels


EXPECTED_ATTACK_CATEGORIES = {
    "Normal",
    "Fuzzers",
    "Analysis",
    "Backdoors",
    "DoS",
    "Exploits",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Put the official UNSW-NB15 CSV at this path."
        )
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_unsw_nb15(train_path: str | Path, test_path: str | Path):
    train = _read_csv(Path(train_path))
    test = _read_csv(Path(test_path))

    train = normalize_attack_labels(train, "attack_cat", "label")
    test = normalize_attack_labels(test, "attack_cat", "label")

    invalid_train = sorted(set(train["attack_cat"].dropna().unique()) - EXPECTED_ATTACK_CATEGORIES)
    invalid_test = sorted(set(test["attack_cat"].dropna().unique()) - EXPECTED_ATTACK_CATEGORIES)
    if invalid_train or invalid_test:
        raise ValueError(
            "Unexpected attack categories found. "
            f"train={invalid_train}, test={invalid_test}"
        )

    return train, test
