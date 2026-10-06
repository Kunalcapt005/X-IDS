from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    validation_size: float = 0.2,
    random_state: int = 42,
):
    """Create a reproducible validation split from the official training partition.

    The official UNSW-NB15 testing partition remains untouched for final evaluation.
    This is a stratified row split for the initial skeleton. Before final reporting,
    run a dataset-specific leakage study for attack bursts/session correlations and
    replace this with grouped/time-window splitting when supported by the source data.
    """
    return train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
        stratify=y,
    )
