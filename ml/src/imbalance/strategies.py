from __future__ import annotations

from collections.abc import Callable

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_sample_weight


def balanced_sample_weights(y):
    return compute_sample_weight(class_weight="balanced", y=y)


def apply_resampling(X, y, strategy: str, random_state: int = 42):
    if strategy == "baseline" or strategy == "class_weight":
        return X, y, None
    if strategy == "smote":
        sampler = SMOTE(random_state=random_state)
        return (*sampler.fit_resample(X, y), None)
    if strategy == "undersampling":
        sampler = RandomUnderSampler(random_state=random_state)
        return (*sampler.fit_resample(X, y), None)
    raise ValueError(f"Unknown imbalance strategy: {strategy}")
