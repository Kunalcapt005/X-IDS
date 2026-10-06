from __future__ import annotations

import numpy as np
import pandas as pd


def prune_redundant_features(frame: pd.DataFrame, protected_columns: list[str], correlation_threshold: float = 0.98, missing_threshold: float = 0.98):
    result = frame.copy()
    drop_columns: list[str] = []

    for column in result.columns:
        if column in protected_columns:
            continue
        if result[column].isna().mean() >= missing_threshold:
            drop_columns.append(column)
        elif result[column].nunique(dropna=False) <= 1:
            drop_columns.append(column)

    result = result.drop(columns=drop_columns, errors="ignore")

    numeric = result.select_dtypes(include=[np.number])
    if len(numeric.columns) > 1:
        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        correlated = [column for column in upper.columns if (upper[column] >= correlation_threshold).any()]
        correlated = [column for column in correlated if column not in protected_columns]
        result = result.drop(columns=correlated, errors="ignore")
        drop_columns.extend(correlated)

    return result, drop_columns
