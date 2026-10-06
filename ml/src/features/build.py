from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeatureSchema:
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]


def infer_feature_schema(
    frame: pd.DataFrame,
    target_col: str,
    drop_columns: list[str],
    forced_categoricals: list[str],
) -> FeatureSchema:
    excluded = set(drop_columns) | {target_col}
    feature_columns = [column for column in frame.columns if column not in excluded]

    categorical = [
        column
        for column in feature_columns
        if column in forced_categoricals
        or frame[column].dtype == "object"
        or str(frame[column].dtype).startswith("string")
    ]
    numeric = [column for column in feature_columns if column not in categorical]

    return FeatureSchema(feature_columns, categorical, numeric)


def build_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, schema.numeric_columns),
            ("categorical", categorical_pipeline, schema.categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
