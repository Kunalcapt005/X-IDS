from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    model_version_id: int | None = None
    model_file: str | None = None
    dataset_source: str = "UNSW_NB15_testing-set.csv"
    replay_rate: int = Field(default=50, ge=1, le=10000)
    max_flows: int | None = Field(default=250, ge=1, le=82332)


class SessionResponse(BaseModel):
    id: int
    status: str
    replay_rate: int
    max_flows: int | None
    started_at: datetime
    completed_at: datetime | None = None
    model_file: str | None = None


class PredictionRequest(BaseModel):
    model_file: str | None = None
    features: dict[str, Any] = Field(min_length=1)


class TopFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str
    magnitude: float


class PredictionResponse(BaseModel):
    model_file: str
    predicted_label: str
    confidence: float = Field(ge=0, le=1)
    probability_distribution: dict[str, float]
    severity: str
    explanation: str
    top_features: list[TopFeature]


class SessionSummary(BaseModel):
    id: int
    status: str
    replay_rate: int
    max_flows: int | None
    started_at: datetime
    completed_at: datetime | None
    flow_count: int
    alert_count: int
    model_file: str | None


class AlertSummary(BaseModel):
    id: int
    flow_id: int
    predicted_label: str
    confidence: float
    severity: str
    created_at: datetime


class SessionMetricsResponse(BaseModel):
    session_id: int
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    macro_f1: float | None
