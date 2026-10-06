from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..db.session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    algorithm: Mapped[str] = mapped_column(String(80))
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    feature_list: Mapped[list] = mapped_column(JSON, default=list)
    scaler_artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model_artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    evaluation_metrics: Mapped[dict] = mapped_column(JSON, default=dict)


class ReplaySession(Base):
    __tablename__ = "replay_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    dataset_source: Mapped[str] = mapped_column(String(255))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="created", index=True)
    replay_rate: Mapped[int] = mapped_column(Integer)
    max_flows: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Flow(Base):
    __tablename__ = "flows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    replay_session_id: Mapped[int] = mapped_column(ForeignKey("replay_sessions.id"), index=True)
    timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dest_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dest_port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    protocol: Mapped[str | None] = mapped_column(String(32), nullable=True)
    feature_vector: Mapped[dict] = mapped_column(JSON, default=dict)
    true_label: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    flow_id: Mapped[int] = mapped_column(ForeignKey("flows.id"), index=True)
    predicted_label: Mapped[str] = mapped_column(String(80), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    probability_distribution: Mapped[dict] = mapped_column(JSON, default=dict)
    severity: Mapped[str] = mapped_column(String(30), index=True)
    shap_explanation: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class SessionMetric(Base):
    __tablename__ = "session_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    replay_session_id: Mapped[int] = mapped_column(ForeignKey("replay_sessions.id"), index=True)
    per_class_precision: Mapped[dict] = mapped_column(JSON, default=dict)
    per_class_recall: Mapped[dict] = mapped_column(JSON, default=dict)
    per_class_f1: Mapped[dict] = mapped_column(JSON, default=dict)
    macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
