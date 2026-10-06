from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sqlalchemy.orm import Session

from ..core.config import settings
from ..db.session import SessionLocal
from ..models.core import Alert, Flow, ReplaySession, SessionMetric
from .inference import InferenceService
from .model_registry import ModelRegistry
from .replay import load_test_rows, replay_dataframe
from .stream_hub import stream_hub


logger = logging.getLogger(__name__)


class ReplayRunner:
    CLASS_NAMES = [
        "Analysis",
        "Backdoors",
        "DoS",
        "Exploits",
        "Fuzzers",
        "Generic",
        "Normal",
        "Reconnaissance",
        "Shellcode",
        "Worms",
    ]

    def __init__(self, registry: ModelRegistry | None = None, inference: InferenceService | None = None):
        self.registry = registry or ModelRegistry()
        self.inference = inference or InferenceService(self.registry)

    async def run(self, session_id: int) -> None:
        db = SessionLocal()
        try:
            session = db.get(ReplaySession, session_id)
            if session is None:
                return

            session.status = "running"
            session.started_at = datetime.now(timezone.utc)
            db.commit()
            await stream_hub.publish(session.id, {"type": "session_started", "session_id": session.id})

            model_file = self.registry.model_file_from_version(db, session.model_version_id) if session.model_version_id else self.registry.default_model()
            logger.info("Replay %s: model=%s", session.id, model_file)
            feature_columns = self.registry.feature_columns()
            logger.info("Replay %s: feature_columns=%d", session.id, len(feature_columns))
            csv_path = self._resolve_dataset(session.dataset_source)
            logger.info("Replay %s: dataset=%s", session.id, csv_path)
            frame = load_test_rows(csv_path, feature_columns, session.max_flows)
            logger.info("Replay %s: loaded %d rows", session.id, len(frame))

            y_true: list[str] = []
            y_pred: list[str] = []
            pending_events: list[dict[str, Any]] = []
            batch_size = 25

            async for record in replay_dataframe(frame, session.replay_rate):
                true_label = str(record.pop("attack_cat"))
                feature_payload = self._jsonable(record)

                result = await asyncio.to_thread(self.inference.predict, model_file, feature_payload)
                flow = Flow(
                    replay_session_id=session.id,
                    protocol=self._as_optional_string(feature_payload.get("proto")),
                    feature_vector=feature_payload,
                    true_label=true_label,
                )
                alert = Alert(
                    flow_id=0,
                    predicted_label=result["predicted_label"],
                    confidence=float(result["confidence"]),
                    probability_distribution=result["probability_distribution"],
                    severity=result["severity"],
                    shap_explanation=json.dumps({
                        "text": result["explanation"],
                        "top_features": result["top_features"],
                    }),
                )
                db.add(flow)
                pending_events.append((flow, alert, result, true_label))
                y_true.append(true_label)
                y_pred.append(result["predicted_label"])

                if len(pending_events) >= batch_size:
                    await self._flush_batch(db, session.id, pending_events)
                    pending_events.clear()

            if pending_events:
                await self._flush_batch(db, session.id, pending_events)
                pending_events.clear()

            metrics = self._metrics(y_true, y_pred)
            db.add(SessionMetric(replay_session_id=session.id, **metrics))
            session.status = "completed"
            session.completed_at = datetime.now(timezone.utc)
            db.commit()
            await stream_hub.publish(session.id, {
                "type": "session_completed",
                "session_id": session.id,
                "metrics": {
                    "macro_f1": metrics["macro_f1"],
                    "flow_count": len(y_true),
                },
            })
        except asyncio.CancelledError:
            db.rollback()
            session = db.get(ReplaySession, session_id)
            if session:
                session.status = "cancelled"
                session.completed_at = datetime.now(timezone.utc)
                db.commit()
            await stream_hub.publish(session_id, {"type": "session_cancelled", "session_id": session_id})
            raise
        except Exception as exc:
            logger.exception("Replay %s failed", session_id)
            db.rollback()
            session = db.get(ReplaySession, session_id)
            if session:
                session.status = "failed"
                session.completed_at = datetime.now(timezone.utc)
                db.commit()
            await stream_hub.publish(session_id, {
                "type": "session_failed",
                "session_id": session_id,
                "error": str(exc),
            })
        finally:
            db.close()

    async def _flush_batch(self, db: Session, session_id: int, pending: list[tuple[Flow, Alert, dict[str, Any], str]]) -> None:
        # Flush flows first so their primary keys exist, then attach alerts.
        db.flush()
        for flow, alert, _result, _true_label in pending:
            alert.flow_id = flow.id
            db.add(alert)
        db.flush()
        db.commit()
        for flow, alert, result, true_label in pending:
            await stream_hub.publish(session_id, {
                "type": "alert",
                "session_id": session_id,
                "flow_id": flow.id,
                "alert_id": alert.id,
                "true_label": true_label,
                "predicted_label": result["predicted_label"],
                "confidence": result["confidence"],
                "severity": result["severity"],
                "probability_distribution": result["probability_distribution"],
                "explanation": result["explanation"],
                "top_features": result["top_features"],
            })

    def _resolve_dataset(self, dataset_source: str) -> Path:
        filename = Path(dataset_source).name
        if filename != "UNSW_NB15_testing-set.csv":
            raise ValueError("Only UNSW_NB15_testing-set.csv is enabled for replay in v1")

        configured_artifacts = Path(settings.ml_artifact_dir).resolve()
        ml_root = configured_artifacts.parent
        return ml_root / "data" / "raw" / filename

    @classmethod
    def _metrics(cls, y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
        # Keep all 10 known classes visible in the API response. For a replay
        # session, macro-F1 is averaged only over classes that actually occur
        # in the ground-truth sample, so a small diagnostic session containing
        # only Normal traffic is not diluted by unrelated absent attack classes.
        labels = cls.CLASS_NAMES
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
        active = support > 0
        macro_f1 = float(f1[active].mean()) if active.any() else None
        return {
            "per_class_precision": {label: float(value) for label, value in zip(labels, precision)},
            "per_class_recall": {label: float(value) for label, value in zip(labels, recall)},
            "per_class_f1": {label: float(value) for label, value in zip(labels, f1)},
            "macro_f1": macro_f1,
        }

    @staticmethod
    def _jsonable(record: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in record.items():
            if pd.isna(value):
                out[key] = None
            elif hasattr(value, "item"):
                out[key] = value.item()
            else:
                out[key] = value
        return out

    @staticmethod
    def _as_optional_string(value: Any) -> str | None:
        return None if value is None else str(value)
