from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..db.session import database_ping, get_db
from ..models.core import Alert, Flow, ModelVersion, ReplaySession, SessionMetric
from ..schemas.api import (
    AlertSummary,
    PredictionRequest,
    PredictionResponse,
    SessionCreate,
    SessionMetricsResponse,
    SessionResponse,
    SessionSummary,
)
from ..services.inference import InferenceService
from ..services.model_registry import ModelRegistry
from ..services.replay_runner import ReplayRunner
from ..services.stream_hub import stream_hub

router = APIRouter()
registry = ModelRegistry()
inference = InferenceService(registry)
replay_runner = ReplayRunner(registry, inference)
_replay_tasks: dict[int, asyncio.Task] = {}


@router.get("/health")
def health(db: Session = Depends(get_db)):
    models = registry.available_models()
    return {
        "status": "ok",
        "service": "X-IDS",
        "model_artifacts_available": bool(models),
        "model_count": len(models),
        "default_model": registry.default_model() if models else None,
        "database_available": database_ping(),
    }


@router.post("/auth/login")
def login_placeholder():
    return {"message": "Authentication endpoint placeholder. Implement JWT authentication before production use."}


@router.get("/models")
def list_models(db: Session = Depends(get_db)):
    models = registry.available_models()
    model_payload = []
    for model_file in models:
        try:
            info = registry.model_info(model_file)
            model_payload.append(info)
        except Exception as exc:
            model_payload.append({"model_file": model_file, "error": str(exc)})
    if database_ping():
        try:
            registry.sync_model_versions(db)
        except Exception:
            db.rollback()
    return {
        "default_model": registry.default_model() if models else None,
        "models": model_payload,
    }


@router.post("/sessions", response_model=SessionResponse)
async def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    if not database_ping():
        raise HTTPException(status_code=503, detail="Database is unavailable. Start PostgreSQL before creating a replay session.")

    try:
        registry.sync_model_versions(db)
        if payload.model_version_id is not None:
            model_version = db.get(ModelVersion, payload.model_version_id)
            if model_version is None or not model_version.model_artifact_path:
                raise HTTPException(status_code=404, detail="Requested model version was not found")
            model_file = model_version.model_artifact_path
        else:
            model_file = payload.model_file or registry.default_model()
            version = db.query(ModelVersion).filter(ModelVersion.model_artifact_path == model_file).first()
            if version is None:
                registry.sync_model_versions(db)
                version = db.query(ModelVersion).filter(ModelVersion.model_artifact_path == model_file).first()
            if version is None:
                raise HTTPException(status_code=404, detail="Requested model artifact is not registered")
            model_version_id = version.id

        if payload.model_version_id is None:
            model_version_id = version.id

        session = ReplaySession(
            user_id=None,
            model_version_id=model_version_id,
            dataset_source=payload.dataset_source,
            status="created",
            replay_rate=payload.replay_rate,
            max_flows=payload.max_flows,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        task = asyncio.create_task(_start_replay_after_connect_delay(session.id))
        _replay_tasks[session.id] = task
        task.add_done_callback(lambda _: _replay_tasks.pop(session.id, None))

        return SessionResponse(
            id=session.id,
            status=session.status,
            replay_rate=session.replay_rate,
            max_flows=session.max_flows,
            started_at=session.started_at,
            completed_at=session.completed_at,
            model_file=model_file,
        )
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Could not create replay session: {exc}") from exc


async def _start_replay_after_connect_delay(session_id: int) -> None:
    await asyncio.sleep(0.5)
    await replay_runner.run(session_id)


@router.get("/sessions/{session_id}", response_model=SessionSummary)
def get_session(session_id: int, db: Session = Depends(get_db)):
    session = db.get(ReplaySession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    flow_count = db.query(func.count(Flow.id)).filter(Flow.replay_session_id == session_id).scalar() or 0
    alert_count = (
        db.query(func.count(Alert.id))
        .join(Flow, Alert.flow_id == Flow.id)
        .filter(Flow.replay_session_id == session_id)
        .scalar()
        or 0
    )
    model_file = None
    if session.model_version_id:
        model = db.get(ModelVersion, session.model_version_id)
        model_file = model.model_artifact_path if model else None
    return SessionSummary(
        id=session.id,
        status=session.status,
        replay_rate=session.replay_rate,
        max_flows=session.max_flows,
        started_at=session.started_at,
        completed_at=session.completed_at,
        flow_count=flow_count,
        alert_count=alert_count,
        model_file=model_file,
    )


@router.get("/sessions/{session_id}/alerts")
def session_alerts(
    session_id: int,
    severity: str | None = Query(default=None),
    predicted_label: str | None = Query(default=None),
    min_confidence: float | None = Query(default=None, ge=0, le=1),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    if db.get(ReplaySession, session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    query = (
        db.query(Alert)
        .join(Flow, Alert.flow_id == Flow.id)
        .filter(Flow.replay_session_id == session_id)
    )
    if severity:
        query = query.filter(Alert.severity == severity)
    if predicted_label:
        query = query.filter(Alert.predicted_label == predicted_label)
    if min_confidence is not None:
        query = query.filter(Alert.confidence >= min_confidence)
    total = query.count()
    rows = query.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "session_id": session_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "alerts": [AlertSummary.model_validate(row, from_attributes=True) for row in rows],
        "next_offset": offset + limit if offset + limit < total else None,
    }


@router.get("/alerts/{alert_id}")
def alert_detail(alert_id: int, db: Session = Depends(get_db)):
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    flow = db.get(Flow, alert.flow_id)
    explanation = alert.shap_explanation
    try:
        explanation = json.loads(explanation)
    except (TypeError, json.JSONDecodeError):
        pass
    return {
        "id": alert.id,
        "flow_id": alert.flow_id,
        "session_id": flow.replay_session_id if flow else None,
        "predicted_label": alert.predicted_label,
        "confidence": alert.confidence,
        "probability_distribution": alert.probability_distribution,
        "severity": alert.severity,
        "shap_explanation": explanation,
        "created_at": alert.created_at,
        "flow": {
            "timestamp": flow.timestamp if flow else None,
            "source_ip": flow.source_ip if flow else None,
            "dest_ip": flow.dest_ip if flow else None,
            "source_port": flow.source_port if flow else None,
            "dest_port": flow.dest_port if flow else None,
            "protocol": flow.protocol if flow else None,
            "true_label": flow.true_label if flow else None,
            "feature_vector": flow.feature_vector if flow else {},
        },
    }


@router.get("/sessions/{session_id}/metrics", response_model=SessionMetricsResponse)
def session_metrics(session_id: int, db: Session = Depends(get_db)):
    if db.get(ReplaySession, session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    metric = (
        db.query(SessionMetric)
        .filter(SessionMetric.replay_session_id == session_id)
        .order_by(SessionMetric.id.desc())
        .first()
    )
    if metric is None:
        return SessionMetricsResponse(
            session_id=session_id,
            per_class_precision={},
            per_class_recall={},
            per_class_f1={},
            macro_f1=None,
        )
    return SessionMetricsResponse(
        session_id=session_id,
        per_class_precision=metric.per_class_precision,
        per_class_recall=metric.per_class_recall,
        per_class_f1=metric.per_class_f1,
        macro_f1=metric.macro_f1,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    model_file = payload.model_file or registry.default_model()
    try:
        result = inference.predict(model_file, payload.features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    return PredictionResponse(**result)


@router.websocket("/sessions/{session_id}/stream")
async def session_stream(websocket: WebSocket, session_id: int):
    await stream_hub.register(session_id, websocket)
    try:
        await websocket.send_json({"type": "connected", "session_id": session_id})
        while True:
            # Keep the socket alive and allow the client to disconnect cleanly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await stream_hub.unregister(session_id, websocket)
