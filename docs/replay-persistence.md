# X-IDS Replay + Persistence v1

This milestone wires the held-out UNSW-NB15 testing-set replay into PostgreSQL and WebSocket delivery.

## Local setup

From the repository root:

```text
.venv\Scripts\activate
python -m pip install -r backend\requirements.txt
```

Start PostgreSQL with Docker Compose:

```text
docker compose up -d postgres
```

Start FastAPI locally:

```text
python -m uvicorn backend.app.main:app --reload --port 8000
```

Check:

```text
GET http://127.0.0.1:8000/health
```

`database_available` should be `true`.

## Replay session

Create a session using `POST /sessions`. By default the development replay uses the first 250 rows so the end-to-end path can be tested quickly. Set `max_flows` to a larger value, up to the available test-set rows, for longer demonstrations.

Example body:

```json
{
  "model_file": "selected_model.joblib",
  "dataset_source": "UNSW_NB15_testing-set.csv",
  "replay_rate": 10,
  "max_flows": 250
}
```

The server creates the session, then starts the replay after a short delay so a frontend can connect to `/sessions/{id}/stream`.

Each streamed `alert` event contains the prediction, confidence, class probabilities, severity, explanation, and top SHAP features. Each flow and alert is persisted in PostgreSQL.

## Dataset boundary

The pre-partitioned UNSW-NB15 training/testing CSVs used by X-IDS do not contain source/destination IP addresses or source/destination ports. The database therefore stores those operational fields as nullable rather than fabricating them. The `proto` field is preserved as the flow protocol.

## Production boundary

The WebSocket hub is intentionally in-memory for this academic v1 and therefore assumes a single backend process. A multi-instance deployment should replace it with a shared pub/sub mechanism. Database schema evolution should move from `create_all` to Alembic migrations before production use.
