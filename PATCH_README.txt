X-IDS Replay + PostgreSQL + WebSocket milestone

Replace the corresponding files under your existing E:\X-IDS project with the files in this patch.
The latest inference.py is included as well; it is compatible with the feature_metadata.json fix already applied.

Main new capability:
- PostgreSQL persistence for model versions, replay sessions, flows, alerts, and session metrics.
- Test-set replay using UNSW_NB15_testing-set.csv.
- WebSocket alert streaming.
- Per-session precision, recall, F1, and macro-F1.
- Alert drill-down data with stored SHAP explanation and flow feature vector.
- Development replay defaults to 250 flows for a quick end-to-end test.

After copying the files:

1) Make sure these files exist:
   ml\\data\\raw\\UNSW_NB15_training-set.csv
   ml\\data\\raw\\UNSW_NB15_testing-set.csv
   ml\\artifacts\\selected_model.joblib
   ml\\artifacts\\feature_metadata.json

2) Install backend dependencies:
   python -m pip install -r backend\\requirements.txt

3) Start PostgreSQL:
   docker compose up -d postgres

4) Start FastAPI locally:
   python -m uvicorn backend.app.main:app --reload --port 8000

5) Open:
   http://127.0.0.1:8000/docs

6) Check GET /health. database_available should be true.

7) Create a replay session with POST /sessions. Example:
   {
     "model_file": "selected_model.joblib",
     "dataset_source": "UNSW_NB15_testing-set.csv",
     "replay_rate": 10,
     "max_flows": 250
   }

8) Copy the returned session id and connect a WebSocket client to:
   ws://127.0.0.1:8000/sessions/{session_id}/stream

The replay emits session_started, alert, and session_completed events.

Important dataset boundary:
The supplied pre-partitioned UNSW-NB15 training/testing CSVs do not contain source/destination IP addresses or source/destination port fields. X-IDS leaves those database fields NULL rather than inventing values.
