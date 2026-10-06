# X-IDS

AI-Based Network Intrusion Detection System using UNSW-NB15 flow data.

## Initial scope

The v1 implementation is organized around the documented architecture:

- Offline ML training on the official UNSW-NB15 train/test split
- Flow-level multi-class classification
- Imbalance experiments: baseline, class weighting, SMOTE, and undersampling
- Logistic Regression, Random Forest, and XGBoost
- Macro-F1 as the primary selection metric
- SHAP-based per-prediction explanations
- FastAPI backend for inference and replay sessions
- PostgreSQL persistence
- Next.js frontend shell for the SOC dashboard

The project intentionally does not claim live network detection, zero-day detection, payload inspection, or automated blocking in v1.

## Repository layout

```text
X-IDS/
├── backend/                 # FastAPI application
├── frontend/                # Next.js dashboard
├── ml/                      # Offline ML pipeline
│   ├── data/raw/            # UNSW-NB15 CSVs here
│   ├── data/processed/      # Generated datasets
│   ├── artifacts/           # Models/scalers/metadata
│   ├── reports/             # Generated evaluation reports
│   ├── scripts/             # CLI entry points
│   └── src/                 # Reusable ML modules
├── tests/                   # Unit/integration test skeletons
├── docs/                    # Project notes
├── docker-compose.yml
└── .env.example
```

## Dataset placement

Download the official UNSW-NB15 training and testing CSV files and place them as:

```text
ml/data/raw/UNSW_NB15_training-set.csv
ml/data/raw/UNSW_NB15_testing-set.csv
```

### 1. Create the Python environment

```bash
cd ml
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Validate and prepare the dataset

```bash
python scripts/run_pipeline.py --stage preprocess
```

### 3. Train the model comparison

```bash
python scripts/run_pipeline.py --stage train
```

### 4. Evaluate saved runs

```bash
python scripts/run_pipeline.py --stage evaluate
```

Artifacts are written under `ml/artifacts/` and reports under `ml/reports/`.

## Backend quick start

```bash
cd backend
python -m venv .venv
# activate environment
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health endpoint:

```text
GET http://localhost:8000/health
```

## Frontend quick start

```bash
cd frontend
npm install
npm run dev
```

The initial frontend is a minimal dashboard shell. UI modules will be expanded after the API/data contract is stable.

## Docker

The root `docker-compose.yml` currently starts PostgreSQL and the backend. The frontend can be added to the same compose workflow once the UI dependencies are finalized.
