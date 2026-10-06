# X-IDS Backend Inference

The `/predict` endpoint consumes the raw flow-level feature schema persisted in the selected model bundle. The backend applies the exact fitted preprocessor used during training, calls the stored classifier, and computes a SHAP explanation from the same transformed feature vector.

## Start locally

From the repository root:

```cmd
.venv\Scripts\activate
python -m uvicorn backend.app.main:app --reload --port 8000
```

The backend expects the trained artifacts under `ml/artifacts`.

## Inspect the API

Open:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET http://127.0.0.1:8000/health
```

Model list:

```text
GET http://127.0.0.1:8000/models
```

## Build a real test request

```cmd
python ml\scripts\make_prediction_request.py --row 0
```

Then use the generated JSON payload with `POST /predict`.

The response contains the predicted category, confidence, full probability distribution, severity, and the top ten transformed-feature SHAP attributions.
