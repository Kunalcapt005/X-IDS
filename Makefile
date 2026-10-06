.PHONY: test ml-preprocess ml-train backend frontend

test:
	pytest

ml-preprocess:
	cd ml && python scripts/run_pipeline.py --stage preprocess

ml-train:
	cd ml && python scripts/run_pipeline.py --stage train

backend:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev
