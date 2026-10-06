from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.common import load_config
from src.preprocessing.clean import clean_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an X-IDS /predict request from one UNSW-NB15 test row")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--output", default="reports/sample_prediction_request.json")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    csv_path = ML_ROOT / config["paths"]["test_csv"]
    frame = clean_frame(pd.read_csv(csv_path))

    if not 0 <= args.row < len(frame):
        raise SystemExit(f"row must be between 0 and {len(frame) - 1}")

    drop = set(config["columns"]["drop"])
    raw_row = {str(k): v for k, v in frame.iloc[args.row].to_dict().items() if k not in drop}
    row = {}
    for key, value in raw_row.items():
        if hasattr(value, "item"):
            value = value.item()
        if pd.isna(value):
            value = None
        row[key] = value
    payload = {"model_file": "selected_model.joblib", "features": row}

    output_path = ML_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote: {output_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
