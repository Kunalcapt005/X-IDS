from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.common import load_config
from src.preprocessing.load_data import load_unsw_nb15

config = load_config()
train, test = load_unsw_nb15(
    ML_ROOT / config["paths"]["train_csv"],
    ML_ROOT / config["paths"]["test_csv"],
)

print("TRAIN SHAPE:", train.shape)
print("TEST SHAPE:", test.shape)
print("\nTRAIN CLASS DISTRIBUTION:")
print(train["attack_cat"].value_counts(dropna=False))
print("\nTEST CLASS DISTRIBUTION:")
print(test["attack_cat"].value_counts(dropna=False))
print("\nDTYPES:")
print(train.dtypes)
