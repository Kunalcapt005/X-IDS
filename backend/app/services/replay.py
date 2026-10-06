from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pandas as pd


async def replay_dataframe(frame: pd.DataFrame, rate: int) -> AsyncIterator[dict[str, Any]]:
    """Yield dataframe rows at approximately the requested rate."""
    delay = 1 / max(rate, 1)
    for record in frame.to_dict(orient="records"):
        yield record
        await asyncio.sleep(delay)


def load_test_rows(csv_path: Path, feature_columns: list[str], max_flows: int | None = None) -> pd.DataFrame:
    """Load only the model features and ground-truth label from the test CSV.

    The official pre-partitioned UNSW-NB15 flow CSVs do not expose source/dest
    IP or source/dest port columns, so the replay layer deliberately leaves
    those operational fields nullable rather than fabricating values.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Replay dataset not found: {csv_path}")

    usecols = list(dict.fromkeys(feature_columns + ["attack_cat"]))
    frame = pd.read_csv(csv_path, usecols=usecols, nrows=max_flows)
    frame = frame.reset_index(drop=True)
    return frame
