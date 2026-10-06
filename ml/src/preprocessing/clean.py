from __future__ import annotations

import numpy as np
import pandas as pd


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned
