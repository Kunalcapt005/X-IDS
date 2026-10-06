import pandas as pd

from ml.src.split import split_training_data


def test_split_is_stratified_and_reproducible():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series(["Normal"] * 50 + ["DoS"] * 50)
    first = split_training_data(X, y, random_state=42)
    second = split_training_data(X, y, random_state=42)
    assert first[0].equals(second[0])
    assert first[2].equals(second[2])
