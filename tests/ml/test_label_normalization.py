import pandas as pd

from ml.src.common import normalize_attack_labels


def test_normalize_normal_label_from_binary_label():
    frame = pd.DataFrame({"attack_cat": [None, "DoS"], "label": [0, 1]})
    result = normalize_attack_labels(frame, "attack_cat", "label")
    assert result["attack_cat"].tolist() == ["Normal", "DoS"]
