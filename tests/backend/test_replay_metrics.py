from backend.app.services.replay_runner import ReplayRunner


def test_replay_metrics_macro_f1():
    metrics = ReplayRunner._metrics(
        ["Normal", "Worms", "Worms", "DoS"],
        ["Normal", "Worms", "DoS", "DoS"],
    )
    assert metrics["per_class_recall"]["Worms"] == 0.5
    assert 0 <= metrics["macro_f1"] <= 1
