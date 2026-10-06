from pathlib import Path

from backend.app.services.model_registry import ModelRegistry


def test_model_registry_artifact_name_validation(tmp_path: Path):
    registry = ModelRegistry(str(tmp_path))
    assert registry.available_models() == []
    try:
        registry._validate_model_name("..\\secret.joblib")
    except ValueError:
        pass
    else:
        raise AssertionError("Path traversal model filename should be rejected")
