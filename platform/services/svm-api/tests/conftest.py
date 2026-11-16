import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]  # platform/services/svm-api/tests → repo root
MODEL_PATH = REPO_ROOT / "docs" / "metrics" / "improved_model_1" / "model.joblib"


@pytest.fixture(scope="session")
def model_path() -> Path:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model artifact missing: {MODEL_PATH}")
    os.environ.setdefault("MODEL_PATH", str(MODEL_PATH))
    return MODEL_PATH
