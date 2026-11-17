"""Team classifier on top of fixed embeddings."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

from src.improved_model_2.config import MAX_ITER, MODEL_NAME


def fit_classifier(X_train: np.ndarray, y_train, C: float = 2.0,
                   max_iter: int = MAX_ITER) -> LogisticRegression:
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter)
    return clf.fit(X_train, np.asarray(y_train))


def calibrate(clf: LogisticRegression, X_val: np.ndarray, y_val
              ) -> CalibratedClassifierCV:
    """Isotonic calibration on validation, same protocol as improved_model_1.
    Training data is capped per issue, so raw probabilities reflect the wrong
    team mix; validation has the real mix."""
    cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    return cal.fit(X_val, np.asarray(y_val))


def make_bundle(calibrated, encoder_name: str, max_seq_length: int,
                global_threshold: float | None,
                model_name: str = MODEL_NAME) -> dict:
    return {
        "model_name": model_name,
        "classifier": calibrated,
        "team_classes": [str(c) for c in calibrated.classes_],
        "encoder_name": encoder_name,
        "max_seq_length": max_seq_length,
        "global_threshold": global_threshold,
    }
