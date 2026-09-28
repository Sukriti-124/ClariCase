"""TF-IDF + Multinomial Naive Bayes baseline for team classification.

Target: `team_id` (11 support teams from data/processed/team_mapping.csv).
Input : `complaint_text`.

Reads pre-built train/validation/test splits from data/processed/splits/.
If they are missing, builds them first via data_loader.build_and_save_splits().

Artifacts written to src/baseline_model_1/artifacts/:
  - model.joblib               fitted sklearn Pipeline
  - metrics.json               accuracy / macro-F1 / weighted-F1 for val + test
  - classification_report.json per-class precision/recall/F1 on test
  - confusion_matrix.csv       test-set confusion matrix
  - predictions_test.csv       row-level test predictions with confidence

Run: python -m src.baseline_model_1.train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.baseline_model_1.config import (ARTIFACTS_DIR, ID_COL, LABEL_COL,
                                         LABEL_NAME_COL, RANDOM_SEED,
                                         SPLITS_DIR, TEXT_COL)
from src.baseline_model_1.data_loader import (build_and_save_splits,
                                              load_splits)


def build_pipeline(max_features: int = 75_000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", MultinomialNB(alpha=0.1, fit_prior=False)),
    ])


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def ensure_splits(verbose: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (SPLITS_DIR / "train.csv").exists():
        if verbose:
            print("[splits] not found — building from full CSV\n")
        build_and_save_splits(verbose=verbose)
    return load_splits()


def train_and_evaluate(output_dir: Path, max_features: int,
                       verbose: bool = True) -> None:
    train, val, test = ensure_splits(verbose)

    x_train, y_train = train[TEXT_COL].astype(str), train[LABEL_COL]
    x_val, y_val = val[TEXT_COL].astype(str), val[LABEL_COL]
    x_test, y_test = test[TEXT_COL].astype(str), test[LABEL_COL]

    if verbose:
        print(f"\n[data] train={len(train):,}  val={len(val):,}  "
              f"test={len(test):,}  classes={y_train.nunique()}")

    # Baseline reference
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train.to_frame(), y_train)
    dummy_val = metrics(y_val, dummy.predict(x_val.to_frame()))
    dummy_test = metrics(y_test, dummy.predict(x_test.to_frame()))

    if verbose:
        print("\n[train] fitting TF-IDF(1,2) + MultinomialNB (balanced sample weights)")
    model = build_pipeline(max_features=max_features)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(x_train, y_train, clf__sample_weight=sample_weight)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_metrics = metrics(y_val, val_pred)
    test_metrics = metrics(y_test, test_pred)

    labels = sorted(set(y_train) | set(y_test))
    report = classification_report(y_test, test_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    cmat = confusion_matrix(y_test, test_pred, labels=labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps({
        "dummy": {"val": dummy_val, "test": dummy_test},
        "model": {"val": val_metrics, "test": test_metrics},
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    }, indent=2))
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2))
    pd.DataFrame(cmat, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv", index_label="actual\\predicted")

    proba = model.predict_proba(x_test)
    confidence = proba.max(axis=1)
    preds_out = pd.DataFrame({
        ID_COL: test[ID_COL].astype(str).values,
        "actual_team_id": y_test.values,
        "actual_team_name": test[LABEL_NAME_COL].values,
        "predicted_team_id": test_pred,
        "confidence": np.round(confidence, 4),
    })
    preds_out.to_csv(output_dir / "predictions_test.csv", index=False)

    if verbose:
        print("\n[results]")
        print(f"  {'metric':<14}{'dummy(val)':>12}{'model(val)':>12}"
              f"{'dummy(test)':>13}{'model(test)':>13}")
        for k in val_metrics:
            print(f"  {k:<14}{dummy_val[k]:>12.3f}{val_metrics[k]:>12.3f}"
                  f"{dummy_test[k]:>13.3f}{test_metrics[k]:>13.3f}")

        print("\n[per class — test]")
        per_class = (
            pd.DataFrame({k: v for k, v in report.items()
                          if k not in ("accuracy", "macro avg", "weighted avg")})
            .T.sort_values("f1-score").round(3)
        )
        per_class["support"] = per_class["support"].astype(int)
        print(per_class.to_string())

        print(f"\n[write] artifacts -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--max-features", type=int, default=75_000)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args.output_dir, args.max_features,
                       verbose=not args.quiet)
