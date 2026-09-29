"""Train and evaluate a TF-IDF baseline for complaint classification.

Run: python -m src.models.baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.config import DATE, LABEL, ROOT, TEXT
from src.data.prepare import get_splits


def build_model(max_features: int = 75_000, calibrated: bool = True) -> Pipeline:
    """Create a sparse text-classification pipeline.
    """
    classifier = LinearSVC(class_weight="balanced", random_state=42)
    if calibrated:
        classifier = CalibratedClassifierCV(classifier, cv=3)
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def metric_summary(y_true: pd.Series, y_pred: object) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 10
) -> float:
    """Gap between stated confidence and observed accuracy, averaged.
    """
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.sum():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def coverage_curve(
    confidence: np.ndarray, correct: np.ndarray, min_routed: int = 50
) -> pd.DataFrame:
    """Coverage and precision at every confidence threshold.
    """
    rows = []
    for threshold in np.linspace(0, 1, 501):
        routed = confidence >= threshold
        if routed.sum() < min_routed:
            continue
        rows.append(
            {
                "threshold": round(float(threshold), 3),
                "coverage": float(routed.mean()),
                "precision": float(correct[routed].mean()),
                "n_routed": int(routed.sum()),
            }
        )
    return pd.DataFrame(rows)


def routing_rate_at(curve: pd.DataFrame, target: float) -> dict | None:
    """Most coverage obtainable while precision on routed stays >= target."""
    feasible = curve[curve["precision"] >= target]
    return None if feasible.empty else feasible.iloc[0].to_dict()


def train_and_evaluate(
    output_dir: Path, max_features: int, calibrated: bool = True
) -> None:
    train, test = get_splits(verbose=True)
    x_train, y_train = train[TEXT], train[LABEL]
    x_test, y_test = test[TEXT], test[LABEL]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train.to_frame(), y_train)
    dummy_metrics = metric_summary(y_test, dummy.predict(x_test.to_frame()))

    print("\n[train] fitting word (1,2)-gram TF-IDF + LinearSVC"
          f"{' (calibrated)' if calibrated else ''}")
    model = build_model(max_features=max_features, calibrated=calibrated)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    model_metrics = metric_summary(y_test, predictions)

    labels = sorted(set(y_train) | set(y_test))
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({"dummy": dummy_metrics, "model": model_metrics}, indent=2)
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2)
    )
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv", index_label="actual\\predicted"
    )

    prediction_columns = [DATE, "Complaint ID"]
    prediction_frame = test[prediction_columns].copy()
    prediction_frame["actual"] = y_test.astype(str)
    prediction_frame["predicted"] = predictions
    
    if "dup_group_size" in test.columns:
        prediction_frame["dup_group_size"] = test["dup_group_size"].values

    print("\n[results]")
    print(f"  {'metric':<14} {'dummy':>9} {'model':>9}")
    for metric in model_metrics:
        print(
            f"  {metric:<14} {dummy_metrics[metric]:>9.3f} "
            f"{model_metrics[metric]:>9.3f}"
        )

    
    print("\n[per class]")
    per_class = (
        pd.DataFrame(
            {k: v for k, v in report.items()
             if k not in ("accuracy", "macro avg", "weighted avg")}
        )
        .T.sort_values("f1-score")
        .round(3)
    )
    per_class["support"] = per_class["support"].astype(int)
    print(per_class.to_string())
    per_class.to_csv(output_dir / "per_class.csv", index_label="class")

    # ---- north star: autonomous routing rate at a precision floor ----
    north_star = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        classes = np.asarray(model.classes_)
        confidence = proba.max(axis=1)
        correct = classes[proba.argmax(axis=1)] == y_test.astype(str).values

        curve = coverage_curve(confidence, correct)
        ece = expected_calibration_error(confidence, correct)
        curve.to_csv(output_dir / "coverage_curve.csv", index=False)

        prediction_frame["confidence"] = confidence

        print("\n[coverage vs precision]")
        print(f"  {'threshold':>10}{'coverage':>11}{'precision':>11}{'routed':>9}")
        for threshold in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            rows = curve[curve["threshold"] >= threshold]
            if rows.empty:
                continue
            row = rows.iloc[0]
            print(
                f"  {row['threshold']:>10.2f}{row['coverage']:>10.1%}"
                f"{row['precision']:>11.1%}{int(row['n_routed']):>9,}"
            )

        operating_points = {}
        print()
        for target in (0.95, 0.90, 0.85):
            point = routing_rate_at(curve, target)
            key = f"coverage_at_{int(target * 100)}p"
            if point is None:
                print(f"  {target:.0%} precision -> not reachable")
                operating_points[key] = None
            else:
                print(
                    f"  {target:.0%} precision -> coverage {point['coverage']:.1%} "
                    f"at threshold {point['threshold']:.3f} "
                    f"({int(point['n_routed']):,} routed)"
                )
                operating_points[key] = round(point["coverage"], 4)

        north_star = {
            "metric": "autonomous routing rate at >=95% precision",
            "value": operating_points.get("coverage_at_95p"),
            "ece": round(ece, 4),
            "operating_points": operating_points,
            "n_test": int(len(correct)),
        }
        (output_dir / "north_star.json").write_text(json.dumps(north_star, indent=2))

        if north_star["value"] is not None:
            print(f"\n  [north star] {north_star['value']:.1%}")
        print(
            f"  [calibration] ECE {ece:.3f} "
            f"({'usable' if ece < 0.05 else 'poor, coverage numbers unreliable'})"
        )
    else:
        print("\n[skip] north star needs predict_proba — rerun without "
              "--uncalibrated")

    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    print(f"\n[write] artifacts saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "metrics" / "baseline",
        help="Directory for the model and evaluation artifacts.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=75_000,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--uncalibrated",
        action="store_true",
        help="Skip probability calibration. Faster and slightly higher "
             "macro-F1, but no confidence scores and no north-star metric.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(
        args.output_dir, args.max_features, calibrated=not args.uncalibrated
    )