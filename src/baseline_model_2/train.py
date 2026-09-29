from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.baseline_model_2.config import (ARTIFACTS_DIR, ID_COL, LABEL_COL,
                                         LABEL_NAME_COL, MIN_ROUTED,
                                         N_BOOTSTRAP, PRECISION_TARGETS,
                                         RANDOM_SEED, TEAM_PRECISION_FLOOR,
                                         TEXT_COL)
from src.baseline_model_2.data_loader import (build_and_save_splits,
                                              load_splits, splits_current)

# TF-IDF + Multinomial Naive Bayes baseline for team classification.
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


def predict_with_confidence(model, x) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(x)
    return np.asarray(model.classes_)[proba.argmax(axis=1)], proba.max(axis=1)


# Gap between stated confidence and observed accuracy, averaged over bins.
def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.sum():
            error += mask.mean() * abs(correct[mask].mean()
                                       - confidence[mask].mean())
    return float(error)


# Coverage and precision at every confidence threshold.
def coverage_curve(confidence: np.ndarray, correct: np.ndarray,
                   min_routed: int = MIN_ROUTED) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0, 1, 501):
        routed = confidence >= threshold
        if routed.sum() < min_routed:
            continue
        rows.append({
            "threshold": round(float(threshold), 3),
            "coverage": float(routed.mean()),
            "precision": float(correct[routed].mean()),
            "n_routed": int(routed.sum()),
        })
    return pd.DataFrame(rows)


# Lowest threshold (most coverage) whose precision on routed stays >= target.
def threshold_for(curve: pd.DataFrame, target: float) -> float | None:
    feasible = curve[curve["precision"] >= target]
    return None if feasible.empty else float(feasible.iloc[0]["threshold"])


def routed_stats(confidence: np.ndarray, correct: np.ndarray,
                 threshold: float) -> dict:
    routed = confidence >= threshold
    return {
        "threshold": threshold,
        "coverage": round(float(routed.mean()), 4),
        "precision": (round(float(correct[routed].mean()), 4)
                      if routed.any() else None),
        "n_routed": int(routed.sum()),
    }


# Percentile bootstrap CI for coverage and precision at a fixed threshold.
def bootstrap_ci(confidence: np.ndarray, correct: np.ndarray,
                 threshold: float, n_boot: int = N_BOOTSTRAP,
                 seed: int = RANDOM_SEED) -> dict:
    rng = np.random.default_rng(seed)
    routed = confidence >= threshold
    n = len(routed)
    cov, prec = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        r, c = routed[idx], correct[idx]
        cov[b] = r.mean()
        prec[b] = c[r].mean() if r.any() else np.nan
    return {
        "coverage_95ci": [round(float(v), 4)
                          for v in np.percentile(cov, [2.5, 97.5])],
        "precision_95ci": [round(float(v), 4)
                           for v in np.nanpercentile(prec, [2.5, 97.5])],
    }


# Per-team guardrail at a threshold: precision of what was routed TO each
# team, and the share of each team's own complaints that were auto-routed.
def per_team_routing(y_true: np.ndarray, y_pred: np.ndarray,
                     confidence: np.ndarray, threshold: float,
                     names: dict[str, str]) -> pd.DataFrame:
    routed = confidence >= threshold
    rows = []
    for team in sorted(set(y_true) | set(y_pred)):
        to_team = routed & (y_pred == team)
        own = y_true == team
        n_to = int(to_team.sum())
        precision = (float((y_true[to_team] == team).mean())
                     if n_to else np.nan)
        rows.append({
            "team_id": team,
            "team_name": names.get(team, ""),
            "n_test": int(own.sum()),
            "n_routed_to": n_to,
            "precision_routed": round(precision, 4),
            "auto_routed_share": round(float(routed[own].mean()), 4)
            if own.any() else np.nan,
            "meets_floor": bool(precision >= TEAM_PRECISION_FLOOR)
            if n_to else None,
        })
    return pd.DataFrame(rows)


def north_star_summary(val_conf, val_correct, test_conf, test_correct,
                       test_true, test_pred, names) -> tuple[dict, pd.DataFrame,
                                                             pd.DataFrame | None]:
    # Thresholds are chosen on validation and applied unchanged to test.
    val_curve = coverage_curve(val_conf, val_correct)
    test_curve = coverage_curve(test_conf, test_correct)
    curves = pd.concat([val_curve.assign(split="validation"),
                        test_curve.assign(split="test")], ignore_index=True)

    operating_points = {}
    for target in PRECISION_TARGETS:
        key = f"at_{int(target * 100)}p"
        threshold = threshold_for(val_curve, target)
        if threshold is None:
            operating_points[key] = None
            continue
        point = routed_stats(test_conf, test_correct, threshold)
        point["validation_coverage"] = routed_stats(
            val_conf, val_correct, threshold)["coverage"]
        point["target_met_on_test"] = (point["precision"] is not None
                                       and point["precision"] >= target)
        operating_points[key] = point

    head = f"at_{int(PRECISION_TARGETS[0] * 100)}p"
    main = operating_points[head]
    teams = None
    if main is not None:
        main.update(bootstrap_ci(test_conf, test_correct, main["threshold"]))
        teams = per_team_routing(test_true, test_pred, test_conf,
                                 main["threshold"], names)

    summary = {
        "metric": (f"autonomous routing rate at >={PRECISION_TARGETS[0]:.0%} "
                   "precision"),
        "protocol": ("temporal split, real team mix in validation and test; "
                     "isotonic calibration and thresholds fit on validation, "
                     "frozen for test"),
        "value": None if main is None else main["coverage"],
        "operating_points": operating_points,
        "ece_test": round(expected_calibration_error(test_conf, test_correct), 4),
        "guardrail": None if teams is None else {
            "team_precision_floor": TEAM_PRECISION_FLOOR,
            "teams_below_floor": teams.loc[teams["meets_floor"] == False,
                                           "team_id"].tolist(),
        },
        "n_validation": int(len(val_correct)),
        "n_test": int(len(test_correct)),
    }
    return summary, curves, teams


def ensure_splits(rebuild: bool, verbose: bool
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rebuild or not splits_current():
        if verbose:
            print("[splits] missing or built with another scheme — "
                  "rebuilding from full CSV\n")
        build_and_save_splits(verbose=verbose)
    return load_splits()


def train_and_evaluate(output_dir: Path, max_features: int,
                       rebuild_splits: bool = False,
                       verbose: bool = True) -> None:
    train, val, test = ensure_splits(rebuild_splits, verbose)

    x_train, y_train = train[TEXT_COL].astype(str), train[LABEL_COL]
    x_val, y_val = val[TEXT_COL].astype(str), val[LABEL_COL]
    x_test, y_test = test[TEXT_COL].astype(str), test[LABEL_COL]
    names = dict(zip(pd.concat([train, val, test])[LABEL_COL],
                     pd.concat([train, val, test])[LABEL_NAME_COL]))

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
    base = build_pipeline(max_features=max_features)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    base.fit(x_train, y_train, clf__sample_weight=sample_weight)

    # Isotonic calibration on validation (real team mix). This also corrects
    # the uniform prior the model learned from the capped training sample.
    if verbose:
        print("[calibrate] isotonic on validation")
    model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    model.fit(x_val, y_val)

    raw_test_pred, raw_test_conf = predict_with_confidence(base, x_test)
    val_pred, val_conf = predict_with_confidence(model, x_val)
    test_pred, test_conf = predict_with_confidence(model, x_test)
    val_metrics = metrics(y_val, val_pred)
    test_metrics = metrics(y_test, test_pred)
    raw_test_metrics = metrics(y_test, raw_test_pred)

    y_val_arr, y_test_arr = y_val.to_numpy(), y_test.to_numpy()
    val_correct = val_pred == y_val_arr
    test_correct = test_pred == y_test_arr
    raw_correct = raw_test_pred == y_test_arr

    labels = sorted(set(y_train) | set(y_test))
    report = classification_report(y_test, test_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    cmat = confusion_matrix(y_test, test_pred, labels=labels)

    north_star, curves, teams = north_star_summary(
        val_conf, val_correct, test_conf, test_correct,
        y_test_arr, test_pred, names)
    north_star["ece_test_uncalibrated"] = round(
        expected_calibration_error(raw_test_conf, raw_correct), 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps({
        "dummy": {"val": dummy_val, "test": dummy_test},
        "model": {"val": val_metrics, "test": test_metrics},
        "model_uncalibrated": {"test": raw_test_metrics},
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    }, indent=2))
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2))
    pd.DataFrame(cmat, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv", index_label="actual\\predicted")
    curves.to_csv(output_dir / "coverage_curve.csv", index=False)
    (output_dir / "north_star.json").write_text(
        json.dumps(north_star, indent=2))
    if teams is not None:
        teams.to_csv(output_dir / "per_team_routing.csv", index=False)

    main = north_star["operating_points"][f"at_{int(PRECISION_TARGETS[0] * 100)}p"]
    preds_out = pd.DataFrame({
        ID_COL: test[ID_COL].astype(str).values,
        "actual_team_id": y_test_arr,
        "actual_team_name": test[LABEL_NAME_COL].values,
        "predicted_team_id": test_pred,
        "confidence": np.round(test_conf, 4),
        "auto_routed": (test_conf >= main["threshold"]) if main
                       else np.zeros(len(test), dtype=bool),
    })
    preds_out.to_csv(output_dir / "predictions_test.csv", index=False)

    if verbose:
        print("\n[results]")
        print(f"  {'metric':<14}{'dummy(val)':>12}{'model(val)':>12}"
              f"{'dummy(test)':>13}{'model(test)':>13}{'uncal(test)':>13}")
        for k in val_metrics:
            print(f"  {k:<14}{dummy_val[k]:>12.3f}{val_metrics[k]:>12.3f}"
                  f"{dummy_test[k]:>13.3f}{test_metrics[k]:>13.3f}"
                  f"{raw_test_metrics[k]:>13.3f}")

        print("\n[per class — test]")
        per_class = (
            pd.DataFrame({k: v for k, v in report.items()
                          if k not in ("accuracy", "macro avg", "weighted avg")})
            .T.sort_values("f1-score").round(3)
        )
        per_class["support"] = per_class["support"].astype(int)
        print(per_class.to_string())

        print("\n[operating points — threshold from validation, applied to test]")
        print(f"  {'target':>7}{'threshold':>11}{'val cov':>10}"
              f"{'test cov':>10}{'test prec':>11}{'routed':>9}")
        for target in PRECISION_TARGETS:
            point = north_star["operating_points"][f"at_{int(target * 100)}p"]
            if point is None:
                print(f"  {target:>7.0%}   not reachable on validation")
                continue
            print(f"  {target:>7.0%}{point['threshold']:>11.3f}"
                  f"{point['validation_coverage']:>10.1%}"
                  f"{point['coverage']:>10.1%}{point['precision']:>11.1%}"
                  f"{point['n_routed']:>9,}")

        if main is None:
            print("\n  [north star] not reachable")
        else:
            lo, hi = main["coverage_95ci"]
            plo, phi = main["precision_95ci"]
            print(f"\n  [north star] {main['coverage']:.1%} "
                  f"(95% CI {lo:.1%}–{hi:.1%}) at precision "
                  f"{main['precision']:.1%} (95% CI {plo:.1%}–{phi:.1%})")
            print(f"\n[guardrail — per team at threshold {main['threshold']:.3f}, "
                  f"floor {TEAM_PRECISION_FLOOR:.0%}]")
            print(teams.drop(columns="team_name").to_string(index=False))
        ece, raw_ece = north_star["ece_test"], north_star["ece_test_uncalibrated"]
        print(f"\n  [calibration] ECE {ece:.3f} (uncalibrated {raw_ece:.3f}) "
              f"({'usable' if ece < 0.05 else 'poor, coverage numbers unreliable'})")

        print(f"\n[write] artifacts -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--max-features", type=int, default=75_000)
    parser.add_argument("--rebuild-splits", action="store_true",
                        help="Rebuild the temporal splits from the full CSV.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args.output_dir, args.max_features,
                       rebuild_splits=args.rebuild_splits,
                       verbose=not args.quiet)
