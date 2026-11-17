"""Train and evaluate improved_model_2 (sentence embeddings + logistic regression).

Data selection is identical to improved_model_1: the persisted `splits_v2`
files (train_cap_issue for training; val and test with the real team mix,
temporal and group-aware). The test complaint IDs are checked against
improved_model_1's committed predictions before anything is trained.

Usage:
    python -m src.improved_model_2.train                 # full run
    python -m src.improved_model_2.train --smoke 2000    # quick end-to-end check

Embedding is the slow step and runs once: vectors are cached per encoder under
data/cache/improved_model_2/. Use a GPU if you have one (--device cuda).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.ensemble.data import write_predictions
from src.improved_model_2 import config as M
from src.improved_model_2.encoder import Encoder, load_encoder
from src.improved_model_2.model import calibrate, fit_classifier, make_bundle
from src.utils import config as C
from src.utils.data import load_split
from src.utils.routing import north_star_summary


def _slug(name: str) -> str:
    return name.replace("/", "__")


def assert_same_test_ids(test: pd.DataFrame) -> None:
    """The comparison with improved_model_1 only means something on the same
    complaints. Skipped (with a note) when that predictions file is absent."""
    ref = C.METRICS_ROOT / "improved_model_1" / "predictions_test.csv"
    if not ref.exists():
        print(f"[data] {ref} not found; skipping test-ID parity check")
        return
    theirs = set(pd.read_csv(ref, usecols=[C.ID_COL],
                             dtype={C.ID_COL: "string"})[C.ID_COL].astype(str))
    ours = set(test[C.ID_COL].astype(str))
    if ours != theirs:
        raise AssertionError(
            f"test IDs differ from improved_model_1: only ours="
            f"{len(ours - theirs):,}, only theirs={len(theirs - ours):,}")
    print(f"[data] test IDs match improved_model_1 ({len(ours):,} rows)")


def embed_split(name: str, df: pd.DataFrame, encoder: Encoder,
                cache_dir: Path | None, batch_size: int,
                verbose: bool = True) -> np.ndarray:
    """Embed one split, reusing the cache when it holds exactly these IDs."""
    ids = np.asarray(df[C.ID_COL].astype(str).tolist(), dtype=str)
    if cache_dir is not None:
        vec_path = cache_dir / f"{name}.npy"
        ids_path = cache_dir / f"{name}_ids.npy"
        if vec_path.exists() and ids_path.exists():
            cached_ids = np.load(ids_path, allow_pickle=False)
            if len(cached_ids) == len(ids) and (cached_ids == ids).all():
                if verbose:
                    print(f"[embed] {name}: cached ({len(ids):,} rows)")
                return np.load(vec_path)
    t0 = time.time()
    X = encoder.encode(df[C.TEXT_COL].astype(str).tolist(),
                       batch_size=batch_size, show_progress=verbose)
    if verbose:
        print(f"[embed] {name}: {len(ids):,} rows in {time.time() - t0:.0f}s")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / f"{name}.npy", X)
        np.save(cache_dir / f"{name}_ids.npy", ids)
    return X


def tune_C(X_train, y_train, X_val, y_val, grid=M.C_GRID,
           verbose: bool = True) -> tuple[float, list[dict]]:
    rows = []
    for c in grid:
        clf = fit_classifier(X_train, y_train, C=c)
        f1 = float(f1_score(y_val, clf.predict(X_val), average="macro",
                            zero_division=0))
        rows.append({"C": c, "val_team_macro_f1": f1})
        if verbose:
            print(f"  [tune] C={c}: val macro-F1 = {f1:.4f}")
    return max(rows, key=lambda r: r["val_team_macro_f1"])["C"], rows


def _team_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float((np.asarray(y_pred) == np.asarray(y_true)).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
    }


def _pred_frame(df: pd.DataFrame, pred, conf, proba=None,
                classes=None) -> pd.DataFrame:
    out = pd.DataFrame({
        C.ID_COL: df[C.ID_COL].astype(str).to_numpy(),
        "actual_team_id": df[C.TEAM_COL].to_numpy(),
        "predicted_team_id": pred,
        "confidence": conf,
        "n_words": df[C.TEXT_COL].astype(str).str.split().str.len()
                   .fillna(0).astype(int).to_numpy(),
    })
    if proba is not None:
        for j, c in enumerate(classes):
            out[f"p_{c}"] = proba[:, j].astype(np.float32)
    return out


def run(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
        encoder: Encoder, cache_dir: Path | None, batch_size: int,
        max_seq_length: int, team_names: dict[str, str],
        verbose: bool = True, c_grid=None,
        model_name: str = M.MODEL_NAME) -> dict:
    """Embed, tune, calibrate, route. Returns everything the caller may want
    to write; touches disk only through the embedding cache."""
    X_train = embed_split("train", train, encoder, cache_dir, batch_size, verbose)
    X_val = embed_split("val", val, encoder, cache_dir, batch_size, verbose)
    X_test = embed_split("test", test, encoder, cache_dir, batch_size, verbose)
    y_train = train[C.TEAM_COL].to_numpy()
    y_val = val[C.TEAM_COL].to_numpy()
    y_test = test[C.TEAM_COL].to_numpy()

    best_C, tuning = tune_C(X_train, y_train, X_val, y_val,
                            grid=c_grid or M.C_GRID, verbose=verbose)
    clf = fit_classifier(X_train, y_train, C=best_C)
    cal = calibrate(clf, X_val, y_val)
    classes = np.asarray(cal.classes_)

    val_proba, test_proba = cal.predict_proba(X_val), cal.predict_proba(X_test)
    val_pred, val_conf = classes[val_proba.argmax(1)], val_proba.max(1)
    test_pred, test_conf = classes[test_proba.argmax(1)], test_proba.max(1)

    ns, curves, teams = north_star_summary(
        val_conf, val_pred == y_val, test_conf, test_pred == y_test,
        y_test, test_pred, team_names)
    main = ns["operating_points"].get(C.NORTH_STAR_KEY)
    bundle = make_bundle(cal, encoder.name, max_seq_length,
                         None if main is None else main["threshold"],
                         model_name=model_name)
    return {
        "bundle": bundle, "ns": ns, "curves": curves, "teams": teams,
        "tuning": {"grid": tuning, "best_C": best_C},
        "metrics": {"val": _team_metrics(y_val, val_pred),
                    "test": _team_metrics(y_test, test_pred)},
        "val_predictions": _pred_frame(val, val_pred, val_conf, val_proba,
                                       classes),
        "test_predictions": _pred_frame(test, test_pred, test_conf,
                                        test_proba, classes),
        "y_test": y_test, "test_pred": test_pred,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--train-split", default="train_cap_issue",
                   choices=["train_cap_issue", "train_3k_team", "train_full"])
    p.add_argument("--encoder", default=M.ENCODER_NAME)
    p.add_argument("--max-seq-length", type=int, default=M.MAX_SEQ_LENGTH)
    p.add_argument("--batch-size", type=int, default=M.BATCH_SIZE)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--smoke", type=int, default=0, metavar="N",
                   help="use N rows per split, no cache, no artifacts written")
    p.add_argument("--c-grid", type=float, nargs="+", default=list(M.C_GRID))
    p.add_argument("--tag", default="",
                   help="save as improved_model_2_<tag> (e.g. a larger encoder) "
                        "instead of overwriting improved_model_2")
    args = p.parse_args()
    model_name = f"{M.MODEL_NAME}_{args.tag}" if args.tag else M.MODEL_NAME

    from src.utils.labels import team_names
    from src.utils.registry import log_result, north_star_columns

    t0 = time.time()
    train = load_split(args.train_split)
    val, test = load_split("val"), load_split("test")
    assert_same_test_ids(test)
    print(f"[data] train={len(train):,} ({args.train_split}) "
          f"val={len(val):,} test={len(test):,}")

    smoke = args.smoke > 0
    if smoke:
        train, val, test = (
            d.sample(n=min(args.smoke, len(d)), random_state=C.RANDOM_SEED)
            .reset_index(drop=True) for d in (train, val, test))

    encoder = load_encoder(args.encoder, args.max_seq_length, args.device)
    cache_dir = None if smoke else (
        M.EMBED_CACHE / f"{_slug(args.encoder)}_len{args.max_seq_length}"
        / args.train_split)
    out = run(train, val, test, encoder, cache_dir, args.batch_size,
              args.max_seq_length, team_names, c_grid=args.c_grid,
              model_name=model_name)

    main_pt = out["ns"]["operating_points"].get(C.NORTH_STAR_KEY)
    print(json.dumps({"metrics": out["metrics"],
                      f"north_star_{C.NORTH_STAR_KEY}": main_pt},
                     indent=2))
    if smoke:
        print("[smoke] done; nothing written")
        return

    art = M.ARTIFACT_DIR.parent / model_name
    art.mkdir(parents=True, exist_ok=True)
    joblib.dump(out["bundle"], art / "model.joblib", compress=3)
    (art / "metrics.json").write_text(json.dumps({
        "model": out["metrics"], "n_train": len(train), "n_val": len(val),
        "n_test": len(test)}, indent=2))
    (art / "north_star.json").write_text(json.dumps(out["ns"], indent=2))
    (art / "tuning.json").write_text(json.dumps(out["tuning"], indent=2))
    out["curves"].to_csv(art / "coverage_curve.csv", index=False)
    if out["teams"] is not None:
        out["teams"].to_csv(art / "per_team_routing.csv", index=False)
    labels = sorted(set(out["y_test"]) | set(out["test_pred"]))
    (art / "classification_report.json").write_text(json.dumps(
        classification_report(out["y_test"], out["test_pred"], labels=labels,
                              output_dict=True, zero_division=0), indent=2))
    pd.DataFrame(confusion_matrix(out["y_test"], out["test_pred"],
                                  labels=labels),
                 index=labels, columns=labels).to_csv(
        art / "confusion_matrix.csv", index_label="actual\\predicted")
    (art / "run_info.json").write_text(json.dumps({
        "model_name": model_name,
        "encoder": args.encoder, "max_seq_length": args.max_seq_length,
        "train_split": args.train_split,
        "wall_clock_sec": round(time.time() - t0, 1)}, indent=2))

    for split in ("val", "test"):
        write_predictions(model_name, split, out[f"{split}_predictions"])

    guard = out["ns"].get("guardrail") or {}
    log_result(model_name, "test", {
        "train_set": args.train_split,
        "team_acc": out["metrics"]["test"]["accuracy"],
        "team_macro_f1": out["metrics"]["test"]["macro_f1"],
        **north_star_columns(out["ns"]),
        "teams_below_floor": ",".join(guard.get("teams_below_floor", [])),
        "ece": out["ns"]["ece_test"],
        "n_eval": len(test),
    })
    print(f"[done] wrote {art} and prediction files for the gate")


if __name__ == "__main__":
    main()
