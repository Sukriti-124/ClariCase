"""Write predictions with full probability vectors for validation and test
(and optionally the 2026 holdout), in the layout the stack, gate and issue
stage read: data/cache/predictions/<model>/<split>.parquet (team) and
<split>_issues.parquet (issue).

Models:
  improved_model_1   TF-IDF + SVM: team probabilities and issue probabilities
  improved_model_2   sentence embeddings + logistic regression: team
                     probabilities from the cached embeddings. Variants
                     trained with --tag (e.g. improved_model_2_base) work too.
  baseline_model_2   TF-IDF + Naive Bayes: team probabilities
  knn_embed          vote among the 25 most similar training complaints:
                     team and issue
  embed_issue_head   logistic regression per team on the embeddings: issue

All five use the same splits as improved_model_1 (data/cache/splits_v2).
The nearest-neighbour index and the embedding issue head are built from
train_cap_issue only.

Usage:
    python -m src.ensemble.export_predictions
    python -m src.ensemble.export_predictions --models knn_embed --splits val test holdout_2026
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd

from src.ensemble.data import write_predictions
from src.ensemble.embeddings import load_embed_bundle, split_embeddings
from src.ensemble.issue_head import conditional_issue_probs, fit_issue_heads
from src.ensemble.knn import topk_neighbours, vote
from src.ensemble.probs import (ISSUE_TEAM, ISSUES, TEAMS, block_normalize,
                                reorder, with_team_probs, write_issue_probs)
from src.utils.config import (BASELINE2_METRICS, CACHE, ID_COL, ISSUE_COL,
                              METRICS_ROOT, SPLITS_V2, TEAM_COL, TEXT_COL)

# Serving copy of the nearest-neighbour index (gitignored; mounted into
# embed-api). float16 halves the size; the vote is unaffected at k=25.
KNN_INDEX_PATH = CACHE / "knn_embed" / "index.npz"

ALL_MODELS = ("improved_model_1", "improved_model_2", "baseline_model_2",
              "knn_embed", "embed_issue_head")
COLS = [ID_COL, TEAM_COL, ISSUE_COL, TEXT_COL]


def load_frame(split: str) -> pd.DataFrame:
    if split == "holdout_2026":
        from src.utils.data import load_holdout_2026
        df = load_holdout_2026()[COLS]
    else:
        df = pd.read_parquet(SPLITS_V2 / f"{split}.parquet", columns=COLS)
    df[ID_COL] = df[ID_COL].astype(str)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df.reset_index(drop=True)


def team_frame(df: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    """Shared prediction-file columns plus p_<team> for every team."""
    teams = np.asarray(TEAMS)
    frame = pd.DataFrame({
        ID_COL: df[ID_COL].to_numpy(),
        "actual_team_id": df[TEAM_COL].astype(str).to_numpy(),
        "predicted_team_id": teams[P.argmax(1)],
        "confidence": P.max(1),
        "n_words": df[TEXT_COL].str.split().str.len().fillna(0)
                   .astype(int).to_numpy(),
    })
    return with_team_probs(frame, P, TEAMS)


def _batches(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(s + size, n)


# ---- improved_model_1 -------------------------------------------------------

def im1_issue_conditional(hsvm, X) -> np.ndarray:
    """Every team head's issue distribution for every row, in one pass per
    team. Same probabilities as HierarchicalSVM.predict_issue_topk gives row
    by row, but batched."""
    from src.improved_model_1.model import _ConstantIssue
    col = {iss: j for j, iss in enumerate(ISSUES)}
    Q = np.zeros((X.shape[0], len(ISSUES)))
    for team, clf in hsvm.issue_clfs_.items():
        if isinstance(clf, _ConstantIssue):
            if clf.issue_id in col:
                Q[:, col[clf.issue_id]] = 1.0
            continue
        cal = hsvm.calibrated_issue_.get(team)
        if cal is not None:
            P, classes = cal.predict_proba(X), cal.classes_
        else:
            s = np.asarray(clf.decision_function(X), dtype=np.float64)
            if s.ndim == 1:  # binary LinearSVC returns one score per row
                s = np.column_stack([-s, s])
            P = np.exp(s - s.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
            classes = clf.classes_
        for j, c in enumerate(classes):
            if str(c) in col:
                Q[:, col[str(c)]] = P[:, j]
    return Q


def export_improved_model_1(splits, batch_size: int = 20_000, **_):
    from src.improved_model_1.predict import load_model
    bundle = load_model()
    vec, hsvm = bundle["vectorizer"], bundle["model"]
    for split in splits:
        df = load_frame(split)
        print(f"[export] improved_model_1 / {split}: {len(df):,} rows",
              flush=True)
        Ps, Qs = [], []
        for s, e in _batches(len(df), batch_size):
            X = vec.transform(df[TEXT_COL].iloc[s:e].tolist())
            Ps.append(reorder(hsvm.team_proba(X), bundle["team_classes"], TEAMS))
            Qs.append(im1_issue_conditional(hsvm, X))
            print(f"  {e:,}/{len(df):,}", flush=True)
        write_predictions("improved_model_1", split,
                          team_frame(df, np.vstack(Ps)))
        write_issue_probs("improved_model_1", split, df, np.vstack(Qs))


# ---- improved_model_2 -------------------------------------------------------

def export_improved_model_2(splits, embed_model: str = "improved_model_2",
                            device=None, **_):
    bundle = load_embed_bundle(embed_model)
    for split in splits:
        df = load_frame(split)
        print(f"[export] {embed_model} / {split}: {len(df):,} rows", flush=True)
        X = split_embeddings(split, df, bundle, device)
        P = reorder(bundle["classifier"].predict_proba(X),
                    bundle["team_classes"], TEAMS)
        write_predictions(embed_model, split, team_frame(df, P))


# ---- baseline_model_2 -------------------------------------------------------

def export_baseline_model_2(splits, batch_size: int = 20_000, **_):
    model = joblib.load(BASELINE2_METRICS / "model.joblib")
    for split in splits:
        df = load_frame(split)
        print(f"[export] baseline_model_2 / {split}: {len(df):,} rows",
              flush=True)
        P = np.vstack([
            reorder(model.predict_proba(df[TEXT_COL].iloc[s:e].tolist()),
                    model.classes_, TEAMS)
            for s, e in _batches(len(df), batch_size)])
        write_predictions("baseline_model_2", split, team_frame(df, P))


# ---- embedding neighbours and issue head -----------------------------------

_TRAIN_CACHE: dict = {}


def _train_embeddings(embed_model: str, device=None):
    if embed_model not in _TRAIN_CACHE:
        bundle = load_embed_bundle(embed_model)
        df = load_frame("train_cap_issue")
        X = split_embeddings("train_cap_issue", df, bundle, device)
        _TRAIN_CACHE[embed_model] = (bundle, df, X)
    return _TRAIN_CACHE[embed_model]


def export_knn(splits, embed_model: str = "improved_model_2", device=None,
               knn_k: int = 25, knn_backend: str = "auto", **_):
    bundle, tdf, TX = _train_embeddings(embed_model, device)
    tix = {t: i for i, t in enumerate(TEAMS)}
    iix = {iss: i for i, iss in enumerate(ISSUES)}
    keep = tdf[ISSUE_COL].isin(iix).to_numpy()
    if not keep.all():
        print(f"[knn] dropping {int((~keep).sum())} training rows whose issue "
              "is not in the canonical list")
    TX = TX[keep]
    team_lab = tdf.loc[keep, TEAM_COL].map(tix).to_numpy()
    issue_lab = tdf.loc[keep, ISSUE_COL].map(iix).to_numpy()
    # Same keep mask and row order as the labels, so row i of the index is
    # one training complaint.
    complaint_ids = tdf.loc[keep, ID_COL].astype(str).to_numpy()
    write_knn_index(TX, team_lab, issue_lab, bundle["encoder_name"],
                    bundle["max_seq_length"], knn_k,
                    complaint_ids=complaint_ids)
    for split in splits:
        df = load_frame(split)
        print(f"[export] knn_embed / {split}: {len(df):,} rows, k={knn_k}",
              flush=True)
        t0 = time.time()
        QX = split_embeddings(split, df, bundle, device)
        sims, idx = topk_neighbours(TX, QX, knn_k, backend=knn_backend,
                                    verbose=True)
        print(f"  neighbours in {time.time() - t0:.0f}s", flush=True)
        write_predictions("knn_embed", split,
                          team_frame(df, vote(idx, sims, team_lab, len(TEAMS))))
        write_issue_probs("knn_embed", split, df, block_normalize(
            vote(idx, sims, issue_lab, len(ISSUES)), ISSUE_TEAM))


def write_knn_index(TX, team_lab, issue_lab, encoder_name: str,
                    max_seq_length: int, k: int,
                    path=KNN_INDEX_PATH, *, complaint_ids=None) -> None:
    """Write the serving index. `complaint_ids`, when given, must have one
    entry per row of `TX` in the same order; it is stored as fixed-width
    unicode so the file still loads with allow_pickle=False. Omitting it
    writes the old layout."""
    arrays = dict(embeddings=np.asarray(TX, dtype=np.float16),
                  team_idx=np.asarray(team_lab, dtype=np.int16),
                  issue_idx=np.asarray(issue_lab, dtype=np.int16),
                  teams=np.array(TEAMS), issues=np.array(ISSUES),
                  encoder_name=np.array(encoder_name),
                  max_seq_length=np.array(max_seq_length), k=np.array(k))
    if complaint_ids is not None:
        ids = np.asarray([str(c) for c in complaint_ids], dtype=str)
        if ids.shape != (arrays["embeddings"].shape[0],):
            raise ValueError(
                f"complaint_ids has shape {ids.shape}; expected "
                f"({arrays['embeddings'].shape[0]},), one id per index row")
        arrays["complaint_ids"] = ids
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print(f"[knn] wrote serving index {path} "
          f"({path.stat().st_size / 1e6:.0f} MB)", flush=True)


def export_embed_issue_head(splits, embed_model: str = "improved_model_2",
                            device=None, **_):
    bundle, tdf, TX = _train_embeddings(embed_model, device)
    print("[export] fitting embedding issue heads on train_cap_issue",
          flush=True)
    heads = fit_issue_heads(TX, tdf[TEAM_COL].to_numpy(),
                            tdf[ISSUE_COL].to_numpy())
    out = METRICS_ROOT / "embed_issue_head"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_name": "embed_issue_head", "heads": heads,
                 "embed_model": embed_model}, out / "model.joblib", compress=3)
    for split in splits:
        df = load_frame(split)
        print(f"[export] embed_issue_head / {split}: {len(df):,} rows",
              flush=True)
        X = split_embeddings(split, df, bundle, device)
        write_issue_probs("embed_issue_head", split, df,
                          conditional_issue_probs(heads, X))


EXPORTERS = {
    "improved_model_1": export_improved_model_1,
    "improved_model_2": export_improved_model_2,
    "baseline_model_2": export_baseline_model_2,
    "knn_embed": export_knn,
    "embed_issue_head": export_embed_issue_head,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS),
                   help="any of the above, or an improved_model_2_<tag> variant")
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   choices=["val", "test", "holdout_2026"])
    p.add_argument("--embed-model", default="improved_model_2",
                   help="whose cached embeddings knn_embed and "
                        "embed_issue_head use")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--knn-k", type=int, default=25)
    p.add_argument("--knn-backend", default="auto",
                   choices=["auto", "torch", "numpy"])
    args = p.parse_args()
    for name in args.models:
        t0 = time.time()
        if name.startswith("improved_model_2"):
            export_improved_model_2(args.splits, embed_model=name,
                                    device=args.device)
        elif name in EXPORTERS:
            EXPORTERS[name](args.splits, embed_model=args.embed_model,
                            device=args.device, knn_k=args.knn_k,
                            knn_backend=args.knn_backend)
        else:
            raise SystemExit(f"unknown model {name!r}")
        print(f"[export] {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
