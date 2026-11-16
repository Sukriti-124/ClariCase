"""Prediction files shared by every model, and the joined frame the gate uses.

Layout (gitignored, under data/cache):
    data/cache/predictions/<model_name>/<split>.parquet
Columns: complaint_id, actual_team_id, predicted_team_id, confidence, n_words
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import (CACHE, DATE_COL, GROUP_COL, HOLDOUT_PARQUET,
                              ID_COL, SPLITS_V2)

PREDICTIONS_ROOT = CACHE / "predictions"
PRED_COLS = [ID_COL, "actual_team_id", "predicted_team_id", "confidence",
             "n_words"]


def predictions_path(model_name: str, split: str) -> Path:
    return PREDICTIONS_ROOT / model_name / f"{split}.parquet"


def write_predictions(model_name: str, split: str, df: pd.DataFrame) -> Path:
    missing = set(PRED_COLS) - set(df.columns)
    if missing:
        raise KeyError(f"predictions frame missing {sorted(missing)}")
    out = df.copy()
    out[ID_COL] = out[ID_COL].astype(str)
    path = predictions_path(model_name, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def read_predictions(model_name: str, split: str) -> pd.DataFrame:
    path = predictions_path(model_name, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Export it first, e.g. "
            f"`python -m src.ensemble.export_predictions --model {model_name}`")
    df = pd.read_parquet(path)
    df[ID_COL] = df[ID_COL].astype(str)
    return df


def assemble(primary: pd.DataFrame, secondaries: dict[str, pd.DataFrame],
             groups: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join per-model predictions on complaint_id.

    Returns columns: complaint_id, actual_team_id, n_words, primary_team,
    primary_conf, <name>__team, <name>__conf for each secondary, and `group`
    when a groups frame (complaint_id, leakage_group_id) is given.
    Rows missing from any model are dropped (inner join) so every model is
    scored on exactly the same complaints.
    """
    out = primary[[ID_COL, "actual_team_id", "n_words",
                   "predicted_team_id", "confidence"]].rename(columns={
                       "predicted_team_id": "primary_team",
                       "confidence": "primary_conf"})
    out[ID_COL] = out[ID_COL].astype(str)
    for name, df in secondaries.items():
        sec = df[[ID_COL, "predicted_team_id", "confidence"]].rename(columns={
            "predicted_team_id": f"{name}__team",
            "confidence": f"{name}__conf"})
        sec[ID_COL] = sec[ID_COL].astype(str)
        out = out.merge(sec, on=ID_COL, how="inner", validate="one_to_one")
    if groups is not None:
        g = groups[[ID_COL, GROUP_COL]].rename(columns={GROUP_COL: "group"})
        g[ID_COL] = g[ID_COL].astype(str)
        out = out.merge(g, on=ID_COL, how="left", validate="one_to_one")
    return out.reset_index(drop=True)


def load_groups(split: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS_V2 / f"{split}.parquet",
                           columns=[ID_COL, GROUP_COL])


def load_meta(split: str) -> pd.DataFrame:
    """complaint_id, leakage_group_id and date_received for a split. Columns
    the split does not have (the holdout may lack groups) come back empty."""
    import pyarrow.parquet as pq
    if split == "holdout_2026":
        from src.utils.data import _extract_holdout
        path = _extract_holdout()
    else:
        path = SPLITS_V2 / f"{split}.parquet"
    have = set(pq.read_schema(path).names)
    want = [ID_COL, GROUP_COL, DATE_COL]
    df = pd.read_parquet(path, columns=[c for c in want if c in have])
    for c in want:
        if c not in df.columns:
            df[c] = None
    df[ID_COL] = df[ID_COL].astype(str)
    return df[want]
