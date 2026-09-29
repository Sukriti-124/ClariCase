from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.baseline_model_2.config import (
    CHUNKSIZE, DATE_COL, FULL_CSV, GROUP_COL, ID_COL, LABEL_COL, RANDOM_SEED,
    SAMPLE_PER_TEAM, SPLIT_SCHEME, SPLITS_DIR, TEST_END, TRAIN_END, USECOLS,
    VAL_END,
)

SPLITS = ("train", "validation", "test")
SPLIT_INFO = SPLITS_DIR / "split_info.json"


def _read(csv_path, usecols):
    return pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNKSIZE,
                       dtype={ID_COL: "string"}, keep_default_na=False)


def assign_splits(csv_path=FULL_CSV, per_team=SAMPLE_PER_TEAM,
                  seed=RANDOM_SEED, verbose=True) -> pd.Series:
    # Pass 1 (no text): period per record, drop groups spanning periods,
    # cap training per team. Returns complaint_id -> split.
    meta = pd.concat(
        _read(csv_path, [ID_COL, DATE_COL, LABEL_COL, GROUP_COL]),
        ignore_index=True)
    meta = meta[meta[LABEL_COL].astype(bool)]

    dates = pd.to_datetime(meta[DATE_COL])
    meta["split"] = np.select(
        [dates < TRAIN_END, dates < VAL_END, dates < TEST_END],
        list(SPLITS), default="")
    meta = meta[meta["split"] != ""]

    spans = meta.groupby(GROUP_COL)["split"].transform("nunique") > 1
    if verbose:
        print(f"[split] temporal: train < {TRAIN_END} <= val < {VAL_END} "
              f"<= test < {TEST_END}")
        print(f"  dropped {spans.sum():,} records in "
              f"{meta.loc[spans, GROUP_COL].nunique():,} leakage groups "
              f"spanning periods")
    meta = meta[~spans]

    # Training is capped per team; validation and test keep the real mix.
    train = (meta[meta["split"] == "train"]
             .sample(frac=1, random_state=seed)
             .groupby(LABEL_COL).head(per_team))
    keep = pd.concat([train, meta[meta["split"] != "train"]])
    return keep.set_index(ID_COL)["split"]


def build_and_save_splits(verbose: bool = True) -> None:
    assignment = assign_splits(verbose=verbose)

    # Pass 2: pull full rows for the selected ids only.
    parts: dict[str, list[pd.DataFrame]] = {s: [] for s in SPLITS}
    for chunk in _read(FULL_CSV, USECOLS):
        split = chunk[ID_COL].map(assignment)
        for name in SPLITS:
            parts[name].append(chunk[split == name])
    frames = {s: pd.concat(p, ignore_index=True) for s, p in parts.items()}
    frames["train"] = frames["train"].sample(
        frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Invariants
    groups = {s: set(f[GROUP_COL]) for s, f in frames.items()}
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
    SPLIT_INFO.write_text(json.dumps(
        {**SPLIT_SCHEME, "counts": {s: len(f) for s, f in frames.items()}},
        indent=2))

    if verbose:
        counts = pd.DataFrame({s: f[LABEL_COL].value_counts()
                               for s, f in frames.items()}).fillna(0).astype(int)
        print(f"\n{counts.to_string()}")
        for name, frame in frames.items():
            print(f"  {name:<10} {len(frame):>8,}  "
                  f"{frame[DATE_COL].min()} to {frame[DATE_COL].max()}")
        print(f"\n[write] {SPLITS_DIR}")


def splits_current() -> bool:
    # True when the saved splits were built with the current SPLIT_SCHEME.
    if not SPLIT_INFO.exists():
        return False
    info = json.loads(SPLIT_INFO.read_text())
    return {k: info.get(k) for k in SPLIT_SCHEME} == SPLIT_SCHEME


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Read the persisted splits. Call build_and_save_splits() first.
    return tuple(pd.read_csv(SPLITS_DIR / f"{name}.csv",
                             dtype={ID_COL: "string"}, keep_default_na=False)
                 for name in SPLITS)
