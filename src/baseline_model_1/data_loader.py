"""Stratified sampling + group-aware train/val/test split.

Reads the full processed CSV in chunks, samples up to SAMPLE_PER_TEAM records
per team_id, then splits with GroupShuffleSplit on leakage_group_id so template
near-duplicates never straddle partitions.

Splits are persisted under data/processed/splits/ so every teammate trains and
evaluates on the same rows.

Run: python -m src.baseline_model_1.data_loader
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.baseline_model_1.config import (
    ARTIFACTS_DIR, CHUNKSIZE, FULL_CSV, GROUP_COL, LABEL_COL, RANDOM_SEED,
    SAMPLE_PER_TEAM, SPLITS_DIR, TEST_FRAC, TRAIN_FRAC, USECOLS, VAL_FRAC,
)


def sample_stratified(csv_path=FULL_CSV, per_team=SAMPLE_PER_TEAM,
                      chunksize=CHUNKSIZE, seed=RANDOM_SEED,
                      verbose=True) -> pd.DataFrame:
    """Fill a per-team bucket up to `per_team` rows by streaming the CSV.

    Iterates chunks; each chunk contributes a random sub-sample to every team
    that still has room. Stops early once every team's bucket is full.
    """
    buckets: dict[str, list[pd.DataFrame]] = {}
    counts: dict[str, int] = {}

    reader = pd.read_csv(
        csv_path,
        usecols=USECOLS,
        dtype={"complaint_id": "string"},
        chunksize=chunksize,
        keep_default_na=False,
    )

    for i, chunk in enumerate(reader, start=1):
        chunk = chunk[chunk[LABEL_COL].astype(bool)]
        for tid, group in chunk.groupby(LABEL_COL, observed=True):
            have = counts.get(tid, 0)
            if have >= per_team:
                continue
            need = per_team - have
            take = group if len(group) <= need else group.sample(
                n=need, random_state=seed + i)
            buckets.setdefault(tid, []).append(take)
            counts[tid] = have + len(take)

        if verbose:
            filled = sum(1 for v in counts.values() if v >= per_team)
            print(f"  chunk {i:>3}: teams filled {filled}/{len(counts)} "
                  f"(seen {len(counts)} distinct)")

        # Early stop: every team we've seen is full AND we've seen all 11
        if len(counts) >= 11 and all(v >= per_team for v in counts.values()):
            break

    df = pd.concat([pd.concat(parts, ignore_index=True)
                    for parts in buckets.values()], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if verbose:
        print(f"\n[sample] {len(df):,} rows across {df[LABEL_COL].nunique()} teams")
        print(df[LABEL_COL].value_counts().to_string())
    return df


def group_split(df: pd.DataFrame, seed: int = RANDOM_SEED
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-aware 60/20/20 split on leakage_group_id.

    Two-step GroupShuffleSplit: first carve out TEST, then split remaining
    into TRAIN and VAL. Ensures no leakage_group_id is present in more than
    one partition.
    """
    groups = df[GROUP_COL].values

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, groups=groups))
    trainval = df.iloc[trainval_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    val_ratio = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(
        gss2.split(trainval, groups=trainval[GROUP_COL].values))
    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)

    # Invariants
    assert set(train[GROUP_COL]).isdisjoint(val[GROUP_COL])
    assert set(train[GROUP_COL]).isdisjoint(test[GROUP_COL])
    assert set(val[GROUP_COL]).isdisjoint(test[GROUP_COL])
    return train, val, test


def build_and_save_splits(verbose: bool = True) -> None:
    df = sample_stratified(verbose=verbose)
    train, val, test = group_split(df)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "validation.csv", index=False)
    test.to_csv(SPLITS_DIR / "test.csv", index=False)

    if verbose:
        print(f"\n[split] group-aware on {GROUP_COL}")
        for name, part in (("train", train), ("val", val), ("test", test)):
            share = len(part) / len(df)
            print(f"  {name:<5} {len(part):>6,}  ({share:>5.1%})")
        print(f"\n[write] {SPLITS_DIR}")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the persisted splits. Call build_and_save_splits() first."""
    train = pd.read_csv(SPLITS_DIR / "train.csv", keep_default_na=False)
    val = pd.read_csv(SPLITS_DIR / "validation.csv", keep_default_na=False)
    test = pd.read_csv(SPLITS_DIR / "test.csv", keep_default_na=False)
    return train, val, test


if __name__ == "__main__":
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    build_and_save_splits(verbose=True)
