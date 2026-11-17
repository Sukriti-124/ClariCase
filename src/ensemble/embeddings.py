"""Sentence embeddings for any split, reusing improved_model_2's cache.

Training already cached train/val/test vectors under data/cache/improved_model_2/.
Other splits (the 2026 holdout) are encoded on first use with the same
encoder and cached next to them.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.config import METRICS_ROOT

SPLIT_FILE = {"train_cap_issue": "train", "val": "val", "test": "test",
              "holdout_2026": "holdout_2026"}


def load_embed_bundle(model_name: str = "improved_model_2") -> dict:
    return joblib.load(METRICS_ROOT / model_name / "model.joblib")


def cache_dir_for(bundle: dict, train_split: str = "train_cap_issue") -> Path:
    from src.improved_model_2.config import EMBED_CACHE
    from src.improved_model_2.train import _slug
    return (EMBED_CACHE / f"{_slug(bundle['encoder_name'])}"
            f"_len{bundle['max_seq_length']}" / train_split)


class LazyEncoder:
    """Loads the sentence encoder only if something actually needs encoding,
    so splits that are already cached need neither torch nor a download."""

    def __init__(self, bundle: dict, device: str | None = None):
        self._bundle, self._device, self._enc = bundle, device, None
        self.name = bundle["encoder_name"]

    def encode(self, texts, batch_size=64, show_progress=False):
        if self._enc is None:
            from src.improved_model_2.predict import encoder_for
            self._enc = encoder_for(self._bundle, self._device)
        return self._enc.encode(texts, batch_size=batch_size,
                                show_progress=show_progress)


def split_embeddings(split: str, df: pd.DataFrame, bundle: dict,
                     device: str | None = None, batch_size: int = 64,
                     train_split: str = "train_cap_issue") -> np.ndarray:
    from src.improved_model_2.train import embed_split
    return embed_split(SPLIT_FILE.get(split, split), df,
                       LazyEncoder(bundle, device),
                       cache_dir_for(bundle, train_split), batch_size)
