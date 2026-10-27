"""Dataset builders for FLAN-T5 team classification.

Reads the frozen splits via src.utils.data.load_split, applies the preprocess
+ head+tail truncation from tokenizer.py, and emits HF Datasets ready for
Seq2SeqTrainer with a DataCollatorForSeq2Seq collator.
"""
from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset

from src.improved_model_3.config import (
    MAX_TARGET_LEN, NUM_TEAMS, SEED, TEAM_IDS, TEAM_PROMPT_TEMPLATE,
)
from src.improved_model_3.tokenizer import head_tail_encode, preprocess
from src.utils.data import load_split


def _format_prompt(text: str) -> str:
    return TEAM_PROMPT_TEMPLATE.format(text=preprocess(text))


def build_team_dataset(split: str, tok, rows: int | None = None) -> Dataset:
    df = load_split(split)
    if rows is not None and len(df) > rows:
        df = df.sample(n=rows, random_state=SEED).reset_index(drop=True)

    texts = df["complaint_text"].tolist()
    teams = df["team_id"].tolist()

    def gen():
        for text, team in zip(texts, teams):
            prompt = _format_prompt(text)
            input_ids = head_tail_encode(prompt, tok)
            label_ids = tok(
                team,
                add_special_tokens=True,
                max_length=MAX_TARGET_LEN,
                truncation=True,
            )["input_ids"]
            yield {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": label_ids,
                "team_id": team,
            }

    return Dataset.from_generator(gen)


def class_weights_from_train(train_df=None) -> torch.Tensor:
    if train_df is None:
        train_df = load_split("train_cap_issue")
    counts = train_df["team_id"].value_counts().reindex(TEAM_IDS, fill_value=0)
    freq = counts.to_numpy().astype(np.float64)
    freq = np.maximum(freq, 1.0)
    inv = 1.0 / freq
    weights = inv * NUM_TEAMS / inv.sum()
    return torch.tensor(weights, dtype=torch.float32)
