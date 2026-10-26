"""Tokenizer setup: add CFPB redaction special tokens, head+tail truncation.

The CFPB narratives are heavily redacted. We normalize three patterns to
dedicated special tokens so the model sees a stable vocabulary:
    XXXX...     -> [REDACTED]
    {$1,200.00} -> [AMOUNT]
    XX/XX/2024  -> [DATE]

Head+tail truncation keeps the first 75% and last 25% of the token budget,
because complainants often repeat the ask at the end.
"""
from __future__ import annotations

import re

from transformers import AutoTokenizer

from src.improved_model_3.config import (
    HEAD_FRACTION, MAX_INPUT_LEN, MODEL_NAME, REDACT_PATTERNS,
    SPECIAL_TOKENS, TOKENIZER_DIR,
)

_RE_REDACT = re.compile(REDACT_PATTERNS["redact"])
_RE_AMOUNT = re.compile(REDACT_PATTERNS["amount"])
_RE_DATE = re.compile(REDACT_PATTERNS["date"])


def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _RE_DATE.sub(" [DATE] ", text)
    text = _RE_AMOUNT.sub(" [AMOUNT] ", text)
    text = _RE_REDACT.sub(" [REDACTED] ", text)
    return " ".join(text.split())


def load_tokenizer(reload: bool = False):
    if TOKENIZER_DIR.exists() and any(TOKENIZER_DIR.iterdir()) and not reload:
        return AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.add_special_tokens(
        {"additional_special_tokens": list(SPECIAL_TOKENS)}
    )
    tok.save_pretrained(TOKENIZER_DIR)
    return tok


def head_tail_encode(text: str, tok, max_len: int = MAX_INPUT_LEN) -> list[int]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id
    budget = max_len - 1 if eos is not None else max_len
    if len(ids) <= budget:
        truncated = ids
    else:
        head_n = int(budget * HEAD_FRACTION)
        tail_n = budget - head_n
        truncated = ids[:head_n] + ids[-tail_n:]
    if eos is not None:
        truncated = truncated + [eos]
    return truncated
