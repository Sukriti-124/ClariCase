"""Model builder: FLAN-T5-base + LoRA, plus constrained-decoding helpers.

Why constrained decoding: the model must emit exactly one of the 11 team ids
(e.g., "T01"). A `prefix_allowed_tokens_fn` built from a trie over the
tokenized team strings guarantees that at every decode step the only allowed
next tokens are those on at least one valid team path.
"""
from __future__ import annotations

from typing import Callable

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM

from src.improved_model_3.config import (
    LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGET_MODULES, MODEL_NAME,
    TEAM_IDS,
)


def build_team_model(tok, bf16: bool = True):
    dtype = torch.bfloat16 if bf16 else torch.float32
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    base.resize_token_embeddings(len(tok))
    base.gradient_checkpointing_enable()
    base.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model


def _tokenize_team_strings(tok) -> list[list[int]]:
    """Target sequences HF puts into `labels` already include eos; we need the
    sequence as seen on the DECODER side, which starts with decoder_start_token
    and then produces the label tokens. We only need the label tokens here —
    the constrained-decode trie keys off decoder output step indices 0..N-1."""
    seqs: list[list[int]] = []
    for team in TEAM_IDS:
        ids = tok(team, add_special_tokens=True)["input_ids"]
        # strip trailing eos — generate() picks eos naturally at the end
        if tok.eos_token_id is not None and ids and ids[-1] == tok.eos_token_id:
            ids = ids[:-1]
        seqs.append(ids)
    return seqs


def build_team_prefix_fn(tok) -> Callable[[int, torch.Tensor], list[int]]:
    """Returns a prefix_allowed_tokens_fn for HF generate().

    HF calls fn(batch_id, input_ids_so_far) and expects a list of allowed next
    token ids. For T5, input_ids_so_far starts with the decoder_start_token,
    then the generated tokens. We skip the first element (decoder start) and
    match the remainder against our team-string prefixes.
    """
    team_seqs = _tokenize_team_strings(tok)
    eos = tok.eos_token_id

    def fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        ids = input_ids.tolist()
        # drop decoder_start_token at index 0
        generated = ids[1:] if len(ids) >= 1 else []
        allowed: set[int] = set()
        for seq in team_seqs:
            if len(generated) < len(seq) and generated == seq[: len(generated)]:
                allowed.add(seq[len(generated)])
            elif generated == seq and eos is not None:
                allowed.add(eos)
        if not allowed and eos is not None:
            allowed.add(eos)
        return sorted(allowed)

    return fn
