"""Evaluate the trained FLAN-T5-base + LoRA team classifier on the test split.

Usage:
    python -m src.improved_model_3.predict                       # latest ckpt, test split
    python -m src.improved_model_3.predict --split val
    python -m src.improved_model_3.predict --checkpoint data/cache/improved_model_3/checkpoints/checkpoint-3125
    python -m src.improved_model_3.predict --rows 5000           # subsample for a quick look

Writes to docs/metrics/improved_model_3/:
    metrics_test.json              overall accuracy + macro/weighted F1
    per_team_metrics_test.json     precision/recall/f1 per team
    confusion_matrix_test.csv      rows=true team, cols=predicted team
    predictions_test.parquet       complaint_id, team_id (true), pred_team_id
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

from src.improved_model_3.config import (
    ARTIFACT_DIR, CHECKPOINT_DIR, MAX_TARGET_LEN, MODEL_NAME, TEAM_IDS,
    TEAM_PROMPT_TEMPLATE,
)
from src.improved_model_3.model import build_team_prefix_fn
from src.improved_model_3.tokenizer import head_tail_encode, load_tokenizer, preprocess
from src.utils.data import load_split


def _latest_checkpoint() -> Path:
    ckpts = sorted(
        CHECKPOINT_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints under {CHECKPOINT_DIR}")
    return ckpts[-1]


def _load_model(checkpoint: Path, tok):
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    base.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base, str(checkpoint))
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def _encode_row(text: str, tok) -> dict:
    prompt = TEAM_PROMPT_TEMPLATE.format(text=preprocess(text))
    ids = head_tail_encode(prompt, tok)
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test",
                   choices=["test", "val"])
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to checkpoint-N dir. Default: latest under CHECKPOINT_DIR.")
    p.add_argument("--rows", type=int, default=None,
                   help="Optional: subsample N rows for a quick evaluation.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-beams", type=int, default=1,
                   help="1 = greedy (fastest). Constrained decoding already guarantees valid labels.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = args.checkpoint or _latest_checkpoint()
    print(f"[predict] checkpoint: {ckpt}", flush=True)
    print(f"[predict] split: {args.split}", flush=True)

    tok = load_tokenizer()
    model = _load_model(ckpt, tok)
    prefix_fn = build_team_prefix_fn(tok)

    df = load_split(args.split).reset_index(drop=True)
    if args.rows is not None and len(df) > args.rows:
        df = df.sample(n=args.rows, random_state=0).reset_index(drop=True)
    print(f"[predict] rows: {len(df)}", flush=True)

    collator = DataCollatorForSeq2Seq(tok, padding="longest",
                                      return_tensors="pt",
                                      pad_to_multiple_of=8)
    encoded = [_encode_row(t, tok) for t in df["complaint_text"].tolist()]
    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collator)

    device = next(model.parameters()).device
    preds: list[str] = []
    t0 = time.time()
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ("input_ids", "attention_mask")}
            out = model.generate(
                **batch,
                max_new_tokens=MAX_TARGET_LEN,
                num_beams=args.num_beams,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_fn,
            )
            preds.extend(tok.batch_decode(out, skip_special_tokens=True))
            if (i + 1) % 20 == 0:
                done = (i + 1) * args.batch_size
                rate = done / (time.time() - t0 + 1e-9)
                print(f"[predict] {min(done, len(df))}/{len(df)} "
                      f"({rate:.0f}/s)", flush=True)
    wall = time.time() - t0

    team_set = set(TEAM_IDS)
    y_true = [s.strip() for s in df["team_id"].tolist()]
    y_pred = [s.strip() if s.strip() in team_set else "UNK" for s in preds]

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro",
                     labels=list(TEAM_IDS), zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted",
                        labels=list(TEAM_IDS), zero_division=0)
    unk_rate = sum(p == "UNK" for p in y_pred) / len(y_pred)

    metrics = {
        "checkpoint": str(ckpt),
        "split": args.split,
        "rows": len(df),
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "unk_rate": float(unk_rate),
        "wall_clock_sec": round(wall, 1),
        "throughput_rows_per_sec": round(len(df) / wall, 1),
    }
    out_name = f"metrics_{args.split}.json"
    (ARTIFACT_DIR / out_name).write_text(json.dumps(metrics, indent=2))
    print(f"[predict] {out_name}: acc={acc:.4f} "
          f"macro_f1={macro:.4f} weighted_f1={weighted:.4f} "
          f"unk={unk_rate:.4f}", flush=True)

    report = classification_report(
        y_true, y_pred, labels=list(TEAM_IDS),
        output_dict=True, zero_division=0,
    )
    per_team = {t: {k: float(v) for k, v in report[t].items()}
                for t in TEAM_IDS if t in report}
    (ARTIFACT_DIR / f"per_team_metrics_{args.split}.json").write_text(
        json.dumps(per_team, indent=2))

    cm = confusion_matrix(y_true, y_pred, labels=list(TEAM_IDS))
    pd.DataFrame(cm, index=list(TEAM_IDS), columns=list(TEAM_IDS)).to_csv(
        ARTIFACT_DIR / f"confusion_matrix_{args.split}.csv")

    out_df = pd.DataFrame({
        "complaint_id": df.get("complaint_id", pd.Series(range(len(df)))),
        "team_id": y_true,
        "pred_team_id": y_pred,
    })
    out_df.to_parquet(ARTIFACT_DIR / f"predictions_{args.split}.parquet",
                      index=False)
    print(f"[predict] wrote per-team, confusion matrix, and predictions "
          f"to {ARTIFACT_DIR}", flush=True)


if __name__ == "__main__":
    main()
