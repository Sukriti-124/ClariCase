"""Smoke-run-first training entrypoint for the FLAN-T5-base seq2seq classifier.

Usage:
    python -m src.improved_model_3.train --smoke          # 50k rows, 1 epoch
    python -m src.improved_model_3.train                  # full train, 2 epochs
    python -m src.improved_model_3.train --rows 25000
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_callback import EarlyStoppingCallback

from src.improved_model_3.config import (
    ARTIFACT_DIR, CHECKPOINT_DIR, EARLY_STOP_PATIENCE, EVAL_STEPS,
    GRAD_ACCUM_STEPS, LEARNING_RATE, LOGGING_STEPS, MAX_TARGET_LEN,
    NUM_EPOCHS_TEAM, PER_DEVICE_BATCH_SIZE, SAVE_STEPS, SAVE_TOTAL_LIMIT,
    SEED, TEAM_IDS, VAL_SUBSET_SIZE, WARMUP_STEPS,
)
from src.improved_model_3.data import build_team_dataset
from src.improved_model_3.model import build_team_model
from src.improved_model_3.tokenizer import load_tokenizer
from src.utils.device import gpu_info


def _make_compute_metrics(tok, team_ids):
    team_set = set(team_ids)

    def compute(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        labels = np.where(labels == -100, tok.pad_token_id, labels)
        pred_strs = tok.batch_decode(preds, skip_special_tokens=True)
        label_strs = tok.batch_decode(labels, skip_special_tokens=True)
        y_true = [s.strip() for s in label_strs]
        y_pred = [s.strip() if s.strip() in team_set else "UNK"
                  for s in pred_strs]
        macro = f1_score(y_true, y_pred, average="macro",
                         labels=list(team_ids), zero_division=0)
        acc = sum(a == b for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)
        return {"team_macro_f1": float(macro), "team_accuracy": float(acc)}

    return compute


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="50k rows, 1 epoch, no early stop")
    p.add_argument("--rows", type=int, default=None,
                   help="Override train row count")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override num epochs")
    p.add_argument("--split", type=str, default="train_cap_issue",
                   choices=["train_cap_issue", "train_full", "train_3k_team"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    rows = 50_000 if args.smoke else args.rows
    epochs = (1 if args.smoke else NUM_EPOCHS_TEAM) \
        if args.epochs is None else args.epochs

    tok = load_tokenizer()

    print(f"[train] device info: {gpu_info()}", flush=True)
    print(f"[train] building datasets: split={args.split} rows={rows}",
          flush=True)
    train_ds = build_team_dataset(args.split, tok, rows=rows)
    val_ds = build_team_dataset("val", tok, rows=VAL_SUBSET_SIZE)
    print(f"[train] train={len(train_ds)} val_eval_subset={len(val_ds)}",
          flush=True)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    model = build_team_model(tok, bf16=bf16)

    args_tr = Seq2SeqTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="linear",
        bf16=bf16,
        fp16=fp16,
        optim="adamw_bnb_8bit" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=True,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        generation_num_beams=1,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="team_macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )

    collator = DataCollatorForSeq2Seq(tok, padding="longest",
                                      return_tensors="pt",
                                      pad_to_multiple_of=8)

    callbacks = [] if args.smoke else [
        EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)]

    trainer = Seq2SeqTrainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds.remove_columns(["team_id"]),
        eval_dataset=val_ds.remove_columns(["team_id"]),
        processing_class=tok,
        data_collator=collator,
        compute_metrics=_make_compute_metrics(tok, TEAM_IDS),
        callbacks=callbacks,
    )

    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    peak_vram = (torch.cuda.max_memory_allocated() / 1024 ** 3
                 if torch.cuda.is_available() else 0.0)
    run_info = {
        "gpu_info": gpu_info(),
        "peak_vram_gb": round(float(peak_vram), 2),
        "wall_clock_sec": round(wall, 1),
        "train_rows": len(train_ds),
        "val_eval_subset": len(val_ds),
        "epochs": epochs,
        "split": args.split,
        "smoke": args.smoke,
        "model_name": "google/flan-t5-base",
        "lora_r": 16,
        "per_device_batch": PER_DEVICE_BATCH_SIZE,
        "grad_accum": GRAD_ACCUM_STEPS,
    }
    (ARTIFACT_DIR / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"[train] done in {wall:.1f}s, peak_vram={peak_vram:.2f} GB",
          flush=True)
    print(f"[train] run_info -> {ARTIFACT_DIR / 'run_info.json'}", flush=True)

    final_metrics = trainer.evaluate()
    (ARTIFACT_DIR / ("metrics_smoke.json" if args.smoke
                     else "metrics_eval_subset.json")).write_text(
        json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
