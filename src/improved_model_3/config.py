"""Static configuration for the FLAN-T5-base seq2seq classifier.

Values here are read by tokenizer.py, data.py, model.py, train.py, predict.py,
and calibrate.py. See plan 06 for the rationale of every choice.
"""
from __future__ import annotations

from src.utils.config import CACHE, METRICS_ROOT, RANDOM_SEED
from src.utils.labels import team_to_issues

MODEL_NAME = "google/flan-t5-base"

ARTIFACT_DIR = METRICS_ROOT / "improved_model_3"
CACHE_DIR = CACHE / "improved_model_3"
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"
TOKENIZER_DIR = CACHE_DIR / "tokenizer_with_specials"

TEAM_PROMPT_TEMPLATE = "classify complaint team: {text}"
ISSUE_PROMPT_TEMPLATE = "issue in {team}: {text}"

SPECIAL_TOKENS = ("[REDACTED]", "[AMOUNT]", "[DATE]")
REDACT_PATTERNS = {
    "redact": r"X{3,}",
    "amount": r"\{\$[^}]*\}",
    "date": r"XX/XX/(?:XX|\d{2,4})",
}

MAX_INPUT_LEN = 320
HEAD_FRACTION = 0.75
MAX_TARGET_LEN = 8

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("q", "v")

PER_DEVICE_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
NUM_EPOCHS_TEAM = 2
NUM_EPOCHS_ISSUE = 1
EVAL_STEPS = 1000
SAVE_STEPS = 1000
LOGGING_STEPS = 50
SAVE_TOTAL_LIMIT = 1
EARLY_STOP_PATIENCE = 2

VAL_SUBSET_SIZE = 20_000

TEAM_IDS: tuple[str, ...] = tuple(sorted(team_to_issues.keys()))
NUM_TEAMS = len(TEAM_IDS)

SEED = RANDOM_SEED

for _p in (ARTIFACT_DIR, CACHE_DIR, CHECKPOINT_DIR, TOKENIZER_DIR):
    _p.mkdir(parents=True, exist_ok=True)
