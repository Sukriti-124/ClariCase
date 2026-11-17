"""Settings for improved_model_2 (frozen sentence embeddings + logistic regression)."""
import os

from src.utils.config import CACHE, METRICS_ROOT

MODEL_NAME = "improved_model_2"
ARTIFACT_DIR = METRICS_ROOT / MODEL_NAME
EMBED_CACHE = CACHE / MODEL_NAME

# Small, fast, CPU-friendly encoder. Override with IM2_ENCODER to try a larger
# one (e.g. BAAI/bge-base-en-v1.5); embeddings are cached per encoder.
ENCODER_NAME = os.environ.get("IM2_ENCODER", "BAAI/bge-small-en-v1.5")
MAX_SEQ_LENGTH = int(os.environ.get("IM2_MAX_SEQ_LENGTH", "256"))
BATCH_SIZE = int(os.environ.get("IM2_BATCH_SIZE", "64"))

# The first full run (grid 0.5, 2, 8) picked the smallest value, so the grid
# now reaches lower.
C_GRID = (0.1, 0.25, 0.5, 2.0)
MAX_ITER = 1000
