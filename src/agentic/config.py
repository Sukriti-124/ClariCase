"""Settings for the agentic candidate-choice stage.

Env vars (a .env at the repo root is read if python-dotenv is installed):
    AGENTIC_BACKEND      openai_compat | mock          (default openai_compat)
    AGENTIC_BASE_URL     OpenAI-compatible endpoint    (default Groq)
    AGENTIC_API_KEY      key for that endpoint; GROQ_API_KEY / OPENAI_API_KEY
                         are read as fallbacks
    AGENTIC_MODEL        model id (default llama-3.3-70b-versatile)
    AGENTIC_CONCURRENCY  parallel requests (default 4)
"""
from __future__ import annotations

import os

from src.utils.config import CACHE, METRICS_ROOT, ROOT

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

BACKEND = os.environ.get("AGENTIC_BACKEND", "openai_compat")
BASE_URL = os.environ.get("AGENTIC_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = (os.environ.get("AGENTIC_API_KEY") or os.environ.get("GROQ_API_KEY")
           or os.environ.get("OPENAI_API_KEY") or "")
MODEL = os.environ.get("AGENTIC_MODEL", "gpt-oss-120b")
CONCURRENCY = int(os.environ.get("AGENTIC_CONCURRENCY", "4"))
TIMEOUT_S = 30.0
MAX_RETRIES = 5

CACHE_DIR = CACHE / "agentic"
LLM_CACHE = CACHE_DIR / "llm_cache.sqlite"
EXAMPLE_POOL = CACHE_DIR / "example_pool"      # .npz + .parquet
ARTIFACT_DIR = METRICS_ROOT / "agentic"

# Candidate sets
N_TEAM_CANDIDATES = 3
N_ISSUE_CANDIDATES = 3
# Ask about the issue only when the ML head's top-1 minus top-2 probability
# is below this. Measured on test: top-1 accuracy is 44-54% below 0.3 and
# 70-87% above it, while top-3 is 93-98% throughout.
ISSUE_GAP_THRESHOLD = float(os.environ.get("AGENTIC_ISSUE_GAP", "0.3"))

# Retrieval
EXAMPLES_PER_CANDIDATE = 2
POOL_CAP_PER_ISSUE = 400
MAX_COMPLAINT_CHARS = 1500
MAX_EXAMPLE_CHARS = 350
