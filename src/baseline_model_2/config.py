from pathlib import Path

# Path and constants for the TF-IDF team-classification baseline.
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FULL_CSV = PROCESSED / "complaints_product_issues_2024_2025.csv"
SPLITS_DIR = PROCESSED / "splits"
TEAM_MAPPING = PROCESSED / "team_mapping.csv"
ARTIFACTS_DIR = ROOT / "docs" / "metrics" / "baseline_model_2"

TEXT_COL = "complaint_text"
LABEL_COL = "team_id"
LABEL_NAME_COL = "team_name"
GROUP_COL = "leakage_group_id"
DATE_COL = "date_received"
ID_COL = "complaint_id"

USECOLS = [ID_COL, DATE_COL, "product", "issue", TEXT_COL,
           LABEL_COL, LABEL_NAME_COL, GROUP_COL]

# Training is capped per team; validation and test keep every eligible record
# so they reflect the real team mix.
SAMPLE_PER_TEAM = 3000
CHUNKSIZE = 100_000
RANDOM_SEED = 42

# Temporal split: train < TRAIN_END <= validation < VAL_END <= test < TEST_END.
# Leakage groups that span more than one period are dropped.
TRAIN_END = "2025-07-01"
VAL_END = "2025-10-01"
TEST_END = "2026-01-01"
SPLIT_SCHEME = {"scheme": "temporal", "train_end": TRAIN_END,
                "val_end": VAL_END, "test_end": TEST_END,
                "sample_per_team": SAMPLE_PER_TEAM, "seed": RANDOM_SEED}

# North star: share auto-routed at this precision, thresholds chosen on
# validation and frozen for test. Teams below the floor fail the guardrail.
PRECISION_TARGETS = (0.95, 0.90, 0.85)
TEAM_PRECISION_FLOOR = 0.85
MIN_ROUTED = 50
N_BOOTSTRAP = 1000