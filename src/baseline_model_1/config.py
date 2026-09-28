"""Paths and constants for the TF-IDF team-classification baseline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FULL_CSV = PROCESSED / "complaints_product_issues_2024_2025.csv"
SPLITS_DIR = PROCESSED / "splits"
TEAM_MAPPING = PROCESSED / "team_mapping.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

TEXT_COL = "complaint_text"
LABEL_COL = "team_id"
LABEL_NAME_COL = "team_name"
GROUP_COL = "leakage_group_id"
DATE_COL = "date_received"
ID_COL = "complaint_id"

USECOLS = [ID_COL, DATE_COL, "product", "issue", TEXT_COL,
           LABEL_COL, LABEL_NAME_COL, GROUP_COL]

SAMPLE_PER_TEAM = 3000
CHUNKSIZE = 100_000
RANDOM_SEED = 42

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20
