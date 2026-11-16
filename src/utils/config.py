from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FULL_CSV = PROCESSED / "complaints_product_issues_2024_2025.csv"
CACHE = ROOT / "data" / "cache"
FULL_PARQUET = CACHE / "complaints_product_issues_2024_2025.parquet"
SPLITS_V2 = CACHE / "splits_v2"
HOLDOUT_ZIP = ROOT / "data" / "complaint_router_2026_holdout.zip"
HOLDOUT_PARQUET = CACHE / "holdout_2026" / "fresh_holdout_2026_01_02.parquet"
ISSUE_MAPPING = PROCESSED / "issue_mapping.csv"
TEAM_MAPPING = PROCESSED / "team_mapping.csv"
BASELINE2_METRICS = ROOT / "docs" / "metrics" / "baseline_model_2"
METRICS_ROOT = ROOT / "docs" / "metrics"
REPORTS = ROOT / "reports"

TEXT_COL, TEAM_COL, ISSUE_COL = "complaint_text", "team_id", "issue_id"
GROUP_COL, DATE_COL, ID_COL = "leakage_group_id", "date_received", "complaint_id"
USECOLS = [ID_COL, DATE_COL, "company", TEXT_COL, TEAM_COL, "team_name",
           "product_issue_id", GROUP_COL]

RANDOM_SEED = 42
CHUNKSIZE = 100_000
TRAIN_END = "2025-07-01"
VAL_END = "2025-10-01"
TEST_END = "2026-01-01"
TRAIN_CAP_PER_ISSUE = 10_000
BASELINE2_CAP_PER_TEAM = 3_000
MERGE_ISSUE_VARIANTS = True
ISSUE_VARIANT_MERGES = {"T01_I007": "T01_I008",
                        "T01_I003": "T01_I001"}

PRECISION_TARGETS = (0.95, 0.90, 0.85)
# North star: share of complaints auto-routed at this precision or better.
# Moved from 0.95 to 0.90 on 2026-11-15. Every target in PRECISION_TARGETS is
# still computed and saved, so 95% numbers stay comparable with older runs.
NORTH_STAR_TARGET = 0.90
NORTH_STAR_KEY = f"at_{int(NORTH_STAR_TARGET * 100)}p"
TEAM_PRECISION_FLOOR = 0.85
MIN_ROUTED = 50
N_BOOTSTRAP = 1000
ECE_BINS = 10
