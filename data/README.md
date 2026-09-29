# Complaint datasets

Collected and prepared by **Sriramm S S ([SriRammSS](https://github.com/SriRammSS))** for [Data collection #7](https://github.com/farcypeer46/ClariCase/issues/7).

| Archive | Coverage | Contents |
|---|---|---|
| [2024–2025 complaint dataset](complaint_router_11_teams_2024_2025.zip) | 1,044,615 complaints; 3,801 companies; 11 teams; 73 product-specific issues | Full CSV with original narratives, mappings, support counts, excluded-record audit, source metadata and construction checks |
| [January–February 2026 evaluation data](complaint_router_2026_holdout.zip) | 29,798 complaints; 11 teams; 72 supported issues | Parquet dataset, source/exclusion metadata, issue support and usage notes |

The first archive is the unchanged data-construction snapshot. Its README and metadata describe that stage. Subsequent text preprocessing and temporal preparation are documented in [preparation/README.md](preparation/README.md). The 2026 archive is separate from training data and has already been used for evaluation; it should not be described as unseen for future changes.

## Download

The ZIPs use Git LFS because the complete 2024–2025 archive exceeds GitHub's normal file limit. On GitHub, open an archive and use **Download raw file** to download its actual contents. For a clone with Git LFS installed:

```sh
git lfs install
git lfs pull --include="data/complaint_router*.zip"
```

If a downloaded file is a short text document beginning with `version https://git-lfs.github.com/spec/v1`, it is the pointer rather than the dataset; download it through Git LFS or the raw-file button. File sizes and SHA-256 checksums are recorded in [archives_manifest.json](archives_manifest.json).

## Load the main dataset

Run from the repository root. This extracts only the CSV into the path already used by the repository's data loader:

```python
from pathlib import Path
from zipfile import ZipFile
import pandas as pd

destination = Path("data/processed")
destination.mkdir(parents=True, exist_ok=True)
with ZipFile("data/complaint_router_11_teams_2024_2025.zip") as archive:
    archive.extract("complaints_product_issues_2024_2025.csv", destination)

chunks = pd.read_csv(
    destination / "complaints_product_issues_2024_2025.csv",
    dtype={"complaint_id": "string"},
    keep_default_na=False,
    chunksize=50_000,
)
for frame in chunks:
    texts = frame["complaint_text"]
    issue_labels = frame["product_issue_id"]
```

The CSV expands to approximately 1.91 GB. The original narrative is the only text feature. Product, issue, sub-issue, company, team IDs, group hashes and source metadata are retained for labels and auditing. The 73 issue labels map to the 11 project teams in [issue_mapping.csv](processed/issue_mapping.csv).

For the separate 2026 dataset, extract the second ZIP into its own directory and load `fresh_holdout_2026_01_02.parquet` with `pandas.read_parquet` and a Parquet engine such as PyArrow. Do not append it to the training data while continuing to report it as an independent evaluation set.

## Collection and validation

- [Construction rules and team coverage](processed/README.md)
- [Archive sources, exclusions and checksums](processed/dataset_metadata.json)
- [Ten passing construction checks](processed/validation.json)
- [Text preprocessing and temporal preparation](preparation/README.md)
- [Contribution in session 04](../reports/session04.md#data-collection-and-preparation)

The data consists of published CFPB complaint narratives. Consumer-selected labels can be ambiguous, and the retained complaints are not a representative sample of all financial complaints. Source coverage is January 2024–December 2025 plus the separate January–February 2026 cohort.
