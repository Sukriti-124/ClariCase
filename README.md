# ClariCase

We help financial services teams understand consumer complaints and identify actionable patterns more efficiently than manual review.

**Repository:** https://github.com/farcypeer46/ClariCase

**Live app:** https://claricase-enmuypbz78oqjgiec98bhw.streamlit.app/

---

## Team

| Name | Role |
|------|-----|
| Chaitanya Bagul | Developer, Engineering |
| Sukriti Srivastava | Developer, Product |
| Salman Farcy | Developer, Data and Evaluation |
| Sriramm S S | Developer, Users and Research |

## What it does

A consumer describes their problem in plain words. ClariCase reads the
complaint, routes it to the support team that handles it (one of 11 teams,
such as Credit Reporting, Mortgage, or Money Transfers), suggests the most
likely issues, and gives the consumer a tracking ID to check on it later. No
product or issue dropdowns are needed.

The running model is **Improved Model 1**: a hierarchical TF-IDF + Linear SVM
classifier that picks the team first, then the issue within that team. It is
trained on CFPB complaints from all companies (2024–2025), with isotonic
calibration and a single routing threshold, both fit on validation data and
frozen for test.

| Metric (test set, Oct–Dec 2025, 109,834 complaints) | Baseline 2 | **Improved Model 1** |
|---|---:|---:|
| Accuracy | 0.800 | **0.844** |
| Macro F1 | 0.647 | **0.726** |
| **North star:** auto-routed at the 95% precision threshold | 46.5% (at 94.4%) | **61.5%** (at 94.8%) |
| Teams below the 0.85 precision floor | T09 | **none** |
| Issue macro-F1 (predicted team) | — | 0.367 |

Complaints at or above the threshold (confidence 0.818) are routed
automatically; the rest go to a person for review. Full write-ups:
[docs/improved_model_1.md](docs/improved_model_1.md) and
[docs/baseline_model_2.md](docs/baseline_model_2.md).

## Repository layout

| Path | Contents |
|---|---|
| `streamlit_app.py` | The complaint intake app |
| `src/app/` | Complaint storage (Supabase or local SQLite) and the Supabase table schema |
| `src/improved_model_1/` | **Running model:** hierarchical TF-IDF + Linear SVM (team, then issue) |
| `src/baseline_model_2/` | Previous team-only model (TF-IDF + Naive Bayes) |
| `src/baseline_model_1/` | Earlier product classifier (JPMorgan Chase data only) |
| `src/utils/` | Shared library: splits, metrics, routing, labels, device, registry |
| `src/data/` | Data pipeline for Baseline 1 |
| `tests/` | Unit + parity tests (`pytest -q` runs everything, `-m slow` for data-heavy) |
| `docs/` | Model write-ups; metrics and artifacts under `docs/metrics/` |
| `docs/plans/` | Local planning notes (gitignored, not in the repository) |
| `data/processed/` | Constructed all-company dataset and its documentation |
| `data/cache/` | Local-only Parquet splits and model caches (gitignored) |
| `reports/` | Weekly session reports and `model_comparison.csv` |

## Local setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

`scikit-learn` is pinned to the version the deployed model was saved with.
Do not upgrade it without retraining the model.

## Running the app

```powershell
streamlit run streamlit_app.py
```

The app opens at http://localhost:8501 with two tabs:

- **Submit a complaint:** the consumer describes the problem and clicks
  Submit. The app shows whether the complaint was **Routed** or is **Under
  review**, the team (the top 2 or 3 teams with percentages when the model is
  less sure), the top 3 likely issues, and a tracking ID (for example
  `CC-72LSLC`).
- **Track my complaint:** entering the tracking ID shows the status, the
  team, the most likely issue, and the original complaint.

Without Supabase credentials, the app runs in **local mode** and stores
complaints in `data/app/complaints.db`. This file is not committed.

### Connecting Supabase

1. Create a Supabase project and run [src/app/schema.sql](src/app/schema.sql)
   in its SQL Editor.
2. Copy the **Project URL** and the **secret** key (`sb_secret_…`, or
   `service_role` on older projects) from Project Settings → API Keys.
   The schema turns on row-level security with no access policies, so the
   publishable (anon) key cannot read or write complaints; only the secret
   key, kept on the server, can.
3. To run locally against Supabase, create `.streamlit/secrets.toml` (a file
   inside the `.streamlit` folder). It is gitignored and must never be
   committed. Streamlit reads secrets only from this file or from the Cloud
   **Secrets** box.

   ```toml
   [supabase]
   url = "https://your-project.supabase.co"
   key = "your-secret-key"
   ```

### Deploying to Streamlit Community Cloud

1. Push the repository to GitHub, including
   `docs/metrics/improved_model_1/model.joblib`, which the app loads.
2. In Streamlit Community Cloud, create an app from this repository with
   main file `streamlit_app.py`.
3. Under **Advanced settings**, choose the newest Python version available.
4. Paste the `[supabase]` block above, with the secret key, into the app's
   **Secrets**. Never put the secret key in any file in the repository.

Free Supabase projects pause after about a week without activity, and
Streamlit apps sleep when unused. Open both before a demo.

## Improved Model 1: the running model

```powershell
# Route a single complaint: team, top issues, and auto/review decision
python -m src.improved_model_1.predict "complaint text here"

# Run the test suite (fast tests only; drop -m to include data-heavy ones)
python -m pytest -q -m "not slow"

# Retrain: build the Parquet splits once, then train (~30-40 min without
# character n-grams; the full run takes several hours)
python -m src.utils.build_splits
python -m src.improved_model_1.train --no-char
```

Training needs the full dataset described below. Outputs are written to
`docs/metrics/improved_model_1/`. Step-by-step instructions are in
[docs/how_to_run.md](docs/how_to_run.md).

## Baseline 2: team routing model

The input dataset `data/processed/complaints_product_issues_2024_2025.csv`
(about 1.9 GB) is not stored in Git. Place it in `data/processed/` before
training. Its construction is documented in
[data/processed/README.md](data/processed/README.md).

```powershell
# Build the temporal splits (if needed), train, calibrate, and evaluate
python -m src.baseline_model_2.train

# Force the splits to be rebuilt from the full dataset
python -m src.baseline_model_2.train --rebuild-splits

# Route a single complaint and show the top 3 teams
python -m src.baseline_model_2.predict "complaint text here" --top-k 3
```

Outputs are written to `docs/metrics/baseline_model_2/`: the calibrated
model, metrics, classification report, confusion matrix, north star summary,
per-team routing guardrail, coverage curve, and test predictions.

## Baseline 1: product classifier (JPMorgan Chase)

The first baseline predicts the product category of JPMorgan Chase complaints
with TF-IDF and a calibrated LinearSVC. It reached a north star of 26.4%. See
[docs/baseline_model_1.md](docs/baseline_model_1.md).

The raw input is `data/raw/complaints.csv`, which is not stored in Git.
Run the modules from the repository root so imports resolve correctly:

```powershell
# Load the raw CSV and display label counts
python -m src.data.ingest

# Rebuild docs/data_profile.md
python -m src.data.explore

# Clean, deduplicate, and display the temporal train/test split
python -m src.data.prepare

# Train and evaluate the baseline
python -m src.baseline_model_1.baseline --output-dir docs/metrics/baseline_model_1
```

Cleaning is performed in memory; `prepare.py` does not write a processed copy
to disk.
