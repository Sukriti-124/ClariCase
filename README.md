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
complaint and routes it to the support team that handles it (one of 11 teams,
such as Credit Reporting, Mortgage, or Money Transfers), then gives the
consumer a tracking ID to check on it later. No product or issue dropdowns are
needed.

The running model is **Baseline 2**, TF-IDF with a calibrated Multinomial
Naive Bayes classifier, trained on CFPB complaints from all companies
(2024–2025).

| Metric (test set, Oct–Dec 2025, 109,834 complaints) | Value |
|---|---:|
| Accuracy | 0.800 |
| Macro F1 | 0.647 |
| **North star:** complaints auto-routed at the 95% precision threshold | **46.5%** (at 94.4% precision) |

Full details are in [docs/baseline_model_2.md](docs/baseline_model_2.md).

## Repository layout

| Path | Contents |
|---|---|
| `streamlit_app.py` | The complaint intake app |
| `src/app/` | Complaint storage (Supabase or local SQLite) and the Supabase table schema |
| `src/baseline_model_2/` | Running model: splits, training, evaluation, prediction |
| `src/baseline_model_1/` | Earlier product classifier (JPMorgan Chase data only) |
| `src/data/` | Data pipeline for Baseline 1 |
| `docs/` | Model write-ups; metrics and artifacts under `docs/metrics/` |
| `data/processed/` | Constructed all-company dataset and its documentation |
| `reports/` | Weekly session reports |

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
  Submit. The app shows the team it was sent to and a tracking ID
  (for example `CC-72LSLC`).
- **Track my complaint:** entering the tracking ID shows the status, the
  team, and the original complaint.

Without Supabase credentials, the app runs in **local mode** and stores
complaints in `data/app/complaints.db`. This file is not committed.

### Connecting Supabase

1. Create a Supabase project and run [src/app/schema.sql](src/app/schema.sql)
   in its SQL Editor.
2. Copy the **Project URL** and the **anon / publishable** key from
   Project Settings → API.
3. Create `.streamlit_secrets.toml`. It is gitignored and must never be
   committed.

   ```toml
   [supabase]
   url = "https://your-project.supabase.co"
   key = "your-anon-key"
   ```

### Deploying to Streamlit Community Cloud

1. Push the repository to GitHub, including
   `docs/metrics/baseline_model_2/model.joblib`, which the app loads.
2. In Streamlit Community Cloud, create an app from this repository with
   main file `streamlit_app.py`.
3. Under **Advanced settings**, choose the newest Python version available.
4. Paste the `[supabase]` block above into the app's **Secrets**.

Free Supabase projects pause after about a week without activity, and
Streamlit apps sleep when unused. Open both before a demo.

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
