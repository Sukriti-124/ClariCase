# Baseline NLP Classification Model

## Objective

The baseline model classifies a consumer complaint into its financial product
category using only the written complaint narrative.

| Role | Dataset column | Description |
|---|---|---|
| Input feature | `Consumer complaint narrative` | Free-text complaint submitted by the consumer |
| Target variable | `Product` | Product category that the model predicts |

Example product classes include `Credit card`, `Mortgage`, `Debt collection`,
`Checking or savings account`, and `Vehicle loan or lease`.

## Data Preparation

The existing pipeline in `src/data/prepare.py` performs the following steps:

1. Removes records without a complaint narrative.
2. Keeps records from January 2024 through January 2026.
3. Normalizes whitespace and Unicode characters.
4. Removes narratives containing fewer than five tokens.
5. Removes four classes with insufficient examples.
6. Removes duplicate narratives before splitting the data.
7. Uses complaints before July 1, 2025 for training and later complaints for
   testing.

The prepared dataset contains 22,096 records:

| Split | Records | Date range |
|---|---:|---|
| Training | 15,714 | January 1, 2024 to June 30, 2025 |
| Test | 6,382 | July 1, 2025 to January 30, 2026 |

This temporal split tests the model on future complaints instead of randomly
mixing older and newer records. Post-outcome fields such as company response,
timely response, and tags are not used as model features.

## Dummy Baseline

The dummy classifier uses the `most_frequent` strategy. It ignores the
complaint narrative and always predicts the most common training class,
`Checking or savings account`.

The dummy model provides a minimum performance reference. A useful classifier
should perform substantially better than this result.

## TF-IDF and LinearSVC Model

The actual baseline is a scikit-learn pipeline with two stages:

1. `TfidfVectorizer` converts complaint text into numeric word and two-word
   phrase features. It uses sublinear term frequency and a maximum vocabulary
   of 75,000 features.
2. `LinearSVC` learns a linear decision boundary for each product category.
   Balanced class weights reduce the dominance of large classes.

The fitted pipeline is stored as one artifact, so the same TF-IDF vocabulary
and classifier can be loaded together for later predictions.

## Evaluation Results

| Metric | Dummy | TF-IDF + LinearSVC |
|---|---:|---:|
| Accuracy | 0.463 | 0.799 |
| Macro F1 | 0.090 | 0.731 |
| Weighted F1 | 0.293 | 0.799 |

Accuracy is the overall fraction of correct predictions. Macro F1 gives every
class equal importance, while weighted F1 accounts for the number of examples
in each class. The macro F1 improvement indicates that the trained model learns
smaller classes instead of only predicting the majority class.

Mortgage and vehicle-loan classes have fewer than 200 test records, so their
individual metrics should be treated as less stable.

## Running the Pipeline

Run all commands from the repository root in PowerShell:

```powershell
cd F:\NLP_MSML641
.\.venv\Scripts\Activate.ps1

# Inspect the raw dataset and label counts
python -m src.data.ingest

# Rebuild the exploratory data profile
python -m src.data.explore

# Run cleaning and display the temporal split
python -m src.data.prepare

# Train and evaluate the baseline classifier
python -m src.models.baseline
```

## Generated Artifacts

Training writes the following local files under `reports/baseline/`:

| File | Purpose |
|---|---|
| `model.joblib` | Fitted TF-IDF and LinearSVC pipeline |
| `metrics.json` | Dummy and trained-model summary metrics |
| `classification_report.json` | Precision, recall, and F1 by class |
| `confusion_matrix.csv` | Actual versus predicted class counts |
| `predictions.csv` | Actual and predicted labels for every test record |

The implementation is located in `src/models/baseline.py`. No Git commit or
push is required to run the pipeline locally.
