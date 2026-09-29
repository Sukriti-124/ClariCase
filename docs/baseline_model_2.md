# Baseline Model 2: Team Routing Classifier

## Objective

The second baseline routes a consumer complaint to the support team that
should handle it, using only the written complaint narrative.

| Role | Dataset column | Description |
|---|---|---|
| Input feature | `complaint_text` | Free-text complaint submitted by the consumer |
| Target variable | `team_id` | One of 11 support teams (`T01`–`T11`) |

Each team corresponds to one CFPB product category, as defined in
`data/processed/team_mapping.csv`:

| Team | Name | Product |
|---|---|---|
| T01 | Credit Reporting Team | Credit reporting or other personal consumer reports |
| T02 | Debt Collection Team | Debt collection |
| T03 | Credit Card Team | Credit card |
| T04 | Banking Accounts Team | Checking or savings account |
| T05 | Money Transfers & Digital Currency Team | Money transfer, virtual currency, or money service |
| T06 | Mortgage Team | Mortgage |
| T07 | Student Loan Team | Student loan |
| T08 | Vehicle Finance Team | Vehicle loan or lease |
| T09 | Personal & Short-Term Loans Team | Payday loan, title loan, personal loan, or advance loan |
| T10 | Prepaid Card Team | Prepaid card |
| T11 | Debt & Credit Management Team | Debt or credit management |

Compared with Baseline 1, this model covers **all companies** in the CFPB
archive rather than JPMorgan Chase alone, and all 11 products rather than 7.

## Data

The input is the constructed dataset
`data/processed/complaints_product_issues_2024_2025.csv`: 1,044,615 complaints
received January 1, 2024 – December 31, 2025, from 3,801 companies. Its
construction rules are documented in `data/processed/README.md`. In short, it
removes exact duplicate narratives, excludes narratives whose identical text
carries conflicting labels, and assigns a `leakage_group_id` that groups
near-identical template complaints.

The real team mix is heavily imbalanced: Credit Reporting accounts for 62% of
complaints and Debt & Credit Management for 0.3%.

## Data Preparation

`src/baseline_model_2/data_loader.py` builds a temporal split:

| Split | Period | Records | Team mix |
|---|---|---:|---|
| Training | January 1, 2024 – June 30, 2025 | 32,599 | Capped at 3,000 per team |
| Validation | July 1 – September 30, 2025 | 153,502 | Every eligible record (real mix) |
| Test | October 1 – December 31, 2025 | 109,834 | Every eligible record (real mix) |

1. Each record is assigned to a period by `date_received`.
2. Leakage groups that span more than one period are dropped entirely, so no
   near-duplicate template appears on both sides of a boundary. This removed
   6,867 records in 1,949 groups.
3. Training is sampled at random, up to 3,000 records per team, so small teams
   are well represented. Debt & Credit Management contributes all 2,599 of its
   training-period records.
4. Validation and test are **not** resampled. They keep the team mix the
   product would actually see, which is what makes the routing metrics
   meaningful.

The splits are written to `data/processed/splits/` together with
`split_info.json`, which records the settings used. They are rebuilt
automatically if those settings change.

Only `complaint_text` is used as a feature. Product, issue, company, IDs, and
group keys are never model inputs.

## Dummy Baseline

The dummy classifier uses the `most_frequent` strategy. It ignores the
complaint text and always predicts the most common training class. Because
training is capped at 3,000 per team, ten teams tie, and scikit-learn breaks
the tie with the first label, `T01` Credit Reporting. Its test accuracy
(0.527) is therefore simply the share of Credit Reporting complaints.

## TF-IDF and Multinomial Naive Bayes Model

The model is a scikit-learn pipeline, wrapped in a calibration step:

1. `TfidfVectorizer` converts complaint text into word and two-word phrase
   features, with sublinear term frequency, accent stripping, and a maximum
   vocabulary of 75,000 features.
2. `MultinomialNB` (`alpha=0.1`, uniform class prior) learns a probability
   model for each team. Balanced sample weights are applied during training.
3. `CalibratedClassifierCV` with **isotonic regression** is fitted on the
   validation set, with the trained pipeline frozen.

Calibration matters for two reasons. Naive Bayes probabilities are typically
overconfident, and the model learned a uniform team prior from the capped
training sample. Fitting the calibrator on validation data with the real team
mix corrects both, so the confidence scores can be used to decide which
complaints are safe to route automatically.

The calibrated model is stored as one artifact, so the vocabulary, classifier,
and calibrator are loaded together for predictions.

## Evaluation Results

All results are on the test set (October – December 2025, real team mix).

| Metric | Dummy | TF-IDF + NB (uncalibrated) | TF-IDF + NB (calibrated) |
|---|---:|---:|---:|
| Accuracy | 0.527 | 0.759 | **0.800** |
| Macro F1 | 0.063 | 0.574 | **0.647** |
| Weighted F1 | 0.364 | 0.768 | **0.793** |

Per-team results for the calibrated model:

| Team | Precision | Recall | F1 | Test records |
|---|---:|---:|---:|---:|
| T01 Credit Reporting | 0.865 | 0.931 | 0.897 | 57,889 |
| T06 Mortgage | 0.803 | 0.894 | 0.846 | 3,367 |
| T07 Student Loan | 0.735 | 0.764 | 0.749 | 1,517 |
| T04 Banking Accounts | 0.659 | 0.805 | 0.725 | 10,164 |
| T08 Vehicle Finance | 0.690 | 0.718 | 0.704 | 2,857 |
| T03 Credit Card | 0.710 | 0.664 | 0.686 | 9,725 |
| T02 Debt Collection | 0.813 | 0.584 | 0.680 | 16,116 |
| T05 Money Transfers | 0.704 | 0.433 | 0.536 | 5,236 |
| T09 Personal & Short-Term Loans | 0.483 | 0.562 | 0.519 | 1,640 |
| T10 Prepaid Card | 0.506 | 0.468 | 0.486 | 856 |
| T11 Debt & Credit Management | 0.476 | 0.212 | 0.293 | 467 |

The weakest teams are the smallest ones. Debt & Credit Management and Prepaid
Card have fewer than 1,000 test records, so their metrics should be treated as
less stable.

## North Star: Autonomous Routing Rate

The north star metric is the **share of complaints routed automatically at
≥95% precision**: a complaint is auto-routed when the model's confidence is at
or above a threshold, and everything below it goes to a human.

The protocol is:

1. Choose the threshold on the **validation** set: the lowest confidence at
   which precision on routed complaints is at least 95%.
2. Apply that threshold unchanged to the **test** set and report what it
   actually achieves.
3. Report a 95% bootstrap confidence interval (1,000 resamples).

| Precision target | Threshold | Validation coverage | Test coverage | Test precision | Routed |
|---|---:|---:|---:|---:|---:|
| **95%** | 0.872 | 48.2% | **46.5%** | 94.4% | 51,034 |
| 90% | 0.652 | 73.8% | 72.2% | 89.7% | 79,272 |
| 85% | 0.470 | 90.3% | 89.5% | 84.3% | 98,267 |

**North star: 46.5% (95% CI 46.2–46.8%) at 94.4% precision (95% CI
94.2–94.7%).** The threshold hit 95% precision on validation but lands just
below it on the later test period, which is the expected effect of time drift
and the reason thresholds are not tuned on test.

Calibration error (ECE) on the test set is **0.026**, compared with 0.066
before calibration, so the confidence scores are reliable enough to set
thresholds on.

### Per-team guardrail

A high overall rate could hide poor routing for small teams, because Credit
Reporting makes up most of the traffic. At the 95% threshold, precision is
therefore also checked for the complaints routed *to* each team, against a
floor of 85%.

| Team | Routed to team | Precision of routed | Share of team's complaints auto-routed | Meets 85% floor |
|---|---:|---:|---:|:---:|
| T01 | 40,565 | 0.950 | 67.1% | Yes |
| T02 | 1,750 | 0.902 | 17.5% | Yes |
| T03 | 1,793 | 0.948 | 24.5% | Yes |
| T04 | 2,226 | 0.900 | 20.8% | Yes |
| T05 | 629 | 0.991 | 15.7% | Yes |
| T06 | 1,985 | 0.956 | 56.9% | Yes |
| T07 | 552 | 0.917 | 36.7% | Yes |
| T08 | 912 | 0.906 | 32.1% | Yes |
| T09 | 305 | 0.823 | 18.6% | **No** |
| T10 | 274 | 0.898 | 29.9% | Yes |
| T11 | 43 | 0.930 | 20.8% | Yes |

Personal & Short-Term Loans (T09) falls below the floor. Credit Reporting
accounts for about 80% of all auto-routed complaints; most other teams have
only 15–35% of their complaints handled automatically. The T11 figure rests on
43 routed complaints and is noisy.

## Comparison with Baseline 1

| | Baseline 1 | Baseline 2 |
|---|---|---|
| Data | JPMorgan Chase only, 22,096 records | All companies, 1.04M records |
| Target | Product (7 classes) | Team (11 classes) |
| Model | TF-IDF + calibrated LinearSVC | TF-IDF + Naive Bayes, isotonic calibration |
| Split | Temporal, train/test | Temporal, train/validation/test |
| Threshold chosen on | Test | Validation |
| North star | 26.4% | 46.5% (at 94.4% precision) |

The two numbers are not directly comparable. The datasets, label sets, team
mixes, and protocols differ.

## Known Limitations

- The 95% target is not met on the test set (94.4%).
- Validation is used both to fit the calibrator and to choose thresholds,
  which is slightly optimistic. Splitting validation into two halves would
  separate these.
- Labels are selected by consumers at intake and are not audited. Some share
  of the measured errors is label noise, which caps achievable precision.
- One global threshold favours Credit Reporting. Per-team thresholds are a
  likely next step.

## Running the Pipeline

Run all commands from the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1

# Build the temporal splits (if needed), train, calibrate, and evaluate
python -m src.baseline_model_2.train

# Force the splits to be rebuilt from the full CSV
python -m src.baseline_model_2.train --rebuild-splits

# Route a single complaint and show the top 3 teams
python -m src.baseline_model_2.predict "complaint text here" --top-k 3
```

Building the splits reads the full 1.9 GB dataset and takes about a minute;
later runs reuse the saved splits.

## Generated Artifacts

Training writes the following files under `docs/metrics/baseline_model_2/`:

| File | Purpose |
|---|---|
| `model.joblib` | Calibrated TF-IDF and Naive Bayes pipeline (not committed to Git) |
| `metrics.json` | Dummy, uncalibrated, and calibrated summary metrics on validation and test |
| `classification_report.json` | Precision, recall, and F1 by team on test |
| `confusion_matrix.csv` | Actual versus predicted team counts on test |
| `north_star.json` | North star, operating points, confidence intervals, ECE, and guardrail result |
| `per_team_routing.csv` | Per-team guardrail at the 95% threshold |
| `coverage_curve.csv` | Coverage and precision at every threshold, for validation and test |
| `predictions_test.csv` | Actual and predicted team, confidence, and auto-routed flag for every test record |

The implementation is located in `src/baseline_model_2/`
(`config.py`, `data_loader.py`, `train.py`, `predict.py`).
