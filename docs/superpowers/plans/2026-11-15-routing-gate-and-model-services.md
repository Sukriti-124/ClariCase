# Routing gate and per-model services: plan and status

**Date:** 2026-11-15
**Branch:** `week-11-designing-a-system`
**Status:** Steps 1 to 3, 3b and the ML part of step 5 are trained and scored on the real data (3e). The label-shift correction and the serving of the chosen setup are implemented and unit tested (3f). The images have not been built. Steps 4 and 6 are planned only.
**Related:** `docs/superpowers/specs/2026-11-11-platform-phase-1-foundation-design.md`

## Goal

Raise the north star above what Improved Model 1 reaches alone, and serve each model type as its own container behind the shared `/predict` contract.

The north star is the share of complaints auto-routed at 90% precision or better. It was 95% until 2026-11-15. The target lives in `src/utils/config.py` as `NORTH_STAR_TARGET`, and the 95% operating point is still computed and saved for every model so older numbers stay comparable.

## Results (test set, 109,834 complaints)

All thresholds were fitted on validation and frozen before scoring test.

| | Accuracy | Macro-F1 | Auto-routed at the 90% threshold | Auto-routed at the 95% threshold |
|---|---:|---:|---:|---:|
| Improved Model 1 alone | 0.844 | 0.726 | 85.3% (at 89.8%) | 61.5% (at 94.8%) |
| Improved Model 2 alone | 0.814 | 0.665 | 76.6% (at 89.5%) | 44.1% (at 94.8%) |
| Gate (Model 1 + Model 2) | 0.844 | 0.726 | 87.4% (at 89.5%) | 66.4% (at 94.5%) |

The figure in brackets is the precision actually reached on test. The gate does not change the predicted team, so its accuracy is Improved Model 1's.

What the numbers say:

- The gate routes more than Improved Model 1 alone: 2.1 points more at the 90% threshold and 4.9 points more at the 95% threshold.
- Part of that is paid for in precision. The gate's test precision is about a quarter of a point lower at each threshold. Compared at exactly equal test precision, the gain is 1.4 points at 90% (85.9% against 84.5%) and 3.6 points at 95% (63.4% against 59.8%). Those matched figures read thresholds off the test set, so they are a diagnostic and not a reportable result.
- No model reaches its target on test. Every validation-fitted threshold lands 0.2 to 0.5 points short, for all three rows. Validation is July to September 2025 and test is October to December, so this looks like drift over time.
- Improved Model 2 is weaker than Improved Model 1 by 3 points of accuracy and 6 of macro-F1. It still helps as a second opinion: the two agree on 88.0% of test complaints, and Improved Model 1 is right 88.4% of the time when they agree and 54.5% when they disagree.
- In the disagreements, Improved Model 2 is right 30.1% of the time. One of the two models is right on 88.0% of complaints, against 84.4% for Improved Model 1 alone. That gap is the most a rule for choosing between them could add to accuracy.
- Moving to 90% breaks the per-team guardrail. At the 95% threshold no team was below the 0.85 routed-precision floor. At the 90% threshold three teams are below it in each setup:

| Team | Improved Model 1 alone | Gate |
|---|---:|---:|
| T04 Banking Accounts | 0.853 | 0.845 (below) |
| T05 Money Transfers & Digital Currency | 0.810 (below) | 0.818 (below) |
| T08 Vehicle Finance | 0.849 (below) | 0.871 |
| T09 Personal & Short-Term Loans | 0.780 (below) | 0.789 (below) |

Files: `docs/metrics/improved_model_2/`, `docs/metrics/ensemble_gate/`, and `reports/model_comparison.csv` (which now has 90% columns beside the 95% ones).

## Where we were before this work

Improved Model 1 auto-routes 61.5% of the test set at 94.78% precision (`docs/metrics/improved_model_1/north_star.json`). The threshold was fitted to 95% on validation and lands just under it on test.

Three later attempts did not beat it. FLAN-T5 + LoRA reached 0.740 accuracy on a 2,000-row validation sample and is slow to train. Jev reached 57.4% at 94.7% on a 2,000-row test sample, and adding it as a second opinion changed nothing because its confidence is 1.00 for most answers. The agentic issue classifier has only been scored on 20 rows.

One correction to an earlier suggestion of mine: the 568,050 complaints with identical text and conflicting labels are already excluded when the dataset is built (`data/processed/README.md`, step 5). There is no exact-duplicate cleaning left to do. What remains is complaints that mean the same thing but are worded differently and filed under different products, mostly Debt Collection against Credit Reporting. No relabelling step is planned for that.

## Evidence that a gate helps

Before building anything I checked whether a second model's agreement carries information that Improved Model 1's own confidence does not. The only second model with predictions on the full test set is Baseline 2 (Naive Bayes), so I used that.

On the 109,834 test complaints the two models agree 86.0% of the time. Improved Model 1 is right 88.7% of the time when they agree and 58.0% when they disagree.

I then split the test set in half by `leakage_group_id`, fitted the gate and its threshold on one half, and scored the other half. Improved Model 1 alone had its threshold refitted on the same half, so the comparison is like for like.

| Split seed | Improved Model 1 alone | Logistic gate | Gradient-boosted gate |
|---|---|---|---|
| 0 | 60.3% at 94.9% | 60.9% at 95.1% | 64.1% at 95.1% |
| 1 | 60.1% at 95.0% | 61.2% at 95.0% | 65.0% at 95.1% |

This is exploratory. It fits on test data, so none of these numbers can be reported as a result. What it shows is that a weak second model already moves auto-routing by about four points with the boosted gate. A stronger and more different second model should do better, which is the case for building Improved Model 2.

## Data rules

Every new model uses the data selection Improved Model 1 used.

- Training reads `data/cache/splits_v2/train_cap_issue.parquet` (257,694 rows, at most 10,000 per issue).
- Validation and test are `val.parquet` (153,502 rows) and `test.parquet` (109,834 rows), with the real team mix, split by date, and with no `leakage_group_id` on both sides of a boundary.
- `src/improved_model_2/train.py` refuses to run if its test complaint IDs differ from `docs/metrics/improved_model_1/predictions_test.csv`.
- Calibration, the gate, and every threshold are fitted on validation. Test is scored once with everything frozen.
- The gate's threshold is chosen on out-of-fold scores, grouped by `leakage_group_id`, so it is not picked on rows the gate was fitted to.
- The gate is trained and scored only on complaints that every model has a prediction for.

## Design

### Improved Model 2: sentence embeddings and logistic regression

`src/improved_model_2/`. A frozen sentence encoder (`BAAI/bge-small-en-v1.5` by default, set `IM2_ENCODER` to change it) turns each complaint into a vector. Logistic regression on those vectors predicts the team, with isotonic calibration on validation as in Improved Model 1.

Embedding is the slow step and happens once. Vectors are cached under `data/cache/improved_model_2/`, keyed by encoder, sequence length and training split, and the cache is reused only if the complaint IDs match exactly. After that, fitting the classifier is a CPU job.

It predicts the team only. Issue prediction stays with Improved Model 1 for now.

### Routing gate

`src/ensemble/`. The gate does not pick the team. Improved Model 1 does. The gate estimates the probability that Improved Model 1's team is correct, and that score is thresholded for auto-routing.

Its inputs are what each model service already returns under the contract: top team and confidence. From those it builds the primary's confidence, each secondary's confidence, whether each secondary agrees, the complaint's length, and which team the primary chose. That last one lets the gate learn that some teams need more evidence than others, which is what per-team thresholds were trying to do with too little data.

Two gate models are fitted, logistic regression and gradient boosting. The one with the higher out-of-fold validation coverage at the north-star threshold is kept. Test is not used to choose.

Because the gate needs only (team, confidence) per model, the same feature code runs offline on prediction files and online in the service. Any model that writes `data/cache/predictions/<model>/<split>.parquet` can be added as a secondary, including Jev or the seq2seq model.

### Services

Each model type gets its own container that implements the contract in `platform/shared`.

| Service | Serves | State |
|---|---|---|
| `svm-api` | Improved Model 1 | Existing, unchanged |
| `embed-api` | Improved Model 2 | New |
| `ensemble-api` | Calls `svm-api` and `embed-api`, applies the gate | New |
| `router` | Proxies to `PRIMARY_URL` | Unchanged |

`ensemble-api` returns the primary's team and issues, puts the gate score in `team_confidence`, and sets `auto_route` from the gate threshold. If the primary fails it returns 503, like the router does. If a secondary fails it still answers but sends the complaint to review, because the gate was not trained for a missing input.

To send traffic through the gate, point the router's `PRIMARY_URL` at `http://ensemble-api:8000`. No router code changes. Both new services sit behind the compose profile `ensemble`, so plain `docker compose up` is still the Phase 1 slice.

### Where this departs from the Phase 1 spec

The spec's Phase 3 names `seq2seq-api` and `agentic-api`, and section 8 lists ensemble routing as deferred. This plan adds `embed-api` and `ensemble-api` first instead, because they are the ones with evidence behind them. If the team wants to keep the spec's order, the two new services can stay in the `ensemble` profile and out of the default stack with no other change. This needs a decision from the team.

## Steps

### 1. Improved Model 2 (done)

- [x] `src/improved_model_2/`: config, encoder, model, train, predict
- [x] `tests/test_improved_model_2.py` (4 tests, fake encoder, no torch needed)
- [x] Full training run (30 minutes wall clock, `bge-small-en-v1.5`)
- [x] Test accuracy, macro-F1 and north star recorded next to Improved Model 1

### 2. Routing gate (done)

- [x] `src/ensemble/`: gate, prediction file layout, training script, exporter for Improved Model 1
- [x] `tests/test_ensemble_gate.py` (7 tests)
- [x] Improved Model 1's validation and test predictions exported
- [x] Gate trained (gradient boosting chosen on validation) and compared on test
- [x] North star moved to 90%: thresholds in `gate.joblib` (0.616) and Improved Model 2's `model.joblib` (0.602) now follow the 90% operating point

### 3. Model services (code done, images not yet built)

- [x] `platform/services/embed-api/` with Dockerfile and 8 tests
- [x] `platform/services/ensemble-api/` with Dockerfile and 10 tests
- [x] Compose profile `ensemble`, `make up-ensemble`, `.env.example` entries
- [ ] Build both images and run the stack once the artifacts exist
- [ ] Add an end-to-end smoke script for the ensemble profile

### 3b. Improve the ML-only routing before adding an LLM (implemented, not yet run)

Code: `src/ensemble/` (`train_stack.py`, `stack.py`, `thresholds.py`, `knn.py`, `probs.py`, `export_predictions.py`). Tests: `tests/test_ensemble_round2.py` (17 tests).

What each item became:

1. Per-team floors: after the global threshold is chosen, each team's threshold is raised until the 95% lower bound of its routed precision on validation reaches 0.85. Thresholds only go up. A team that cannot reach the floor is never auto-routed.
2. Safety margin: the global threshold has to reach the target plus a margin, using the 95% lower bound rather than the point estimate. Default margin 0.25 points.
3. Full probabilities: every model now exports a probability for each of the 11 teams. The gate sees, per model, the probability for the chosen team, its own top probability, the gap to its second choice, whether it agrees, and its entropy.
4. Extra opinions: Baseline 2 and a vote among the 25 most similar training complaints by embedding (`knn_embed`).
5. Stronger second model: the C grid now reaches 0.1, and `--tag` saves a variant (for example a larger encoder) next to the original so both can go into the stack. ModernBERT, or any other model, can join by writing a prediction file with `p_<team>` columns.
6. Team stacker: a logistic regression over every model's log-probabilities picks the team.

`train_stack.py` reports four setups under three threshold policies, so the effect of each change can be read off separately:

| Setup | Team from | Routing score from |
|---|---|---|
| `primary_alone` | Improved Model 1 | Improved Model 1's confidence (today) |
| `gate_on_primary` | Improved Model 1 | gate over every model's probabilities |
| `stack_alone` | stacker | stacker's probability |
| `stack_gate` | stacker | gate over stacker and every model |

Policies: `global` (one threshold), `margin` (adds the margin), `margin_floors` (adds the per-team floors). The served candidate is whichever of `gate_on_primary` and `stack_gate` routes more of validation under `margin_floors`. Test is not used to choose.

What is already known from Improved Model 1's real predictions (the other setups need the new exports):

| Improved Model 1 alone, test | Auto-routed | Precision | Teams below 0.85 |
|---|---:|---:|---|
| One threshold (today) | 85.3% | 89.8% | T05, T08, T09 |
| + 0.25-point margin | 83.9% | 90.1% | T05, T09 |
| + 0.25-point margin + floors | 78.2% | 91.1% | none |

- The 0.25-point margin is enough to reach 90% on test, at 1.4 points of coverage.
- The floors cost another 5.7 points. Most of it is Debt Collection (16,116 test complaints), which is below 0.85 on validation at the global threshold and gets its threshold raised to 0.79. Debt & Credit Management never reaches the floor and would not be auto-routed at all.
- I first estimated the margin automatically from drift inside validation: fit on July to 24 August, check on 25 August to September. That drop was about 3 points, so the margin hit its 2-point cap and test precision overshot to 91.9% at 76.9% coverage. The drift from validation to test was only 0.2 to 0.5 points, so the within-validation estimate is not a good guide here. It is still available as `--margin auto`.
- The 0.25-point default was chosen after seeing the first gate's test shortfall, so test numbers with it are not fully clean. The 2026 holdout (January to February 2026) has not been looked at, which is why the commands below score it too.

### Issue classification: where it stands

From Improved Model 1's test predictions:

- Top-1 accuracy is 56.1% and macro-F1 0.367. Top-3 accuracy is 81.3%.
- When the team is right, the true issue is first 66.6% of the time and in the top three 96.3% of the time. The issue heads rank well; the first choice is the weak part.
- 36% of issue errors come from a wrong team, so a better team choice helps issues directly.
- 15 of the 71 issues have F1 below 0.2. They are mostly small (30 to 650 test complaints), for example T03_I010 (271 complaints, F1 0.00) and T11_I003 (109, F1 0.00).
- Accuracy by team when the team is right ranges from 54% (T11) to 76% (T06).
- The agentic issue classifier has only been run on 20 validation rows (accuracy 0.50). Its verify step never ran, because the Groq backend reports confidence 1.0 whenever the answer parses, so the 0.6 trigger is never crossed.

What was built (`src/ensemble/train_issue.py`):

- Two new issue models on the cached embeddings: a logistic regression per team (`embed_issue_head`) and the issue vote among the 25 nearest training complaints (`knn_embed`).
- Improved Model 1's issue heads now export the full probability for every issue, in one batched pass per team. A test checks the batched numbers equal the original row-by-row ones.
- The three are pooled with weights and a prior correction chosen on validation by issue macro-F1 with the true team, then ranked inside the chosen team.
- The report separates the gain from a better team choice and from better issue models: each issue model alone, pooled, with the stacker's team, with Improved Model 1's team, and with the true team.

Not done for issues yet: the LLM step for the hard cases (step 5 below), a joint team-and-issue decision, and anything specific to the 15 weak issues (merging near-identical labels or adding training data for them).

### 3c. Commands for round 2

From the repository root, in the project environment, in this order:

```powershell
# 1. Every model's probabilities for validation, test and the 2026 holdout.
#    The holdout has no cached embeddings, so it is encoded once (GPU helps).
python -m src.ensemble.export_predictions --splits val test holdout_2026 --device cuda

# 2. Team stacker, gate, margin and floors. Prints all setups and policies.
python -m src.ensemble.train_stack --eval-splits test holdout_2026

# 3. Issue pooling on the chosen team.
python -m src.ensemble.train_issue --eval-splits test holdout_2026

# Optional: a larger encoder as an extra opinion, then rerun steps 2 and 3
# with it added to --models.
python -m src.improved_model_2.train --encoder BAAI/bge-base-en-v1.5 --tag base --device cuda
python -m src.ensemble.export_predictions --models improved_model_2_base --splits val test holdout_2026 --device cuda
python -m src.ensemble.train_stack --models improved_model_1 improved_model_2 improved_model_2_base baseline_model_2 knn_embed --eval-splits test holdout_2026
```

Outputs: `docs/metrics/ensemble_stack/` (`comparison.csv` holds every setup and policy) and `docs/metrics/ensemble_issue/` (`comparison.csv`). Both append rows to `reports/model_comparison.csv`.

On a 2-CPU machine with stand-in probabilities, step 2 took about 2 minutes and step 3 about 1 minute at full size.

Serving of this setup is described in 3f.

### 3e. Round 2 results (run 2026-11-16)

All thresholds fitted on validation and frozen. Test is Oct to Dec 2025 (109,834 complaints). The 2026 holdout is Jan to Feb 2026 (29,798).

Team routing on test, with the 0.25-point margin and per-team floors:

| Setup | Accuracy | Macro-F1 | Auto-routed | Precision | Teams below 0.85 |
|---|---:|---:|---:|---:|---|
| Improved Model 1 alone | 0.844 | 0.726 | 78.2% | 91.1% | none |
| Gate on Improved Model 1 | 0.844 | 0.726 | 85.3% | 90.5% | T04 (0.849) |
| Stacker alone | 0.855 | 0.740 | 85.0% | 90.6% | none |
| Stacker + gate (chosen on validation) | 0.855 | 0.740 | 86.9% | 90.3% | T04 (0.849) |

At the 95% target the chosen setup routes 70.8% at 94.6%, against 61.5% at 94.8% for Improved Model 1's original threshold.

What each model adds to the stacker on test (3-fold, stacker only, no gate):

| Models in the stacker | Accuracy | Auto-routed at 90% with floors |
|---|---:|---:|
| Improved Model 1 | 0.843 | 80.8% |
| + Improved Model 2 | 0.851 | 82.9% |
| + Baseline 2 | 0.854 | 83.7% |
| + nearest-neighbour vote instead | 0.853 | 84.4% |
| All four | 0.855 | 84.9% |

Every model earns its place. Baseline 2 is a small TF-IDF model and can run inside `svm-api`. The nearest-neighbour vote needs the encoder and the training embeddings, so it belongs inside `embed-api`. No extra containers are needed.

#### The 2026 holdout is a different team mix

On the holdout every setup does much worse: the chosen one routes 78.1% at 82.9%. The cause is the team mix, not the models.

- Credit Reporting is 1.9% of the holdout, against 60.6% of validation and 52.7% of test. Debt Collection rises to 29.6%.
- The models are calibrated to the old mix, so they send far too much to Credit Reporting: 3,870 holdout complaints are predicted T01 and only 552 are. Routed-to-T01 precision is 17.9%.
- Leave those out and the rest of the routed holdout is 92.3% precise. Accuracy on complaints that are not Credit Reporting is higher on the holdout (0.767) than on test (0.751).

The holdout construction notes say 1,004,032 of the 1,038,383 rows in the Jan to Feb 2026 CFPB export had no usable narrative or labels. Whether Credit Reporting really fell that far, or most of its complaints simply lack published narratives in that export, needs checking against the raw export before the holdout is used to judge models.

A standard fix exists and works here. Estimating the new team mix from the model's own predictions, with no labels (the EM method of Saerens et al., 2002), recovered it closely (T01 estimated at 0.3%, true 1.9%; every other team within a point). Reweighting the stacker's probabilities with that estimate:

| Stacker, holdout | Accuracy | Auto-routed at 0.619 | Precision |
|---|---:|---:|---:|
| As trained | 0.771 | 80.3% | 83.5% |
| Prior re-estimated (no labels) | 0.834 | 85.3% | 89.4% |

On test, where the mix barely moved, the same step changes accuracy from 0.8548 to 0.8552. In serving this would run on a rolling window of recent predictions.

#### Issues

| Issue, test | Top-1 accuracy | Macro-F1 | Top-3 |
|---|---:|---:|---:|
| Today (Improved Model 1's heads and team) | 0.561 | 0.367 | 0.813 |
| Improved Model 1's heads, stacker's team | 0.570 | 0.363 | 0.824 |
| Pooled heads, stacker's team | 0.560 | 0.384 | 0.823 |
| Pooled heads, true team | 0.643 | 0.540 | 0.952 |

- Pooling was tuned to weight the embedding heads at 0.25 and the nearest-neighbour vote at 0. It raises macro-F1 by 1.7 points and costs a point of top-1 accuracy, mostly through the prior correction. Most of the accuracy gain comes from the better team.
- The embedding issue heads are much weaker than the SVM heads (0.47 top-1 with the true team, against 0.65), so more ML tuning here is unlikely to pay.
- With the true team, the right issue is in the top three 95% of the time but first only 64%. Picking among three candidates is the job for the LLM step.

#### Decisions taken from these results

1. Serve the stacker + gate (the validation choice). Done in 3f.
2. Add the label-shift correction before anything else. Done in 3f; with it, T04 clears the floor on test.
3. Check the Jan to Feb 2026 export's raw product mix before relying on the holdout.
4. Stop ML work on issues. Serve Improved Model 1's issue heads with the stacker's team (best top-1 and top-3), and move issue improvement to the LLM top-3 step.

### 3f. Round 3: label-shift correction and serving the chosen setup (implemented 2026-11-16)

#### Label-shift correction, measured

`src/ensemble/label_shift.py` re-estimates the team mix from the stacker's own outputs with no labels (EM, Saerens et al. 2002) and reweights its probabilities. `train_stack.py` now scores every stacker setup both ways on the eval splits (`corrected` column in `comparison.csv`). Stacker + gate, margin and floors, 3-fold run:

| Split | As trained | Corrected |
|---|---:|---:|
| Test (Oct to Dec 2025): auto-routed at precision | 86.6% at 90.4% | 86.3% at 90.7% |
| Test: accuracy | 0.855 | 0.855 |
| Test: teams below 0.85 | T04 | none |
| Holdout (Jan to Feb 2026): auto-routed at precision | 77.7% at 82.8% | 79.7% at 91.3% |
| Holdout: accuracy | 0.771 | 0.834 |
| Holdout: teams below 0.85 | T01, T09 | T09 |

The estimated holdout prior puts Credit Reporting at 0.3% (true 1.9%), with every other team within a point of the truth. On test, where the mix barely moved, the correction costs 0.3 points of coverage and buys 0.3 points of precision. It is on by default in serving.

This run used 3 folds for the gate (the machine here is small); the earlier 5-fold run gave 86.9% at 90.3% on test for the same setup. `docs/metrics/ensemble_stack/` now holds the 3-fold artifacts, because the serving bundle needs two keys the first run did not save (`source_prior`, `primary_fallback_policy`). One rerun of `python -m src.ensemble.train_stack --eval-splits test holdout_2026` restores 5 folds.

#### Serving

The contract gained optional fields only, so Phase 1 clients and services keep working: `team_probs`, `extra_team_probs` (models co-hosted in the same container), `scenario`, `stack_team_probs`, plus a `POST /issues` endpoint.

| Service | Change |
|---|---|
| `svm-api` | Returns all 11 team probabilities; co-hosts `baseline_model_2` (16 MB, same TF-IDF stack) under `extra_team_probs`; new `POST /issues {text, team}` ranks issues inside any team |
| `embed-api` | Returns all 11 probabilities; co-hosts the `knn_embed` vote from a mounted index (`data/cache/knn_embed/index.npz`, written by `export_predictions`, float16, about 200 MB); refuses to start if the index was built with a different encoder |
| `ensemble-api` | Loads `stack.joblib`; stacker, rolling label-shift prior, gate, per-team thresholds; `scenario` in `auto`, `assisted`, `contested`, `degraded`; `GET /prior`, `POST /prior/reset`; eleven new Prometheus series |

Decisions taken from the system's point of view:

- One call per container. Each model service returns its own and its co-hosted model's probabilities, so a request costs two parallel calls, not four, and no new containers were added for Baseline 2 or the neighbour vote.
- Issues follow the chosen team. When the stacker overrides svm-api's team (about 1 in 10 requests) ensemble-api makes one more call to `/issues` for that team. If that call fails the issue list is empty rather than the wrong team's issues.
- Degrade to something validated, not to review-everything. If embed-api is down, the primary-alone policy with its own margin and floors applies (78.2% at 91.1% on test) and the response says `degraded`. If svm-api is down the answer is 503, as the router already does.
- The live prior is bounded (factor 50 per team), needs 500 observations before it activates, refreshes every 100 requests over a 5,000-request window, and can be inspected and reset over HTTP. `LABEL_SHIFT_ENABLED=false` serves the stacker exactly as evaluated.
- Fail at startup, not at request time: wrong encoder for the index, a bundle whose scenario needs a stacker it lacks.

Tests: 13 ensemble-api, 11 embed-api, 12 svm-api (these load the real Improved Model 1 artifact and check parity with `src.improved_model_1.predict`), 12 shared, 20 in `tests/test_ensemble_round2.py`. `make smoke-ensemble` runs one request through router, ensemble-api, both model services, `/issues` and `/prior` in Docker; it has not been run here because this machine has no Docker.

Not done in this round: shadow logging to the Postgres `predictions` table (Phase 4), the check of the raw Jan to Feb 2026 export, and the LLM step for issues and contested teams.

### 3d. (moved) Earlier list of ideas

The list below was the plan for 3b before it was built. It is kept for the reasoning behind each item.

In rough order of expected value for the effort:

1. Fix the per-team guardrail. Give the gate a per-team floor: a complaint is auto-routed only if the gate score also clears a threshold for its predicted team, fitted on validation, for the teams that fall below 0.85 (T04, T05, T08, T09). This trades some coverage for those teams' precision.
2. Add a safety margin to the threshold. Fit it to a validation precision slightly above the target (for example the lower end of the validation confidence interval) so that 90% is actually met on test after drift.
3. Give the gate each model's full probability vector. Today it sees only the top team and its confidence. The gap between the first and second team, and the second model's probability for the primary's team, are both available offline already. Online this needs the `/predict` response to carry the top few teams, which is a contract change.
4. Add Baseline 2 as a third opinion and a nearest-neighbour vote from the cached embeddings.
5. Make the second model stronger: a larger encoder (one flag), or the fine-tuned ModernBERT run in place of frozen embeddings. The logistic regression's best `C` was the smallest in the grid (0.5), so smaller values should be tried too.
6. Let the stack choose the team, not only the routing. A small model over both models' probabilities could recover part of the 3.6-point gap between Improved Model 1 (84.4%) and "either model is right" (88.0%).

### 4. Retrieval and reranking for the review queue (planned)

Complaints the gate sends to review are where an LLM is worth its cost. For those only: retrieve similar labelled training complaints with BM25 plus the cached embeddings, rerank the top 50 with a cross-encoder, and let the LLM choose among Improved Model 1's top three teams with those examples and the structured team descriptions from the Jev work. Auto-route only when the LLM, Improved Model 1 and the retrieved neighbours agree.

The share of retrieved neighbours carrying each team label is also a cheap gate feature that needs no LLM. Try that first.

Accept this step only if the complaints it rescues are themselves at least as precise on validation as the north-star target.

### 5. Issue classification (planned)

Issue macro-F1 is 0.367 with the predicted team. Start with a nearest-neighbour vote inside the predicted team using the cached embeddings, with no LLM, to get a number to beat. Then have the LLM choose among the top five candidate issues. Fix the verify step in `src/agentic_issue_classifier/` first (it fired on 0% of rows, because the Groq backend reports confidence 1.0 whenever the answer parses). Evaluate on at least 500 rows and report top-3 accuracy, since the app shows three issues.

### 6. Remaining services (planned)

`agentic-api` wraps step 4 or 5 once one of them passes its bar. `seq2seq-api` is only worth building if the FLAN-T5 model ends up adding something as a gate secondary. Its validation predictions can be tested for that offline before any container work.

## Commands to run

From the repository root, in the project virtual environment. `sentence-transformers` is already in `requirements-train.txt`.

```powershell
# 1. Quick end-to-end check on 2,000 rows per split. Writes nothing.
python -m src.improved_model_2.train --smoke 2000

# 2. Full run. Use --device cuda if a GPU is available.
python -m src.improved_model_2.train

# 3. Improved Model 1 predictions on validation and test, in the gate's layout.
python -m src.ensemble.export_predictions --model improved_model_1

# 4. Fit the gate on validation, score once on test.
python -m src.ensemble.train_gate --primary improved_model_1 --secondary improved_model_2

# 5. Tests
python -m pytest -q tests/test_improved_model_2.py tests/test_ensemble_gate.py
```

Step 4 prints the gate against Improved Model 1 alone and writes `docs/metrics/ensemble_gate/`. Both training scripts append a row to `reports/model_comparison.csv`.

Then, to serve it:

```powershell
cd platform
$env:PRIMARY_URL = "http://ensemble-api:8000"
docker compose --profile ensemble up -d --build
```

## How we will know it worked

1. The gate's test coverage at its validation-fitted threshold is above Improved Model 1's, with test precision no lower. Coverage: met (87.4% against 85.3%). Precision: not met (89.5% against 89.8%).
2. No team falls below the 0.85 routed-precision floor. Not met at the 90% threshold, for either setup.
3. `make up-ensemble` brings up five healthy containers and a `/predict` call through the router returns `model_name: ensemble_gate_v1`.
4. The existing test suite and the Phase 1 smoke test still pass.

The gate is ahead on coverage and slightly behind on precision, so it is not yet a clear replacement. The services stay opt-in until criteria 1 and 2 are both met.

## Open questions

- Whether the 0.85 per-team floor still applies now that the overall target is 90%. With Improved Model 1 alone it costs 5.7 points of coverage and switches off auto-routing for Debt & Credit Management.
- Whether to treat the 2026 holdout as the clean test set from now on, since the margin was set after looking at test.
- The served Improved Model 1 (`svm-api` and the Streamlit app) still auto-routes at its 95% threshold of 0.818. Moving it to the 90% threshold of 0.604 changes live behaviour and means editing its saved artifact, so it has been left alone.
- Whether to pull ensemble routing ahead of the spec's Phase 3 order (see above).
- Which encoder to settle on. `bge-small` is the default because it is fast on CPU. A larger encoder may be more accurate and is a one-flag change, at the cost of a slower container.
- Whether Improved Model 1's calibration should be refitted on half of validation so the gate sees confidences that were not calibrated on its own training rows. With 153,502 validation rows the effect should be small, but it has not been measured.
