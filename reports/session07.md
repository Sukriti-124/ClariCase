---
team: ClariCase
session: 07
date: 2026-10-20
members:
  - name: Sukriti Srivastava
    github: Sukriti-124
    hat: Product
  - name: Chaitanya Bagul
    github: elderflamevandle
    hat: Engineering
  - name: Salman Farcy
    github: farcypeer46
    hat:  Data&Eval 
  - name: Sriramm S S 
    github: SriRammSS
    hat: Users&Research 
north_star:
  metric: Autonomous routing rate at >=95% precision
  value: 61.5%
  previous: 46.5%
---

## Shipped this week
- **Improved Model 1, the team-and-issue classifier** (`src/improved_model_1/`). A hierarchical TF-IDF (word 1–2 grams plus character 3–5 grams) into Linear SVMs: one classifier picks the team, then a per-team classifier picks the issue within it. Isotonic calibration and the routing threshold are fit on validation and frozen for test. Artifacts are committed to `docs/metrics/improved_model_1/`, with the write-up in `docs/improved_model_1.md`.
- **Shared evaluation library and tests** (`src/utils/`, `tests/`). Parquet split builder, unified metrics for team, issue, joint and oracle evaluation, the north-star harness, label mappings and a results registry (`reports/model_comparison.csv`). 37 tests, including a parity check that the test set is the identical 109,834 complaints Baseline 2 was scored on.
- **Forward-drift cohort.** 29,798 complaints from January–February 2026, packaged separately from the 2024–2025 corpus, give a second evaluation window after the test quarter.
- **The app now runs Improved Model 1** (`streamlit_app.py`). Each complaint shows a **Routed** or **Under review** badge, the team (top 2 or 3 teams with percentages when confidence is below 80%), the top 3 likely issues, and a tracking ID. A three-step progress display replaces the generic loading indicator. The Track page shows status, team and most likely issue. Live at [ClariCase](https://claricase-enmuypbz78oqjgiec98bhw.streamlit.app/).
- **Storage records the routing decision** (`src/app/storage.py`, `src/app/schema.sql`). Each complaint now stores its top issue and route, and its status is set to Routed or Under review. Existing rows remain valid.


## What changed since session 05
- **Model:** A single Naive Bayes team classifier became a hierarchical Linear SVM that predicts the team and the issue. Macro-F1 rose from 0.647 to 0.726 on the identical test set.
- **Training data:** 32,599 records capped at 3,000 per team became 257,694 records capped at 10,000 per issue, so rare issues get full representation.
- **Task:** issue prediction, reserved for later in session 05, is now live. Every complaint arrives with a team and its three most likely issues.
- **Routing:** The served route uses one global threshold (0.818), the same operating point the north star reports. Per-team thresholds were evaluated and not adopted (see What did not work).
- **Product:** The app now shows confidence, alternatives and a review path, instead of a single team for every complaint.

## User evidence
- **TODO (team): add any user testing done this session**, or state plainly that none took place.
- **Raw artifact**: `streamlit_app.py`, `docs/improved_model_1.md`, `docs/metrics/improved_model_1/`, `reports/model_comparison.csv`

## Metrics snapshot
- **Autonomous routing rate @95% precision: 61.5%, up from 46.5%.** 67,550 of 109,834 test complaints routed without a human at a 0.818 confidence threshold (95% CI 61.2–61.8%).
- Realised precision at that threshold is 94.78% (95% CI 94.61–94.95), narrowly short of the 95% definition again.
- The threshold was fitted on validation, where it gave 63.9% coverage, and frozen for test. The other operating points show the same pattern: 89.76% against a 90% target, 84.36% against 85%.
- All figures below are on the test set at the real team mix (October–December 2025).

| Model | Training data | Macro-F1 | Weighted-F1 | Accuracy |
|---|---|---:|---:|---:|
| **Improved Model 1** (hierarchical Linear SVM) | 257,694, capped per issue | **0.726** | **0.840** | **0.844** |
| Naive Bayes, Baseline 2 recipe | 257,694, capped per issue | 0.668 | — | — |
| Flat Linear SVM over all 73 issues | 257,694, capped per issue | 0.631 | — | — |
| Baseline 2 (Naive Bayes) | 32,599, capped per team | 0.647 | 0.793 | 0.800 |

- More data alone moved macro-F1 from 0.647 to 0.668. The model change added another 0.058, to 0.726. Most of the gain came from the model.
- Deciding in two steps beat one flat classifier: team macro-F1 0.726 vs 0.631, and issue macro-F1 0.367 vs 0.312.
- Calibration stays sound: ECE 0.026 on test, the same as Baseline 2.

| Team | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T01 Credit Reporting | 0.884 | 0.932 | 0.907 | 57,889 |
| T06 Mortgage | 0.868 | 0.927 | 0.896 | 3,367 |
| T04 Banking Accounts | 0.773 | 0.829 | 0.800 | 10,164 |
| T07 Student Loan | 0.813 | 0.766 | 0.789 | 1,517 |
| T08 Vehicle Finance | 0.765 | 0.771 | 0.768 | 2,857 |
| T03 Credit Card | 0.822 | 0.716 | 0.766 | 9,725 |
| T02 Debt Collection | 0.809 | 0.713 | 0.758 | 16,116 |
| T05 Money Transfers & Digital Currency | 0.741 | 0.725 | 0.733 | 5,236 |
| T09 Personal & Short-Term Loans | 0.678 | 0.585 | 0.628 | 1,640 |
| T10 Prepaid Card | 0.755 | 0.539 | 0.628 | 856 |
| T11 Debt & Credit Management | 0.748 | 0.197 | 0.312 | 467 |

- Every team improved on F1 except Debt & Credit Management, which is flat (0.293 to 0.312).
- **The guardrail now passes for all 11 teams.** Personal & Short-Term Loans, below the floor last session at 0.823, routes at 0.894. Per-team precision at the threshold ranges from 0.888 (Money Transfers) to 0.962 (Credit Reporting, Mortgage).
- The session 05 confusions shrank but remain:
  - Money Transfers recall rose from 43.3% to 72.5%; 21.2% still go to Banking Accounts.
  - Debt Collection recall rose from 58.4% to 71.3%; 23.4% still go to Credit Reporting.

| Issue prediction (73 issues) | Value |
|---|---:|
| Issue macro-F1, using the predicted team | 0.367 |
| Issue macro-F1, if the team were always right | 0.511 |
| Team **and** issue both correct | 56.1% |

- **Forward drift, January–February 2026 (29,798 complaints):** Team macro-F1 0.697 against Baseline 2's 0.606, accuracy 0.767, joint exact 51.0%. Performance drops about 0.03 macro-F1 from the test quarter. This cohort has now been used for evaluation, so it is a drift check rather than an untouched holdout.


## What did not work
- The precision target was narrowly missed again: 94.78% against 95%. The confidence interval's upper bound (94.95%) still falls short, so the gap is real, not noise.
- **Per-team thresholds made routing worse overall.** Fitting a separate 95% threshold for each team auto-routed 54.9% at 94.5% precision, against 61.5% at 94.8% for one global threshold. Debt Collection and Debt & Credit Management could not reach 95% on validation at any threshold, so they would never be auto-routed. The app serves the global threshold.
- **Debt & Credit Management is still the weakest team.** Recall is 19.7%: 33.8% of its complaints go to Credit Reporting and 24.8% to Debt Collection. It has the least data (467 test complaints), and its issues overlap with both teams.
- **Issue prediction is held back by team errors.** Issue macro-F1 is 0.367 with the predicted team and 0.511 when the team is correct, so 0.144 is lost to team mistakes. Only 56.1% of complaints get both right.



## Challenges / blockers
- PII obfuscation The app currently has no protection for sensitive details.
- The forward-drift cohort is no longer fresh. A new untouched evaluation window will be needed to report future gains credibly.
- - Training is slow -  The full run with character n-grams took about 4.5 hours on a laptop GPU, which limits how many experiments fit in a week.

## Next week's goal
- **Mask PII before storage.** Detect and redact account numbers, Social Security numbers, names and addresses before a complaint is saved or shown.
- Evaluate the fine-tuned transformer. Finish fine-tuning Llama 3.2 3B Instruct and score it on the same test set and protocol as Improved Model 1
- **Close the team-to-issue gap.** Try a joint team-and-issue model or sentence embeddings (Improved Model 2) to recover some of the 0.144 issue F1 lost to team errors.
- **Explore Jev AI and agentic AI.** Jev AI, from TypeSafe AI, is a fast, low-cost model that returns structured probabilistic decisions (choices, scores or yes/no probabilities) rather than conversational text. Because that matches the router's output, a team choice with a confidence, test it on the same test set and 95%-precision rule as Improved Model 1, comparing accuracy, cost and speed. Separately, explore an agentic resolution guidance agent that suggests next steps once a complaint is routed.

## Individual contributions
- **Sukriti Srivastava (Product)** — Integrated Improved Model 1 into the live app and redesigned the intake experience: confidence-based display of one to three teams, the top three issues, Routed and Under review states, a step-by-step progress view, tab reset and a refreshed visual theme.  (PR #22) 
- **Chaitanya Bagul (Engineering)** — Built Improved Model 1, the hierarchical team-and-issue classifier, and the shared `src/utils/` library behind it: Parquet split builder with Baseline 2 parity, unified metrics, the north-star and per-team threshold harness, label mappings and the results registry. Wrote the 37-test suite, ran the hyperparameter search and the flat and Naive Bayes ablations, and documented the model. Moved the north star from 46.5% to 61.5%.
- **Salman Farcy (Data and Evaluation)** — Extended storage and the Supabase schema to record each complaint's issue and routing decision, and set served routing to the single global threshold so the product matches the reported north star.
- **Sriramm S S (Users and Research)** — Began fine-tuning Llama 3.2 3B Instruct for complaint routing with Unsloth, loading the base model in 4-bit and training LoRA adapters in fp16 (QLoRA).

## Lean canvas changes (if any)
