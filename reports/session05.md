---
team: ClariCase
session: 05
date: 2026-09-29
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
  value: 46.5%
  previous: 26.4%
---

## Shipped this week
- **Routing dataset v1.0** (`data/processed/`). 1,044,615 complaints received 2024-01-01 to 2025-12-31 across 3,801 companies, spanning 73 product-specific issue labels that map deterministically to 11 support teams. Construction rules and archive hashes are recorded in `dataset_metadata.json`; all ten checks in `validation.json` pass.
- **Baseline 2, the team routing model** (`src/baseline_model_2/`). TF-IDF into a calibrated Multinomial Naive Bayes over `complaint_text`, predicting one of 11 `team_id` values. Metrics, calibration curve, per-team routing table and test predictions are committed to `docs/metrics/baseline_model_2/`, with the write-up in `docs/baseline_model_2.md`.
- **Leakage-aware temporal evaluation.** Training covers 2024-01-01 to 2025-06-30 at 3,000 records per team (32,599); validation is July–September 2025 and test is October–December 2025, both at the true team distribution (153,502 and 109,834). Leakage groups spanning a period boundary were removed entirely, 6,867 records across 1,949 groups.
- **Complaint intake application** (`streamlit_app.py`, `src/app/`). Consumers describe a problem in free text and receive a routed team and a tracking identifier; the product and issue dropdowns are gone. Storage runs on Supabase with a local SQLite fallback.
- **Repository consolidated.** `src/models/baseline.py` and `docs/metrics/baseline/` became `src/baseline_model_1/` and `docs/metrics/baseline_model_1/`, separating the two baselines and their artifacts.
- **Per-bank EDA branches aligned** to the new structure. `datawellsfargo/eda` and `databofa/eda` now mirror main's layout with bank-prefixed paths. Both remain unmerged.

## What changed since session 04
- **Corpus:** 22,465 JPMorgan complaints to 1,044,615 across 3,801 companies, a 46X increase. The company-confound risk raised last session is now testable.
- **Task:** predicting a consumer's Product selection became routing to one of 11 support teams, with the 73 issue labels reserved for a later model and team resolved by deterministic lookup.
- **Evaluation:** a 6,382-row single-bank test set became 109,834 rows at the real team mix, with a separate validation quarter reserved for fitting thresholds.
- **Calibration now helps rather than costs.** Last session, wrapping the SVM cost 0.005 macro-F1; isotonic calibration this session gains 0.074 and reduces ECE from 0.0664 to 0.0259.
- **The project has a working product for the first time.**

## User evidence
- The application is built and running locally. It has not yet been placed in front of a user, so no external evidence was gathered this session.
- **Raw artifact**: `streamlit_app.py`, `src/app/storage.py`, `docs/baseline_model_2.md`

## Metrics snapshot
- **Autonomous routing rate @95% precision: 46.5%, up from 26.4%.** 51,034 of 109,834 test complaints routed without a human at a 0.872 confidence threshold.
- Realised precision at that threshold is 94.44% (95% CI 94.24–94.65), narrowly short of the 95% definition. Thresholds were fitted on validation, where the same cut gave 48.18% coverage, and frozen for test. The other operating points show the same pattern: 89.68% against a 90% target, 84.31% against 85%.
- The model was evaluated under two protocols. The production figure uses the real team mix, which is what an operations queue receives. The balanced figure samples evenly across teams, isolating classifier ability from traffic skew.

| Test protocol | n | Accuracy | Macro-F1 | Weighted-F1 | Dummy accuracy | Dummy macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Real team mix, Oct–Dec 2025 | 109,834 | 0.800 | 0.647 | 0.793 | 0.527 | 0.063 |
| Balanced, ~600 per team | 6,601 | 0.730 | **0.732** | 0.729 | 0.085 | 0.014 |

- Read together, the two protocols separate ability from skew. On real traffic accuracy reaches 0.800, but 53% of complaints are Credit Reporting and the dummy alone scores 0.527. On the balanced set the dummy collapses to 0.085 while the model holds 0.732 macro-F1. The model is reading the narrative, not the prior.

| Team | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T07 Student Loan | 0.898 | 0.881 | 0.889 | 570 |
| T06 Mortgage | 0.825 | 0.896 | 0.859 | 565 |
| T10 Prepaid Card | 0.811 | 0.782 | 0.796 | 559 |
| T08 Vehicle Finance | 0.807 | 0.737 | 0.770 | 585 |
| T05 Money Transfers & Digital Currency | 0.746 | 0.726 | 0.736 | 649 |
| T01 Credit Reporting | 0.667 | 0.780 | 0.719 | 619 |
| T09 Personal & Short-Term Loans | 0.693 | 0.726 | 0.709 | 632 |
| T03 Credit Card | 0.715 | 0.633 | 0.672 | 603 |
| T04 Banking Accounts | 0.578 | 0.752 | 0.653 | 614 |
| T02 Debt Collection | 0.654 | 0.631 | 0.642 | 613 |
| T11 Debt & Credit Management | 0.731 | 0.510 | 0.601 | 592 |

- Per-class performance tracks how distinctive each team's vocabulary is. Specialist teams lead: Student Loan 0.889, Mortgage 0.859. Overlapping teams trail: Debt & Credit Management 0.601 at 0.510 recall, Debt Collection 0.642, Banking Accounts 0.653 at 0.578 precision. Credit Reporting, the largest class by a wide margin, reaches only 0.719, so volume does not buy separability.
- Calibration improves every figure. Uncalibrated test accuracy is 0.759 and macro-F1 0.574; isotonic calibration raises these to 0.800 and 0.647 and reduces ECE from 0.0664 to 0.0259.
- Per-team precision at the routing threshold spans 0.9905 (Money Transfers) to 0.8230 (Personal & Short-Term Loans). Ten of 11 teams clear the 0.85 floor.
- Auto-routed share is uneven across teams: Credit Reporting 67.1%, Mortgage 56.9%, Money Transfers 15.7%, Debt Collection 17.5%. The headline 46.5% is carried largely by the biggest class.
- Accuracy is the wrong headline on real traffic. Credit Reporting is 61.8% of the corpus and 52.7% of the test window, so a constant prediction scores 0.527. Macro-F1 is the reliable measure under both protocols.
- Is this the same model that is running in the product? Yes. `streamlit_app.py` loads `src/baseline_model_2/predict.py`, the artifact these metrics were computed from.

## What did not work
- The precision target was narrowly missed. Coverage rose from 26.4% to 46.5%, but at 94.44% precision against a 95% definition. The operating point is sound; the guarantee needs a small additional margin before it can be stated as met.
- One team sits just below the precision guardrail. Personal & Short-Term Loans routes at 0.823 against a 0.85 floor. It is the smallest routed group, 305 of 1,640 test complaints, so the shortfall is contained and the guardrail correctly flagged it.
- Overlapping teams remain the weak point. Debt & Credit Management recalls 51.0% and Credit Card 63.3% on the balanced set, both losing complaints to Debt Collection and Credit Reporting, which share dispute and credit-repair vocabulary. This is where the next improvement lies.
- The model is trained on a fraction of the available data, 32,599 of 1,044,615 records under a 3,000-per-team cap. The cap protects the smaller teams, and whether the larger classes benefit from more data is still to be tested.
- Label ambiguity in the source data is now quantified. Dataset construction set aside 568,050 narratives whose identical text carried conflicting product-issue labels, alongside 415,097 duplicates. Last session recorded label quality as unmeasured; this is the measurement, and it puts a ceiling on what any model trained on consumer-selected categories can achieve.

## Challenges / blockers
- Hosting and user testing are the immediate next steps. The app came together late in the week, so it runs locally and has not yet been in front of anyone outside the team. A public URL is the remaining piece.
- Thresholds will need periodic refitting. Fitting on validation and freezing for test is the right protocol and holds well within a quarter, but the mix shift between quarters suggests either refreshing them on a schedule or carrying a slightly wider margin.
- The full dataset sits outside the repository. At 1.91 GB it is impractical to commit, so retraining currently requires fetching the file separately.

## Next week's goal


## Individual contributions
- **Sukriti Srivastava (Product)** — Designed and shipped the consumer-facing product. Built a clean Streamlit intake interface that accepts a complaint in plain language, routes it to the responsible team and returns a tracking ID, removing the product and issue dropdowns entirely. Implemented the complaint storage layer behind it (`src/app/storage.py`, `schema.sql`, Supabase with a local SQLite fallback) and rebuilt the served model so the running app loads the same artifact the reported metrics were computed from. (PR #18)
- **Chaitanya Bagul (Engineering)** — Built the routing layer and Baseline 2, the calibrated team classifier: TF-IDF into Multinomial Naive Bayes over `complaint_text`, isotonic calibration, a leakage-aware temporal split evaluated at the real team mix, and per-team confidence thresholds with a precision guardrail. Produced the evaluation harness and committed the full artifact set to `docs/metrics/baseline_model_2/` with the accompanying write-up. Moved the north star from 26.4% to 46.5%. (PR #16, #17)
- **Salman Farcy (Data and Evaluation)** — Owned evaluation and integration for the week. Reviewed, merged and approved every branch that landed, and brought the repository onto a single consistent structure so that datasets, notebooks, write-ups and metrics sit in predictable locations across all four contributors' work. Evaluated Baseline 2 against its committed artifacts establishing the majority-class floors the model is measured against, and identifying the Personal & Short-Term Loans precision guardrail breach and the quarter-over-quarter threshold drift that causes every operating point to undershoot its validation estimate. Authored this report.
- **Sriramm S S (Users and Research)** — Built the dataset the entire project now runs on. Worked through the full CFPB narrative archive across every company, product category and issue type to assemble a clean, deduplicated corpus of 1,044,615 complaints spanning 3,801 companies, 73 product-specific issue labels and 11 support teams. Defined and documented the construction rules, excluded narratives whose identical text carried conflicting labels, assigned leakage groups so near-duplicate templates cannot straddle a split, and shipped it with archive hashes, reconciliation counts and ten passing validation checks. Every model and every metric in this report is trained and measured on that corpus.

## Lean canvas changes (if any)
- The archive is a closed set. The CFPB stopped publishing narratives, and this dataset is built entirely from the 21 archived export files. The corpus cannot grow from this source, only be re-cut. Any live product will eventually run on inputs drawn from a different distribution than it was trained on.
- Scope widened from one bank to an industry corpus. The product is no longer "route complaints for a bank" but "route complaints across 3,801 companies into 11 support functions", which changes who the buyer is.
- The consumer is now a user, not just a data source. The app removes the product and issue dropdowns and replaces them with free text plus a tracking ID, so the value proposition now includes the person filing the complaint, not only the ops team reading it.
