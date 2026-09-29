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
- **Baseline 2, the team routing model** (`src/baseline_model_2/`). TF-IDF into a calibrated Multinomial Naive Bayes over `complaint_text`, predicting one of 11 `team_id` values. Metrics, coverage curve, per-team routing table and test predictions are committed to `docs/metrics/baseline_model_2/`, with the write-up in `docs/baseline_model_2.md`.
- **Leakage-aware temporal evaluation.** Training covers 2024-01-01 to 2025-06-30 at 3,000 records per team (32,599); validation is July–September 2025 and test is October–December 2025, both at the true team distribution (153,502 and 109,834). Leakage groups spanning a period boundary were removed entirely, 6,867 records across 1,949 groups.
- **Complaint intake application** (`streamlit_app.py`, `src/app/`). Consumers describe a problem in free text and receive a routed team and a tracking identifier; the product and issue dropdowns are gone. Storage runs on Supabase with a local SQLite fallback. Live at [ClariCase](https://claricase-enmuypbz78oqjgiec98bhw.streamlit.app/).
- **Repository consolidated.** `src/models/baseline.py` and `docs/metrics/baseline/` became `src/baseline_model_1/` and `docs/metrics/baseline_model_1/`, separating the two baselines and their artifacts.
- **Per-bank EDA branches aligned** to the new structure. `datawellsfargo/eda` and `databofa/eda` now mirror main's layout with bank-prefixed paths. Both remain unmerged.

## What changed since session 04
- **Corpus:** 22,465 JPMorgan complaints to 1,044,615 across 3,801 companies, a 46× increase. The company-confound risk raised last session is now testable.
- **Task:** predicting a consumer's Product selection became routing to one of 11 support teams, with the 73 issue labels reserved for a later model and team resolved by deterministic lookup.
- **Evaluation:** a 6,382-row single-bank test set became 109,834 rows at the real team mix, with a separate validation quarter reserved for fitting thresholds.
- **Calibration now helps rather than costs.** Last session, wrapping the SVM cost 0.005 macro-F1; isotonic calibration this session gains 0.074 and reduces ECE from 0.0664 to 0.0259.
- **The project has a working product for the first time.**

## User evidence
- The application is deployed and publicly reachable at [ClariCase](https://claricase-enmuypbz78oqjgiec98bhw.streamlit.app/). It has not yet been placed in front of a user, so no external evidence was gathered this session.
- **Raw artifact**: `streamlit_app.py`, `src/app/storage.py`, `docs/baseline_model_2.md`

## Metrics snapshot
- **Autonomous routing rate @95% precision: 46.5%, up from 26.4%.** 51,034 of 109,834 test complaints routed without a human at a 0.872 confidence threshold.
- Realised precision at that threshold is 94.44% (95% CI 94.24–94.65), narrowly short of the 95% definition.
- Thresholds were fitted on validation, where the same cut gave 48.18% coverage, and frozen for test. The other operating points show the same pattern: 89.68% against a 90% target, 84.31% against 85%.
- All figures below are on the test set at the real team mix, which is what an operations queue receives.

| Model | n | Macro-F1 | Weighted-F1 | Accuracy |
|---|---:|---:|---:|---:|
| TF-IDF + Naive Bayes, isotonic calibration | 109,834 | **0.647** | 0.793 | 0.800 |
| Same model, uncalibrated | 109,834 | 0.574 | 0.768 | 0.759 |
| Majority-class dummy | 109,834 | 0.063 | 0.364 | 0.527 |

- Accuracy is the wrong headline on real traffic. Credit Reporting is 61.8% of the corpus and 52.7% of the test window, so a constant prediction already scores 0.527. Macro-F1 is the reliable measure.
- On macro-F1 the dummy scores 0.063 and the model 0.647. The model is reading the narrative, not the prior.
- Calibration improves every figure. Isotonic calibration raises accuracy from 0.759 to 0.800 and macro-F1 from 0.574 to 0.647, and reduces ECE from 0.0664 to 0.0259.

| Team | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T01 Credit Reporting | 0.865 | 0.931 | 0.897 | 57,889 |
| T06 Mortgage | 0.803 | 0.894 | 0.846 | 3,367 |
| T07 Student Loan | 0.735 | 0.764 | 0.749 | 1,517 |
| T04 Banking Accounts | 0.659 | 0.805 | 0.725 | 10,164 |
| T08 Vehicle Finance | 0.690 | 0.718 | 0.704 | 2,857 |
| T03 Credit Card | 0.710 | 0.664 | 0.686 | 9,725 |
| T02 Debt Collection | 0.813 | 0.584 | 0.680 | 16,116 |
| T05 Money Transfers & Digital Currency | 0.704 | 0.433 | 0.536 | 5,236 |
| T09 Personal & Short-Term Loans | 0.483 | 0.562 | 0.519 | 1,640 |
| T10 Prepaid Card | 0.506 | 0.468 | 0.486 | 856 |
| T11 Debt & Credit Management | 0.476 | 0.212 | 0.293 | 467 |

- Credit Reporting, the largest team, is also the best separated at 0.897 F1. Mortgage follows at 0.846.
- The three smallest teams trail: Debt & Credit Management 0.293 (recall 0.212), Prepaid Card 0.486 and Personal & Short-Term Loans 0.519. Each has under 2,000 test complaints, so these figures are also the noisiest.
- Per-team precision at the routing threshold spans 0.9905 (Money Transfers) to 0.8230 (Personal & Short-Term Loans). Ten of 11 teams clear the 0.85 floor.
- Auto-routed share is uneven across teams: Credit Reporting 67.1%, Mortgage 56.9%, Debt Collection 17.5%, Money Transfers 15.7%. The headline 46.5% is carried largely by the biggest team.
- Is this the same model that is running in the product? Yes. `streamlit_app.py` loads `docs/metrics/baseline_model_2/model.joblib` through `src/baseline_model_2/predict.py`, the artifact these metrics were computed from.

## What did not work
- The precision target was narrowly missed. Coverage rose from 26.4% to 46.5%, but at 94.44% precision against a 95% definition. The operating point is sound; the guarantee needs a small additional margin before it can be stated as met.
- One team sits just below the precision guardrail. Personal & Short-Term Loans routes at 0.823 against a 0.85 floor. It is the smallest routed group, 305 of 1,640 test complaints, so the shortfall is contained and the guardrail correctly flagged it.
- Overlapping teams remain the weak point, and the confusions are the same ones seen in session 04.
  - Money Transfers recalls only 43.3%. 45% of its complaints go to Banking Accounts, because a Zelle or wire dispute reads like a checking complaint.
  - Debt Collection recalls 58.4%. 31% of its complaints go to Credit Reporting, which shares its dispute and FCRA/FDCPA vocabulary.
  - Debt & Credit Management recalls 21.2%, losing 30% to Credit Reporting and 16% to Debt Collection.
  - This is where the next improvement lies.
- The model is trained on a fraction of the available data, 32,599 of 1,044,615 records under a 3,000-per-team cap. The cap protects the smaller teams, and whether the larger classes benefit from more data is still to be tested.
- Label ambiguity in the source data is now partly measured. Dataset construction set aside 568,050 narratives whose identical text carried conflicting product-issue labels, alongside 415,097 duplicates.
- Last session recorded label quality as unmeasured. This is a first, partial measurement: it shows how often identical text receives different labels, but not the error rate of the labels that were kept. Both point to a ceiling on what any model trained on consumer-selected categories can achieve.

## Challenges / blockers
- User testing is the immediate next step. The app is now public, but it came together late in the week and has not yet been in front of anyone outside the team.
- Thresholds will need periodic refitting. Fitting on validation and freezing for test is the right protocol and holds well within a quarter, but the mix shift between quarters suggests either refreshing them on a schedule or carrying a slightly wider margin.
- The full dataset sits outside the repository. At 1.91 GB it is impractical to commit, so retraining currently requires fetching the file separately.

## Next week's goal
- **Assign complaints to issues alongside teams.** Extend the model to predict the product-specific issue (73 labels) as well as the team, so each complaint arrives with both where it goes and what it is about.
- **Improve the UI.** Refine the intake and tracking screens based on the first round of use.
- **Prepare an end-to-end, user-friendly product for financial complaints.** Connect intake, routing, storage and tracking into one flow that a consumer can use from start to finish without guidance.
- **Obfuscate PII in sensitive information.** Detect and mask personal identifiers such as account numbers, Social Security numbers, names and addresses before complaints are stored or shown.

## Individual contributions
- **Sukriti Srivastava (Product)** — Designed and shipped the consumer-facing product. Built a clean Streamlit intake interface that accepts a complaint in plain language, routes it to the responsible team and returns a tracking ID, removing the product and issue dropdowns entirely. Implemented the complaint storage layer behind it (`src/app/storage.py`, `schema.sql`, Supabase with a local SQLite fallback) and rebuilt the served model so the running app loads the same artifact the reported metrics were computed from. (PR #18)
- **Chaitanya Bagul (Engineering)** — Built the routing layer and Baseline 2, the calibrated team classifier: TF-IDF into Multinomial Naive Bayes over `complaint_text`, isotonic calibration, a leakage-aware temporal split evaluated at the real team mix, and a single validation-fitted confidence threshold with a per-team precision guardrail. Produced the evaluation harness and committed the full artifact set to `docs/metrics/baseline_model_2/` with the accompanying write-up. Moved the north star from 26.4% to 46.5%. (PR #16, #17)
- **Salman Farcy (Data and Evaluation)** — Owned evaluation and integration for the week. Reviewed, merged and approved every branch that landed, and brought the repository onto a single consistent structure so that datasets, notebooks, write-ups and metrics sit in predictable locations across all four contributors' work. Evaluated Baseline 2 against its committed artifacts establishing the majority-class floors the model is measured against, and identifying the Personal & Short-Term Loans precision guardrail breach and the quarter-over-quarter threshold drift that causes every operating point to undershoot its validation estimate. Authored this report.
- **Sriramm S S (Users and Research)** — Built the dataset the entire project now runs on. Worked through the full CFPB narrative archive across every company, product category and issue type to assemble a clean, deduplicated corpus of 1,044,615 complaints spanning 3,801 companies, 73 product-specific issue labels and 11 support teams. Defined and documented the construction rules, excluded narratives whose identical text carried conflicting labels, assigned leakage groups so near-duplicate templates cannot straddle a split, and shipped it with archive hashes, reconciliation counts and ten passing validation checks. Every model and every metric in this report is trained and measured on that corpus.

## Lean canvas changes (if any)
- **Solution / Unfair advantage:** the archive is a closed set. The CFPB stopped publishing narratives, and this dataset is built entirely from the 10 archived export files covering 2024–2025. The corpus cannot grow from this source, only be re-cut. Any live product will eventually run on inputs drawn from a different distribution than it was trained on.
- **Customer segments:** scope widened from one bank to an industry corpus. The product is no longer "route complaints for a bank" but "route complaints across 3,801 companies into 11 support functions", which changes who the buyer is.
- **Customer segments / Unique value proposition:** the consumer is now a user, not just a data source. The app removes the product and issue dropdowns and replaces them with free text plus a tracking ID, so the value proposition now includes the person filing the complaint, not only the ops team reading it.
- **Key metrics:** the north star's definition was tightened, not just its value. It is now measured on a temporal split at the real team mix, with the confidence threshold fitted on validation and frozen for test, and paired with a per-team precision guardrail (floor 0.85). Session 04's 26.4% was measured on a single-bank test set with the threshold chosen on test, so the two figures are not directly comparable.
