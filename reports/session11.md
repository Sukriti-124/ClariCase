---
team: ClariCase
session: 11
date: 2026-11-17
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
  metric: Autonomous routing rate at >=90% precision
  value: 86.3%
  previous: 85.3%
---

## Shipped this week
- **North star moved to a 90% precision target** (`src/utils/config.py`, 15 November). Every earlier target is still computed and saved, so 95% figures remain comparable with older runs.
- **Improved Model 2** (`src/improved_model_2/`): a calibrated logistic regression on fixed sentence embeddings, a second, differently built team classifier for the ensemble.
- **Ensemble routing** (`src/ensemble/`): a stacker combines Improved Model 1, Improved Model 2, Baseline 2 and a nearest-neighbour model on embeddings; a gate estimates the probability that the chosen team is right; per-team thresholds with an 85% precision floor decide auto-routing. A label-shift correction re-estimates the incoming team mix from unlabelled traffic and reweights the stacker's probabilities.
- **Model explainability** (`src/explain/`): every prediction gets a decision record of why it was routed, a narrator turns it into a plain-language explanation, and a validator checks every cited fact and number against the record before it is shown. A "Why?" panel in the Streamlit app, `/explain` endpoints, and three new database columns (`decision_record`, `explanation`, `explanation_audit`). 41 test files.
- **Platform, phase 1** (`platform/`): a router, model services (`svm-api`, `embed-api`, `ensemble-api`, `agentic-api`), a shared contract and observability library, and Postgres on docker-compose, with a design spec and implementation plan (`docs/superpowers/`). Logging, metrics and tracing hooks are installed in every service.
- **Router fallback** (branch `router-fallback`, not yet merged): if the primary model service times out, is unreachable or returns an error, the router retries once on a fallback service, so a complaint is answered by Improved Model 1 if the ensemble is down. 5 new tests; all 20 router tests pass.
- **Agentic issue-classification code with tests** (`src/agentic/`), and a root `pyproject.toml` that exposes `src/` as a package.
- **Unit economics and pricing analysis** (`docs/unit_economics.md`, PR #39).

## What changed since session 10
- **The north star definition changed** from 95% to 90% precision. 90% is reached by more complaints and holds better on newer data, which suits the product's "route the clear cases, review the rest" design. Under the new definition the previous figure is Improved Model 1's 85.3%.
- **The served model is now an ensemble.** Combining four models with a gate and per-team floors routes more complaints than Improved Model 1 alone and stays above its precision target on the January–February 2026 cohort, where Improved Model 1 does not.
- **The product explains its decisions.** Each routing decision can now be shown with a validated, plain-language reason.
- **Jev is concluded.** Its best configuration (structured descriptions, session 10) does not improve routing, so it is not part of the ensemble.

## User evidence

- **Raw artifact**: `docs/metrics/ensemble_stack/`, `docs/metrics/ensemble_gate/`, `docs/metrics/improved_model_2/`, `src/explain/`, `platform/`

## Metrics snapshot
- **Autonomous routing rate @90% precision: 86.3%, up from 85.3%** (served ensemble, at 90.7% precision, full test set of 109,834 complaints, no team below the 85% floor).
- All figures below are on the full test set (October–December 2025) or the January–February 2026 cohort, with thresholds fitted on validation and frozen.

| Full test set | Improved Model 1 alone | Improved Model 2 alone | **Ensemble (served)** |
|---|---:|---:|---:|
| Accuracy | 0.844 | 0.814 | **0.855** |
| Macro-F1 | 0.726 | 0.665 | **0.742** |
| **North star: auto-routed at 90% precision** | 85.3% (at 89.8%) | 76.6% (at 89.5%) | **86.3% (at 90.7%)** |
| Teams below the 85% floor | Money Transfers, Vehicle Finance, Personal Loans | Banking, Money Transfers, Personal Loans | **none** |

| January–February 2026 cohort (29,798) | Improved Model 1 alone | **Ensemble (served)** |
|---|---:|---:|
| Accuracy | 0.767 | **0.834** |
| Macro-F1 | 0.697 | **0.707** |
| Auto-routed at 90% target | 77.6% (at 84.2%) | **79.7% (at 91.3%)** |
| Teams below the 85% floor | Credit Reporting, Money Transfers, Personal Loans, Prepaid Card | Personal Loans |

- **The ensemble holds on newer data where Improved Model 1 does not.** On the 2026 cohort, Improved Model 1's frozen threshold drops to 84.2% precision; the ensemble stays at 91.3%. The label-shift correction drives this: Credit Reporting fell from about 61% of complaints in training to about 2% in the 2026 cohort, and the correction re-estimates that mix without labels.
- **At the old 95% target, for comparison:** a gate on Improved Model 1 and Model 2 routes 66.4% at 94.5%, against 61.5% at 94.8% for Improved Model 1 alone.
- **Improved Model 3 (sequence-to-sequence) progress:** its latest checkpoint reaches 0.555 macro-F1 and 0.740 accuracy on a 2,000-complaint validation sample, against 0.485 and 0.745 for the first trial checkpoint on the full validation set; the samples differ, so the comparison is indicative. It is not yet part of the ensemble.
- **Cost of serving:** Improved Model 1 routes a complaint in about 4 ms on one CPU core, under $0.10 per million complaints (`docs/unit_economics.md`).

## What did not work
- **Improved Model 2 alone is weaker than Improved Model 1** (macro-F1 0.665 against 0.726), and three teams fall below the precision floor. Its value is as a second, differently built opinion inside the ensemble, not as a standalone model.
- **Personal & Short-Term Loans stays below the floor on the 2026 cohort** even with the ensemble.
- **Accuracy on the 2026 cohort is well below the test set** (0.834 against 0.855), because the team mix shifted sharply. The label-shift correction protects precision but cannot recover all of the accuracy.
- **The headline north star rose partly because its definition changed.** At the same 90% target, the ensemble's gain over Improved Model 1 is 1.0 point on the test set and 2.1 points on the 2026 cohort, with a much larger precision gain on the 2026 cohort.

## Challenges / blockers
- **No continuous integration yet.** Test suites run locally only; nothing stops a failing change from being merged.
- **Human review has no workflow.** Complaints below the routing threshold are marked "Under review", but there is no queue or reviewer screen yet.

## Next week's goal
  - Merge the explainability, platform and router-fallback branches into main, after the Supabase migration.
  - Add a CI pipeline that runs the test suites on every pull request.
  - Build a human review queue for complaints below the threshold, with the model's suggestions and the validated explanation.
  - Fold Improved Model 3 into the ensemble once fully trained.

## Individual contributions
- **Sukriti Srivastava (Product)** — Added an optional fallback model to the platform router so a complaint is still answered when the primary model service is down, with a served-by header, documentation and 5 tests (branch `router-fallback`). Prepared the Supabase migration for the new explainability columns. Wrote this report.
- **Chaitanya Bagul (Engineering)** — Designed and built the platform's first phase (router, model services, shared contract and observability, Postgres, docker-compose), Improved Model 2, the stacked and gated ensemble with label-shift correction, the agentic issue-classification code, and model explainability (decision records, validator, narrator, `/explain` endpoints, Streamlit panel), with tests throughout. Updated Improved Model 3's checkpoint results and moved the north star to the 90% target.
- **Salman Farcy (Data and Evaluation)** — Measured the cost of routing a complaint (about 4 ms per complaint on one CPU core, under $0.10 per million) and compared it with what the market charges for comparable triage work, in `docs/unit_economics.md`. (PR #39)
- **Sriramm S S (Users and Research)** — **TODO (Sriramm): add this session's work.**

## Lean canvas changes (if any)
- **Key metrics:** the north star now targets 90% precision, paired with the 85% per-team floor and a check on the newest data available (the 2026 cohort), where the served model is required to hold its precision.
