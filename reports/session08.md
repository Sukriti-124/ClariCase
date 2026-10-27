---
team: ClariCase
session: 08
date: 2026-10-27
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
  previous: 61.5%
---

## Shipped this week
- **PII masking in the app** (`src/app/pii.py`, PR #29). Emails, URLs, Social Security numbers, payment cards (Luhn-checked), phone numbers, account numbers, street addresses, ZIP codes and dates are replaced with the CFPB's own markers (`XXXX`, `XX/XX/XXXX`) before a complaint is classified, and again before it is stored, so raw details are never predicted on or saved. Because the model's text cleaning already drops these markers, masking adds no unseen tokens. 21 tests.
- **PII masking fixes** (branch `fix/pii-false-positives`, committed, not yet merged). The first version masked ordinary words: "flagged", "stealing", "flight", "floor", "steps", and phrases such as "charged 3 times along the way". Both address rules were tightened, alphanumeric reference codes ("ref #A1B2C3D4E5") are now masked, names are deliberately kept by team decision, and the app now tells users that personal details are hidden. 40 tests.
- **Improved Model 3, a transformer classifier** (`src/improved_model_3/`, PRs #30 and #31). FLAN-T5-base fine-tuned with LoRA (rank 16 on the attention q and v projections, 0.79% of weights trainable) generates the team name from the complaint text. Redaction markers map to dedicated tokens, inputs are cropped to 320 tokens (75% head, 25% tail). Includes training, a prediction script, and a first evaluation on the full validation set.
- **Project slide decks** (`docs/ClariCase_Architecture.pptx`, `ClariCase_Midterm.pptx`, `ClariCase_Workflow.pptx`) and a training-progress monitor (`scripts/watch_training.ps1`).

## What changed since session 07
- **Privacy:** the session 07 blocker "the app has no protection for sensitive details" is closed. PII masking is merged into main, with a fix pending review.
- **Modelling direction:** after two linear models, the team started on transformers. Improved Model 3 is the first, trained so far only as a small trial run.
- **The served model is unchanged.** The app still runs Improved Model 1, so the north star stays at 61.5%.

## User evidence
- **TODO (team): add any user testing done this session**, or state plainly that none took place.
- **Raw artifact**: `src/app/pii.py`, `tests/test_pii.py`, `docs/metrics/improved_model_3/`

## Metrics snapshot
- **Autonomous routing rate @95% precision: 61.5%, unchanged.** Improved Model 1 is still the served model: 67,550 of 109,834 test complaints routed at a 0.818 threshold, at 94.78% precision.
- Improved Model 3 was trained as a trial ("smoke") run: 50,000 of the 257,694 training complaints, one epoch, team prediction only. It was then scored on all 153,502 validation complaints. Test-set results are not available yet.

| Validation set (153,502 complaints) | Improved Model 3 (FLAN-T5, trial) | Baseline 2 | **Improved Model 1** (served) |
|---|---:|---:|---:|
| Team macro-F1 | 0.485 | 0.623 | **0.709** |
| Team accuracy | 0.745 | 0.810 | **0.850** |
| Weighted-F1 | 0.751 | 0.805 | **0.848** |

| Team (validation) | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T01 Credit Reporting | 0.929 | 0.794 | 0.856 | 93,050 |
| T06 Mortgage | 0.626 | 0.867 | 0.727 | 3,651 |
| T04 Banking Accounts | 0.665 | 0.706 | 0.684 | 11,332 |
| T02 Debt Collection | 0.564 | 0.675 | 0.615 | 21,298 |
| T05 Money Transfers & Digital Currency | 0.711 | 0.539 | 0.613 | 5,173 |
| T08 Vehicle Finance | 0.499 | 0.762 | 0.604 | 3,128 |
| T03 Credit Card | 0.435 | 0.778 | 0.558 | 10,517 |
| T07 Student Loan | 0.407 | 0.787 | 0.537 | 1,960 |
| T10 Prepaid Card | 0.768 | 0.074 | 0.135 | 1,026 |
| T09 Personal & Short-Term Loans | 0.500 | 0.001 | 0.001 | 1,808 |
| T11 Debt & Credit Management | 0.000 | 0.000 | 0.000 | 559 |

- The trial run took about 17.4 hours to train (62,568 seconds) on a 6 GB laptop GPU, using only 0.8 GB of its memory. Scoring the validation set took 49 minutes, about 53 complaints per second.
- Every generated output was a valid team name (unknown-output rate 0%).
- PII masking: 40 tests pass on the fix branch, covering each false positive found in review alongside the original 21 cases.

## What did not work
- **Improved Model 3 is well behind the served model.** Validation macro-F1 is 0.485 against Improved Model 1's 0.709, and below Baseline 2's 0.623. This is a trial run on a fifth of the training data for one epoch, so it does not settle whether the approach can work.
- **The three smallest teams collapse.** Personal & Short-Term Loans is predicted correctly once in 1,808 complaints, Debt & Credit Management never, and Prepaid Card 7.4% of the time. Nothing in the trial run compensates for class imbalance, so the generator learned to favour common teams.
- **Mid-sized teams are over-predicted.** Credit Card precision is 0.435: 4,933 Credit Reporting, 1,973 Banking and 533 Prepaid Card complaints were sent there. Credit Reporting also loses 9,692 complaints to Debt Collection.
- **Training was slow and looks unstable.** 17.4 hours for 50,000 complaints, including a stall of over two hours. The average training loss (52.8) is far above the validation loss (0.50), which suggests a problem early in training worth investigating before a full run. Batch size could be raised substantially, since most of the GPU's memory went unused.
- **The recorded run information disagrees with the training log** (2.8 hours and torch 2.14 in `run_info.json`, against 17.4 hours and torch 2.11 in `train.log`), so it was probably overwritten by a later run.
- **The first PII version masked ordinary words**, removing signal the model routes on ("flagged" and "stealing" in fraud complaints, "flight" in travel disputes). Found in review and fixed on the branch.
- **Last week's other goals are in progress, without results yet.** Llama 3.2 3B fine-tuning continued but has not been evaluated, and Jev AI was explored but not yet tested on our data. The team-to-issue gap was not addressed this week.

## Challenges / blockers
- **Compute.** Transformer training on a 6 GB laptop GPU is slow: a full training run at the trial run's speed would take days. Colab or a larger GPU, plus larger batches, are needed to iterate within a week.
- **Model 3 cannot be compared on the north star yet.** It has no calibrated confidence scores, so there is no auto-route or review decision, and it predicts teams only, not issues.
- **Stored complaints contain personal data.** Names are kept by design, and complaints saved before masking existed are still unmasked in the database.
- The forward-drift cohort is no longer fresh. A new untouched evaluation window will be needed to report future gains credibly.

## Next week's goal
- **Train Improved Model 3 properly.** Use the full training set, compensate for class imbalance so the small teams are learned, raise the batch size, and investigate the early training-loss spike. Then score it on the test set with calibration, so it can be compared on the north star.
- **Merge the PII fixes** and mask the complaints already stored in the database.
- **Use Jev AI for classification and fine-tuning.** Test Jev AI as a complaint classifier, since it returns a structured choice with a probability, and fine-tune it on our training data. Score it on the same test set and 95%-precision rule as Improved Model 1.

## Individual contributions
- **Sukriti Srivastava (Product)** — Reviewed the PII masking module and fixed its false positives: tightened both address rules so words such as "flagged", "stealing" and "flight" are no longer removed, added masking for alphanumeric reference codes while keeping loan amounts intact, implemented the team's decision to keep names, and added privacy notes to the submit and track screens. Added 19 tests (branch `fix/pii-false-positives`). Explored Jev AI, how it can be used for complaint classification, and its benefits for the product. Wrote this report.
- **Chaitanya Bagul (Engineering)** — Built Improved Model 3, the FLAN-T5-base + LoRA team classifier: tokenizer with dedicated redaction tokens, data pipeline with head-and-tail cropping, training and prediction scripts. Ran the trial training run and evaluated it on the full validation set, and added the training monitor script. (PRs #30, #31)
- **Salman Farcy (Data and Evaluation)** — Built the PII masking module (`src/app/pii.py`): CFPB-style redaction of emails, URLs, Social Security numbers, Luhn-checked cards, phone and account numbers, addresses, ZIP codes and dates, applied at intake and at storage, with 21 tests. Reviewed and merged PRs #29–#31. (PR #29)
- **Sriramm S S (Users and Research)** — Continued fine-tuning Llama 3.2 3B Instruct for complaint classification.

## Lean canvas changes (if any)
- **Unique value proposition:** privacy is now part of the product. Consumers can describe a problem freely, and account numbers, cards and other identifiers are removed before anything is classified or saved.
- **Cost structure:** the transformer path is much more expensive to train than the linear model (17.4 hours for a fifth of the data, against 4.5 hours for Improved Model 1's full run). Any accuracy gain from Model 3 will need to justify that compute cost.
