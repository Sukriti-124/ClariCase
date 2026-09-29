---
team: ClariCase
session: 04
date: 2026-09-22
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
  value: 26.4%
  previous: N/A (first measurement)
---

## Shipped this week
- Data pipeline: src/data/{ingest,explore,prepare}.py. Loads the CFPB export, writes a profile to docs/data_profile.md, and produces a cleaned temporal split through a single entry point, get_splits(). 22,465 raw rows to 22,096 after cleaning; 15,714 train / 6,382 test.
- Baseline model: src/models/baseline.py. Majority-class dummy plus TF-IDF (1,2)-grams into a calibrated LinearSVC. All metrics written to docs/metrics/baseline/.
- Not deployed. There is no running product yet, so nothing is live

## Data collection and preparation

**Contributor: Sriramm S S ([SriRammSS](https://github.com/SriRammSS))**

Collection and preparation update added on 29 September 2026. Related task: [Data collection #7](https://github.com/farcypeer46/ClariCase/issues/7).

I started with company-specific complaints and prepared a Bank of America dataset containing 16,407 narratives across eight product categories. I checked it against the JPMorgan dataset and the available Morgan Stanley archive narratives to remove shared complaint texts. Once we agreed on 11 support teams, I expanded the collection to all companies so the project would not depend on one bank's product mix.

### Collection and label design

- Collected complaints from the official [CFPB narrative archives](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/), covering January 2024 through December 2025 across ten archive files.
- Kept records with a complaint description, complaint ID and issue. The final dataset contains **1,044,615 complaints from 3,801 companies**.
- Used the 11 agreed product categories as the project's support teams. Created **73 product-specific issue labels**, such as `Credit card :: Fees or interest`, and a fixed issue-to-team mapping. Sub-issues remain available for analysis but are not separate targets.
- Required each retained product–issue pair to have at least 500 unique narratives, 500 template groups and activity in at least 12 months. Kept every eligible complaint without class caps, oversampling or synthetic examples. Missing sub-issues were retained because some CFPB issues do not have them.

### Cleaning and preprocessing

- Preserved the original complaint text and source fields. Used Unicode normalization, whitespace normalization and case folding only to build exact-duplicate keys.
- Removed all copies when the same normalized text had conflicting product–issue labels. For repeated text with one consistent label, kept the earliest complaint, using complaint ID to break date ties.
- Removed descriptions containing no usable words beyond numbers or redaction placeholders. Recorded the excluded complaint IDs and reasons in the data package.
- Prepared separate text-processing functions for the issue-classification experiments: Unicode normalization, lowercasing, URL/email removal, removal of repeated `XX` redaction placeholders and whitespace cleanup. A later version also handles HTML entities and contractions, preserves negation and masks long numeric tokens. These transformations are applied when text features are created; the published CSV retains the original narratives. No stemming, lemmatization or blanket stop-word removal was applied.
- Kept product, issue, sub-issue, company, response fields and identifiers out of the text input. They remain labels or audit fields. Text vocabularies and feature weights were learned from training data only.

The record counts reconcile as follows:

| Outcome among narrative candidates | Records |
|---|---:|
| Retained in the constructed dataset | 1,044,615 |
| Identical text with conflicting labels | 568,050 |
| Additional copies with a consistent label | 415,097 |
| Product–issue pairs below the support requirements | 7,884 |
| No usable lexical content | 789 |
| **Total narrative candidates** | **2,036,435** |

### Split preparation and checks

Created `leakage_group_id` values to group some repeated complaint templates, including variations in numbers and redactions. Removed all 6,867 records in the 1,949 groups crossing time boundaries before preparing the full temporal partitions: **774,412 training records** (January 2024–June 2025), **153,502 validation records** (July–September 2025) and **109,834 test records** (October–December 2025). These are the full prepared partitions, before any experiment-specific training sample. All 73 issue labels and 11 teams are present, with no shared template groups between partitions. The grouping rules do not catch every paraphrase.

Also prepared a separate **29,798-record January–February 2026 evaluation dataset**, excluding known template overlaps with the 2024–2025 source collection. It covers all 11 teams and 72 of the 73 supported issues; the absent issue is identity-theft protection or monitoring services. This is a limited 2026 evaluation period, not a complete 2026 collection, and it has already been used for evaluation.

Checked complaint-ID and exact-text uniqueness, required fields, date boundaries, label-to-team consistency, class support, source reconciliation and CSV/Parquet agreement. All ten construction checks pass. The data remains imbalanced, and the published consumer-selected labels have not been manually verified.

**Deliverables:** [dataset archives and loading instructions](../data/README.md), [construction rules](../data/processed/README.md), [source metadata](../data/processed/dataset_metadata.json), [validation checks](../data/processed/validation.json) and [preparation details](../data/preparation/README.md). This section records the data work; the earlier session's model metrics elsewhere in this report describe its original experiment.

## User evidence
- This week we focused on data collection, exploratin and baseline modeling. Live deployment will be build in coming week.
- **Raw artifact**: docs/data_profile.md, docs/metrics/baseline/


## Metrics snapshot
- Autonomous routing rate @95% precision: 26.4% (First measurement)
- Macro-F1: 0.726 for TF-IDF + LinearSVC vs. 0.090 for a majority-class baseline, which ignores the text and always predicts the largest class
- Accuracy: 0.806 vs. 0.463 baseline
- Measured on: temporal held-out set, 2025-07-01 to 2026-01-30, 6,382 complaints, deduplicated before splitting
- Is this the same model that is running in the product? When the app ships next week it will load this same pipeline, so the deployed model and the reported model will match.

## What did not work
- 26.4% is not a useful product. Hitting 95% precision requires a 0.90 confidence threshold, which leaves three quarters of the queue for a human. An ops team saving a quarter of their reading time is a weak pitch.
- Debt collection recall is 0.41. Well over half get missed, mostly to credit reporting — the two share FCRA and FDCPA vocabulary almost entirely.
- Money transfer misroutes to Checking or savings 41% of the time. A Zelle dispute reads exactly like a checking complaint because the money left a checking account. We do not yet know whether this boundary is learnable or genuinely ambiguous.
- Calibration costs accuracy. Macro-F1 falls 0.731 to 0.726 when the SVM is wrapped for probability estimates. Small, but we are reporting the lower number because it belongs to the model that can actually produce confidence scores.
- Our label quality is unmeasured. Product labels are chosen by the consumer from a dropdown at intake, never audited. Some share of our 19% error rate is the data being wrong, and we cannot currently say how much.

## Challenges / blockers
- No deployed app. Everything else depends on this. Nothing can be tested with a user until a stranger can open a URL.
- The task is partly circular as posed. Every complaint in this corpus arrived through the CFPB web form already labelled by the consumer, so predicting the label on this data proves little. The product only makes sense as: learn the taxonomy here, apply it to unlabelled channels elsewhere.
- Two classes are thin in test - Mortgage (157) and Vehicle loan (104). Per-class recall on these will swing between runs and should be read as noisy, not precise.
## Next week's goal
- Widen the corpus. Add Bank of America and Wells Fargo to the current JPMorgan-only data. Today's numbers describe one bank's product mix, so the class floor and the config thresholds get re-checked after the merge, more data may pull Prepaid card or Payday loan back above the cutoff we dropped them at. We will also check whether per-class performance differs by company, since a model can learn "sounds like Wells Fargo, therefore credit card" instead of learning the language of card complaints.
- Move the north star. 26.4% is the number to beat. Three levers, cheapest first: proper calibration (temperature scaling rather than Platt, measured by ECE), then per-class thresholds instead of one global cut, then a fine-tuned transformer. The two worst classes - debt collection at 0.41 recall and money transfer at 0.50 are where the headroom is, since low-confidence predictions concentrate there.

## Individual contributions
- Sukriti Srivastava (Product): Data ingestion and exploration
- Chaitanya Bagul (Engineering): Baseline Modelling 
- Salman Farcy (Data and Evaluation): Data cleaning  
- Sriramm S S (Users and Research): CFPB archive collection, initial Bank of America preparation, expansion to the all-company dataset, duplicate and conflicting-label handling, product-specific issue/team mappings, text preprocessing and temporal data preparation. See the data collection and preparation section above.

## Lean canvas changes (if any)
- New data risk. The CFPB stopped publishing complaint narratives in August 2026 and removed the consent mechanism in September. Our corpus is a fixed historical snapshot, not a live feed - the "updates daily" claim is gone. It also strengthens the case for the product: the labelled signal we trained on is no longer being generated.
