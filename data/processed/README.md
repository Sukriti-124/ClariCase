# Consumer complaint routing dataset

Complaints received January 1, 2024–December 31, 2025, from all companies in the official CFPB narrative archives. This dataset contains **1,044,615 complaints, 73 product-specific issue labels, 11 teams, and 3,801 companies**.

This is the constructed dataset. Narrative preprocessing, an EDA notebook, train/validation/test splits, and model training have not been performed.

## Prediction task

Use `complaint_text` as the model input and `product_issue_id` as the classification target. `product_issue_label` gives the readable target, for example `Credit card :: Fees or interest`. After predicting the product-specific issue, use `issue_mapping.csv` to obtain its single `team_id` and `team_name`.

The first model therefore predicts both the issue and its product context. Routing is a deterministic lookup; a second trained routing model is unnecessary. Product names represent the project's support teams, not independently verified organizational departments at each company.

Use the published Product and Issue selections as supervision. They are consumer-selected categories and may still contain label errors. No individual complaint has been manually relabeled.

## Files

| File | Contents |
|---|---|
| `complaints_product_issues_2024_2025.csv` | Complete dataset, UTF-8, with the original complaint text. Approximately 1.91 GB. |
| `complaints_product_issues_2024_2025.parquet` | The same rows and columns in a smaller format. Approximately 352 MB. Provided separately from the ZIP. |
| `issue_mapping.csv` | The 73 supported issue labels, their team mappings, and support counts. |
| `team_mapping.csv` | The 11 fixed product-to-team mappings. |
| `team_coverage.csv` | Included records, issue counts, and coverage for each team. |
| `issue_eligibility.csv` | Every observed product–issue pair after record-level exclusions, including the reasons unsupported pairs were excluded. |
| `excluded_records.csv.gz` | Complaint IDs and reasons for record-level exclusions. |
| `dataset_metadata.json` | Construction rules, archive URLs and hashes, reconciliation counts, and output hashes. |
| `validation.json` | Checks performed on the saved dataset. |

The source directory contains the construction code, archive metadata, and cached narrative candidates. It is not needed to load the delivered dataset and is not included in the ZIP.

## Construction rules

1. Read all ten official archive parts overlapping 2024–2025 and restrict the complaint receipt date to those two complete years.
2. Include all eleven agreed Product categories, across all companies. Require a nonblank narrative, an Issue, and a Complaint ID.
3. Exclude descriptions containing no alphabetic word content beyond redaction placeholders. Do not otherwise remove short or long complaints.
4. Detect identical narratives after Unicode NFKC normalization, whitespace collapse, trimming, and case folding. This normalization is applied only to the duplicate-detection key; the delivered narrative stays unchanged.
5. If an identical normalized narrative has different product–issue labels, exclude all copies from the modeling dataset. Do not pick a majority label. These exclusions identify ambiguous supervision; they do not establish that the original complaints are invalid.
6. For identical narratives with the same product–issue label, keep the earliest received record, breaking ties with the smallest numeric Complaint ID. `narrative_duplicate_count` records the number of source copies in that group.
7. Include a product–issue pair when it has at least 500 unique narratives, 500 rule-based template groups, and records in at least 12 distinct months. These are practical support requirements, not statistical guarantees of model performance. The actual 73 retained classes each have at least 517 narratives and template groups, appear in all 24 months, and involve at least 27 companies.
8. Keep every eligible record. No random sampling, class caps, oversampling, synthetic examples, or company quotas were used.

The primary support unit is `(Product, Issue)`. Sub-issues are retained for later coverage analysis and are not prediction targets. Missing sub-issues are valid for many CFPB issues; all 66,513 such records were retained.

## Reconciliation

| Outcome among the 2,036,435 narrative candidates | Records |
|---|---:|
| Included | 1,044,615 |
| Identical text with conflicting product–issue labels | 568,050 |
| Additional copies of an identical narrative with a consistent label | 415,097 |
| Product–issue pair below the support requirements | 7,884 |
| No lexical content beyond numbers/redactions | 789 |

Included records cover 99.25% of the 1,052,499 distinct, unambiguous narratives after record-level exclusions. This percentage is not coverage of all raw CFPB complaints.

| Team | Issue labels | Included records |
|---|---:|---:|
| Credit Reporting | 8 | 645,108 |
| Debt Collection | 7 | 122,899 |
| Credit Card | 12 | 69,943 |
| Banking Accounts | 5 | 76,061 |
| Money Transfers & Digital Currency | 10 | 48,164 |
| Mortgage | 5 | 25,641 |
| Student Loan | 4 | 17,505 |
| Vehicle Finance | 6 | 18,274 |
| Personal & Short-Term Loans | 7 | 10,052 |
| Prepaid Card | 4 | 7,343 |
| Debt & Credit Management | 5 | 3,625 |

Support filtering retains 77.85% of the distinct, unambiguous Personal & Short-Term Loans records and 88.65% of Debt & Credit Management records. The remaining issues for those teams are outside this version's issue-classification scope. The other teams retain 93.74% or more. Use the excluded-issue catalog to define a later manual-review or expansion policy; retaining a team does not mean every possible issue for that team is supported.

## Duplicate groups and later splitting

`narrative_group_id` is the SHA-256 key used for exact normalized-text detection. It is unique in the final dataset.

`leakage_group_id` conservatively groups some repeated templates. For narratives containing at least 20 meaningful word tokens, the key masks numeric/redaction tokens and ignores punctuation, spacing, and case. For shorter narratives it uses the exact narrative key. This does not remove records or modify text. It detects a particular family of template variations, not every semantic near-duplicate or paraphrase.

Keep every `leakage_group_id` wholly within one later split. The group can include different issue labels: `leakage_group_issue_count` makes those cases visible. Inspect these cases during later cleaning. For a temporal evaluation, remove groups that cross the chosen boundaries or use a documented group-aware allocation; do not split simply by row date and ignore groups.

The period counts in `issue_mapping.csv` are construction diagnostics, not assigned partitions. Nine retained labels have fewer than 100 records in October–December 2025. Recheck per-label counts after grouping when choosing an evaluation window, and report uncertainty for small test classes. No test performance was used to choose classes.

## Later preprocessing and training

- Preserve the raw `complaint_text` column. Create a separate processed column when preprocessing begins.
- Fit text vectorizers, learned preprocessing, class weights, and resampling on the training partition only.
- The dataset remains imbalanced because all eligible records were requested. Credit Reporting represents 61.76% of records. Use training-only class weights or balanced training batches and report macro-F1 and per-team/per-issue performance.
- Do not feed `product`, `issue`, `sub_issue`, team fields, issue-label fields, IDs, group hashes, or archive metadata to the text-only classifier. They are targets or audit information. Company is retained for auditing representation, not as a required model input.
- Do not assume Naive Bayes probability scores are calibrated confidence estimates.
- Current safeguards reduce duplicate leakage and ambiguous supervision. They do not certify the truth of narratives, eliminate all label noise, or establish demographic representativeness.

## Loading

For the complete dataset with less storage and faster loading:

```python
import pandas as pd

df = pd.read_parquet("complaints_product_issues_2024_2025.parquet")
X = df["complaint_text"]
y = df["product_issue_id"]
groups = df["leakage_group_id"]
```

For CSV, read in chunks if memory is limited:

```python
import pandas as pd

chunks = pd.read_csv(
    "complaints_product_issues_2024_2025.csv",
    dtype={"complaint_id": "string"},
    keep_default_na=False,
    chunksize=50000,
)
for chunk in chunks:
    pass
```

`complaint_id`, target IDs, and group hashes are identifiers. `date_received` is a date. The record order is chronological, with numeric Complaint ID as the tie breaker; it is not a predefined split.

## Sources

- Narrative archive: https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/
- Field definitions: https://cfpb.github.io/api/ccdb/fields.html
- Product and issue taxonomy: https://files.consumerfinance.gov/f/documents/cfpb_consumer_complaint_form_product_issue_options_August_2023_FINAL.pdf

Each dataset row includes the exact source archive URL, archive number, and CSV record number. Record numbers refer to parsed CSV records including the header, not physical file lines, because narratives can contain embedded newlines.
