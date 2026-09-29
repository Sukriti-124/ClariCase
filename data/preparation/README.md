# Data preparation

Contributor: **Sriramm S S (SriRammSS)**. This directory records preprocessing and split preparation for the collected complaint data.

## Original text and processed text

The published archive preserves `complaint_text` exactly as received from the CFPB narrative source. Duplicate keys are separate audit fields. Normalization for deduplication is not a replacement for the original narrative.

- `text_features.py` contains the text-preparation function used for the initial issue-classification experiments: Unicode NFKC normalization, lowercasing, URL and email removal, repeated-X redaction removal, and whitespace normalization.
- `advanced_text.py` contains the later text-preparation function: the same general cleanup plus HTML unescaping, curly-apostrophe normalization, contraction handling that retains negation, and masking numeric tokens of six or more digits. Short numbers and alphanumeric terms remain available.

Neither function performs stemming, lemmatization or blanket stop-word removal. Processing happens when features are created; the CSV is not overwritten. Any learned vocabulary, IDF weights or class weights must be fit on the appropriate training partition. Product, issue, sub-issue, team, company, IDs and response fields are not supplied as text features.

## Full temporal partitions

| Partition | Complaint receipt dates | Records |
|---|---|---:|
| Training | 2024-01-01–2025-06-30 | 774,412 |
| Validation | 2025-07-01–2025-09-30 | 153,502 |
| Test | 2025-10-01–2025-12-31 | 109,834 |
| Excluded because their template group crossed periods | Across the above boundaries | 6,867 |

Assign each complaint to a period by `date_received`. For every `leakage_group_id` occurring in more than one period, exclude every record in that group from all partitions. This removes 1,949 crossing groups, leaving zero group overlap. All 73 issue labels and 11 teams remain in each partition. The smallest issue supports are 374 training, 54 validation and 30 test records.

These counts describe the full eligible data. The app's separate training procedure may take a smaller training sample; that does not change the collected archive. The source dataset was curated retrospectively across both 2024 and 2025, and template grouping only detects the documented rule-based variations, not every semantic duplicate.

## Separate 2026 cohort

The January–February 2026 archive yielded 29,798 retained narratives after removing known template overlaps with the original 2024–2025 source collection, conflicting exact texts, extra consistent copies, unusable text and unsupported labels. No per-class quota was used. The cohort contains 11 teams and 72 supported issues; identity-theft protection or other monitoring services has no examples.

This cohort has already been used for evaluation. It is preserved for reproducibility, not available as a new untouched holdout. Source and exclusion details are included inside `../complaint_router_2026_holdout.zip`.
