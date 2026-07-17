# Data Preparation Findings

## Purpose

This stage created the first policy-level analytical dataset while
preserving the original raw files and documenting inconsistencies
between the frequency and severity tables.

## Input Data

- Raw frequency records: 678,013
- Raw severity records: 26,639
- Valid policy records: 678,013
- Valid severity records: 26,639

## Cross-Table Consistency

- Claims reported through `ClaimNb`: 36,102
- Valid rows in the severity table: 26,639
- Cross-table mismatched policy IDs: 9,123
- Frequency policies with count mismatches: 9,117
- Severity policy IDs without frequency records: 6

The total cross-table mismatch count includes severity policy IDs that
do not exist in the frequency table. These severity-only IDs cannot be
included in the final policy-level analytical dataset.

## Audit Status Summary

| Audit status | Policy count | Percentage |
|---|---:|---:|
| no_claim_no_severity | 643,953 | 94.976% |
| matched_positive_claims | 24,943 | 3.679% |
| claim_without_severity | 9,116 | 1.345% |
| severity_without_frequency | 6 | 0.001% |
| fewer_severity_rows_than_claimnb | 1 | 0.000% |

## Cleaning Decisions

### Raw data preservation

The raw CSV files were not modified. Cleaned and derived files were
saved separately in `data/interim` and `data/processed`.

### Duplicate severity rows

Duplicate severity rows were retained. The dataset does not provide a
unique claim identifier, so identical policy IDs and claim amounts
cannot safely be classified as accidental duplicates.

### Missing severity information

A policy with `ClaimNb > 0` but no matching severity record is treated
as having incomplete loss information rather than a zero-dollar loss.

### Partial severity information

When the number of severity rows does not match `ClaimNb`, the observed
claim amount is preserved in `SeverityTotalClaimAmount`, but complete
loss variables are set to missing.

### Zero-claim policies

A policy with zero reported claims and zero severity rows receives a
complete total claim amount and pure premium of zero.

### Outliers

No numerical outliers were removed or capped during this step.

## Analytical Samples

### Frequency modelling

All structurally valid policy records are eligible for claim-occurrence
and claim-frequency modelling.

### Severity modelling

Only policies with at least one claim and a matching number of severity
records are eligible for the severity model.

### Pure-premium modelling

Only policies whose `ClaimNb` agrees with the number of severity rows
are treated as having a complete pure-premium outcome.

## Output Dataset

The policy-level analytical dataset contains 678,013 rows.

Each row represents one policy and contains:

- original policy characteristics;
- claim occurrence;
- annualized claim frequency;
- observed severity summaries;
- claim-count consistency flags;
- complete loss variables where available;
- model-eligibility indicators.

## Important Limitation

The severity table does not contain all claims reported in the
frequency table. Pure-premium and severity results must therefore be
interpreted using the completeness and eligibility indicators.
