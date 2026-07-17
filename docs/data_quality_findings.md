# Initial Data-Quality Findings

## Dataset Dimensions

- Frequency dataset: 678,013 rows and 12 columns.
- Severity dataset: 26,639 rows and 2 columns.
- Unique policies in frequency data: 678,013.
- Unique policies represented in severity data: 24,950.

## Missing Values

- Total missing values in frequency data: 0.
- Total missing values in severity data: 0.

## Duplicate Review

- Completely duplicated frequency rows: 0.
- Completely duplicated severity rows: 255.
- Duplicated policy identifiers in frequency data: 0.

Repeated policy identifiers in the severity dataset are expected because
one policy may have multiple claims.

Completely duplicated severity records will not be removed automatically.
Two claims belonging to the same policy may legitimately have the same
claim amount.

## Claim Occurrence

- Policies with at least one reported claim: 34,060.
- Policies without a reported claim: 643,953.
- Observed policy claim rate: 5.02%.

The claim-occurrence target is imbalanced because most policies do not
have a claim. Accuracy alone will therefore not be an appropriate model
evaluation metric.

## Cross-Table Consistency

- Total claims according to `ClaimNb`: 36,102.
- Total rows in the severity dataset: 26,639.
- Policies with inconsistent claim counts: 9,123.
- Severity policy IDs missing from frequency data: 6.
- Claim policies without severity records: 9,116.

Any inconsistencies will be investigated and documented before the
frequency and severity tables are joined.

## Validity Review

Basic validity checks were performed for:

- missing and non-positive identifiers;
- negative or non-integer claim counts;
- non-positive exposure;
- negative driver or vehicle ages;
- negative bonus-malus values;
- non-positive population density;
- non-positive claim amounts.

Unusual values have not yet been removed. Extreme insurance values may
represent legitimate observations and require further investigation.

## Next Actions

The next data-preparation stage will:

1. determine appropriate data types;
2. resolve or document cross-table inconsistencies;
3. create policy-level claim totals;
4. create derived variables;
5. save cleaned interim datasets;
6. preserve the original raw data unchanged.
