# Data Dictionary

## Frequency Dataset

File: `data/raw/freMTPL2freq.csv`

The frequency dataset contains one row per automobile insurance policy.

| Variable | Type | Description | Role |
|---|---|---|---|
| IDpol | Identifier | Unique policy identifier | Join key |
| ClaimNb | Integer | Number of claims during the exposure period | Target |
| Exposure | Numeric | Length of observed policy coverage in years | Weight |
| Area | Categorical | Area category from rural to urban | Predictor |
| VehPower | Ordinal | Vehicle-power category | Predictor |
| VehAge | Numeric | Vehicle age in years | Predictor |
| DrivAge | Numeric | Driver age in years | Predictor |
| BonusMalus | Numeric | Insurance bonus-malus score | Predictor |
| VehBrand | Categorical | Anonymized vehicle-brand category | Predictor |
| VehGas | Categorical | Diesel or regular fuel | Predictor |
| Density | Numeric | Population density of the driver's location | Predictor |
| Region | Categorical | Geographic region code | Predictor |

## Severity Dataset

File: `data/raw/freMTPL2sev.csv`

The severity dataset contains one row per recorded claim.

| Variable | Type | Description | Role |
|---|---|---|---|
| IDpol | Identifier | Policy associated with the claim | Join key |
| ClaimAmount | Numeric | Monetary cost of the claim | Target |

## Planned Derived Variables

These variables will be created during data preparation.

| Variable | Formula or Definition |
|---|---|
| HasClaim | 1 when ClaimNb is greater than zero; otherwise 0 |
| TotalClaimAmount | Sum of ClaimAmount for each policy |
| Frequency | ClaimNb divided by Exposure |
| AverageClaimAmount | TotalClaimAmount divided by ClaimNb |
| PurePremium | TotalClaimAmount divided by Exposure |
| ClaimRateBand | Risk category based on observed claim frequency |
| HighLossPolicy | Policy exceeding a documented loss threshold |

## Policy-Level Analytical Dataset

File: `data/processed/policy_analytics.csv`

Each row represents one valid automobile insurance policy.

| Variable | Description |
|---|---|
| HasClaim | 1 when `ClaimNb` is greater than zero |
| HasSeverityRecord | 1 when at least one severity row exists |
| SeverityRowCount | Number of severity-table rows for the policy |
| SeverityTotalClaimAmount | Sum of observed severity claim amounts |
| SeverityAverageClaimAmount | Average amount among observed severity rows |
| SeverityMinimumClaimAmount | Smallest observed claim amount |
| SeverityMaximumClaimAmount | Largest observed claim amount |
| ClaimCountDifference | `ClaimNb` minus `SeverityRowCount` |
| ClaimCountMatch | Whether both claim counts agree |
| SeverityCompletenessStatus | Description of cross-table completeness |
| AnnualizedClaimFrequency | `ClaimNb` divided by `Exposure` |
| ObservedAverageClaimAmount | Observed claim total divided by severity rows |
| CompleteTotalClaimAmount | Total loss when claim counts agree; otherwise missing |
| CompleteAverageClaimAmount | Average loss for matched positive-claim policies |
| CompletePurePremium | Complete claim cost divided by exposure |
| FrequencyModelEligible | Whether the policy can be used for frequency modelling |
| SeverityModelEligible | Whether the policy can be used for severity modelling |
| PurePremiumModelEligible | Whether the policy has a complete pure-premium outcome |

## Severity Completeness Status

| Status | Meaning |
|---|---|
| no_claim_no_severity | Both tables indicate zero claims |
| matched_positive_claims | Positive claim count agrees across both tables |
| claim_without_severity | `ClaimNb` is positive but no severity row exists |
| fewer_severity_rows_than_claimnb | Severity table contains fewer rows than `ClaimNb` |
| more_severity_rows_than_claimnb | Severity table contains more rows than `ClaimNb` |
| severity_without_frequency | Severity policy ID is absent from frequency data |

`severity_without_frequency` appears in the audit dataset but not in the
final policy-level dataset.

## Notes

- `IDpol` is an identifier and will not be used as a predictive feature.
- `Exposure` is important because policies were not all observed for the
  same amount of time.
- `AverageClaimAmount` is defined only for policies with at least one
  valid claim.
- The high-loss threshold will be determined after examining the claim
  amount distribution.