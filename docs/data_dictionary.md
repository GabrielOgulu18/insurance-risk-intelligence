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

## Notes

- `IDpol` is an identifier and will not be used as a predictive feature.
- `Exposure` is important because policies were not all observed for the
  same amount of time.
- `AverageClaimAmount` is defined only for policies with at least one
  valid claim.
- The high-loss threshold will be determined after examining the claim
  amount distribution.