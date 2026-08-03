# Claim-Frequency Model

## Objective

The frequency model estimates the expected number of claims per
exposure-year.

The modelling target is:

`ClaimNb / Exposure`

Exposure is used as the sample weight. Predicted policy claim counts are
calculated by multiplying predicted annualized frequency by policy
exposure.

## Data Splits

| Sample     |   Policies |   Exposure |   Claims |   ZeroClaimPolicies |   OneClaimPolicies |   MultipleClaimPolicies |   ObservedFrequency |   ClaimsPer100ExposureYears |
|:-----------|-----------:|-----------:|---------:|--------------------:|-------------------:|------------------------:|--------------------:|----------------------------:|
| Training   |     406807 |   215189   |    21669 |              386371 |              19306 |                    1130 |            0.100697 |                     10.0697 |
| Validation |     135603 |    71714   |     7224 |              128791 |               6436 |                     376 |            0.100733 |                     10.0733 |
| Test       |     135603 |    71596.2 |     7209 |              128791 |               6436 |                     376 |            0.10069  |                     10.069  |

The split was stratified using zero-claim, one-claim, and multiple-claim
policy categories.

## Models

### Constant Frequency Baseline

The baseline assigns every policy the training portfolio's overall
claim frequency.

### Poisson Regression

A regularized log-linear Poisson regression was fitted using numerical
standardization and categorical one-hot encoding.

The model completed 139 optimization iterations.

### Histogram Gradient Boosting

The nonlinear model used Poisson loss and native categorical-variable
handling.

It completed 145 boosting iterations.

## Validation Results

| Model                       |   ExposureWeightedPoissonDeviance |    D2Poisson |   ObservedClaims |   PredictedClaims |   PredictedToObservedClaimsRatio |
|:----------------------------|----------------------------------:|-------------:|-----------------:|------------------:|---------------------------------:|
| Constant frequency baseline |                          0.625622 | -2.06217e-08 |             7224 |           7221.42 |                         0.999642 |
| Poisson regression          |                          0.60705  |  0.0296864   |             7224 |           7254.92 |                         1.00428  |
| Histogram gradient boosting |                          0.57623  |  0.078949    |             7224 |           7205.7  |                         0.997467 |

The preferred model was selected using the lowest validation
exposure-weighted Poisson deviance.

**Selected model:** `Histogram gradient boosting`

## Test Results

| Model                       |   ExposureWeightedPoissonDeviance |    D2Poisson |   CountMAE |   CountRMSE |   ObservedClaims |   PredictedClaims |   PredictedToObservedClaimsRatio |
|:----------------------------|----------------------------------:|-------------:|-----------:|------------:|-----------------:|------------------:|---------------------------------:|
| Constant frequency baseline |                          0.622283 | -9.63293e-10 |  0.0998043 |    0.238398 |             7209 |           7209.56 |                          1.00008 |
| Poisson regression          |                          0.601274 |  0.0337604   |  0.0987534 |    0.236929 |             7209 |           7235.9  |                          1.00373 |
| Histogram gradient boosting |                          0.56676  |  0.0892241   |  0.0959158 |    0.234158 |             7209 |           7224.46 |                          1.00214 |

For the selected model:

- Exposure-weighted Poisson deviance:
  0.566760
- Poisson D2:
  0.0892
- Observed claims:
  7,209
- Predicted claims:
  7,224.46
- Predicted-to-observed claims ratio:
  1.0021
- Observed claim frequency:
  0.100690
- Predicted claim frequency:
  0.100906
- Pearson dispersion diagnostic:
  2.0092

The diagnostic is above one, indicating that residual variation remains larger than the basic conditional Poisson variance assumption.

The Pearson value is used as an empirical diagnostic rather than a
formal hypothesis test, particularly for the nonlinear model.

## Risk-Decile Calibration

The lowest predicted-frequency decile had:

- Observed frequency:
  0.038736
- Predicted frequency:
  0.039810

The highest predicted-frequency decile had:

- Observed frequency:
  0.386494
- Predicted frequency:
  0.372092

A useful model should generally assign higher observed claim frequencies
to higher predicted-risk deciles.

## Interpretation

The frequency model estimates expected claim counts, not whether a claim
is certain to occur.

For example, an expected count of 0.10 means approximately one expected
claim per ten comparable exposure-years, not that a specific policy will
experience one-tenth of a claim.

## Limitations

- The data represents a historical French motor-insurance portfolio.
- The model assumes historical relationships are informative for the
  evaluation sample.
- The Poisson assumption may not capture all remaining variation.
- Segment-level deviations can remain even when overall calibration is
  strong.
- Predictive relationships should not be interpreted as causal.
- The model should not be the sole basis for consequential insurance
  decisions.

## Next Steps

The next modelling stage will estimate claim severity using complete
positive-claim policies.

Frequency and severity predictions will later be combined:

`Expected claim frequency × Expected claim severity`

to estimate expected claim cost per exposure-year.
