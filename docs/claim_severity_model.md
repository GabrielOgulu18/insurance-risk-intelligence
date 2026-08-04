# Claim-Severity Model

## Objective

The severity model estimates average claim amount conditional on a
policy having complete positive claim information.

The target is:

`CompleteTotalClaimAmount / ClaimNb`

Policies are weighted by `ClaimNb`, so policies representing multiple
claims contribute proportionally more information.

## Data Splits

| Sample     |   Policies |   Claims |   TotalClaimAmount |   WeightedMeanSeverity |   MedianPolicyAverageSeverity |   MaximumPolicyAverageSeverity |
|:-----------|-----------:|---------:|-------------------:|-----------------------:|------------------------------:|-------------------------------:|
| Training   |      14965 |    15857 |        3.5964e+07  |                2268.02 |                          1172 |                     4.0754e+06 |
| Validation |       4989 |     5307 |        1.24138e+07 |                2339.14 |                          1172 |                702093          |
| Test       |       4989 |     5279 |        1.15308e+07 |                2184.28 |                          1172 |                774412          |

Severity deciles were used for stratification so that the highly skewed
target distribution remained broadly similar across training,
validation, and test samples.

## Models

### Constant Severity Baseline

The baseline predicts the training sample's claim-weighted mean severity
for every policy.

### Gamma Regression

A regularized Gamma generalized linear model was fitted using a log
link.

Numerical predictors were standardized and categorical predictors were
one-hot encoded.

The model completed 193 optimization iterations.

### Histogram Gradient Boosting

The nonlinear model used Gamma loss and native categorical-variable
handling.

It completed 47 boosting iterations.

## Validation Results

| Model                       |   WeightedGammaDeviance |      D2Gamma |   ObservedClaimAmount |   PredictedClaimAmount |   PredictedToObservedAmountRatio |
|:----------------------------|------------------------:|-------------:|----------------------:|-----------------------:|---------------------------------:|
| Constant severity baseline  |                 1.75563 | -0.000548983 |           1.24138e+07 |            1.20364e+07 |                         0.969595 |
| Gamma regression            |                 1.76344 | -0.00500091  |           1.24138e+07 |            1.14807e+07 |                         0.92483  |
| Histogram gradient boosting |                 1.85258 | -0.0558006   |           1.24138e+07 |            9.42645e+06 |                         0.759351 |

The preferred model was selected using the lowest validation
claim-weighted Gamma deviance.

**Selected model:** `Constant severity baseline`

## Test Results

| Model                       |   WeightedGammaDeviance |      D2Gamma |   WeightedMAE |   WeightedRMSE |   ObservedClaimAmount |   PredictedClaimAmount |   PredictedToObservedAmountRatio |
|:----------------------------|------------------------:|-------------:|--------------:|---------------:|----------------------:|-----------------------:|---------------------------------:|
| Constant severity baseline  |                 1.6477  | -0.000848987 |       2172.02 |        14083.7 |           1.15308e+07 |            1.19729e+07 |                         1.03834  |
| Gamma regression            |                 1.82281 | -0.107218    |       2149.5  |        14112.9 |           1.15308e+07 |            1.15997e+07 |                         1.00597  |
| Histogram gradient boosting |                 1.82255 | -0.107059    |       1851.89 |        14106.3 |           1.15308e+07 |            9.45973e+06 |                         0.820386 |

For the selected model:

- Claim-weighted Gamma deviance:
  1.647696
- Gamma D2:
  -0.0008
- Weighted MAE:
  2,172.02
- Weighted RMSE:
  14,083.72
- Observed claim amount:
  11,530,829.88
- Predicted claim amount:
  11,972,883.18
- Predicted-to-observed amount ratio:
  1.0383
- Observed mean severity:
  2,184.28
- Predicted mean severity:
  2,268.02

## Risk-Decile Calibration

The lowest predicted-severity decile had:

- Observed severity:
  1,899.44
- Predicted severity:
  2,268.02

The highest predicted-severity decile had:

- Observed severity:
  1,982.15
- Predicted severity:
  2,268.02

The top 1% of test policies accounted for approximately
36.36% of the observed test claim amount.

## Baseline Comparison

Neither Gamma regression nor histogram gradient boosting improved upon
the constant claim-weighted mean severity baseline on the validation
sample.

The constant baseline also achieved the lowest Gamma deviance on the
test sample. This indicates that the available policy-level driver,
vehicle and geographic variables provide limited predictive information
about realized claim severity.

The trained models remain useful methodological benchmarks, but the
constant severity estimate is retained as the validation-selected model
for the current two-part loss-cost analysis.

The severity result differs from the frequency result. Policy-level
characteristics provided meaningful information about how often claims
occurred, but much less information about how financially severe a claim
became after occurrence.

## Interpretation

The severity model is conditional on having a claim.

It does not predict whether a policy will have a claim and should not be
applied directly to zero-claim policies as an expected policy cost.

Expected loss cost will later be estimated by combining:

`Predicted claim frequency × Predicted claim severity`

## Extreme Claims

Large claims were retained.

They may represent legitimate insurance losses and are financially
important. The Gamma loss and logarithmic-link models are designed for
positive, right-skewed targets, but extreme losses can still produce
substantial prediction error.

The largest-error report is included for audit and sensitivity analysis,
not as an automatic removal list.

## Limitations

- Severity is available only for complete positive-claim policies.
- The severity sample is much smaller than the frequency sample.
- Large claims create substantial random variation.
- The dataset does not contain detailed accident or damage information.
- Group-level calibration can differ from overall calibration.
- Predictive associations should not be interpreted as causal.
- The model should not be the sole basis for consequential insurance
  decisions.

## Next Step

Frequency and severity predictions will be combined to estimate expected
claim cost per exposure-year.

The two-part estimate will then be compared with a direct Tweedie
pure-premium model.
