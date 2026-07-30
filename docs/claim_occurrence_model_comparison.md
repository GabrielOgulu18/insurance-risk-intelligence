# Claim-Occurrence Model Comparison

## Objective

This analysis compares the Step 8 logistic-regression model with a
histogram gradient-boosting classifier.

The comparison uses the same training, validation, and test samples as
the original model.

## Models

### Logistic Regression

The logistic model provides a transparent statistical baseline with
additive effects on the log-odds scale.

### Histogram Gradient Boosting

The gradient-boosting model can represent nonlinear relationships and
interactions among driver, vehicle, geographic, and exposure variables.

The fitted model completed 194 boosting iterations.

## Validation Results

| Model                       |   Threshold |   Precision |   Recall |   F2Score |   ROCAUC |     PRAUC |   BrierScore |
|:----------------------------|------------:|------------:|---------:|----------:|---------:|----------:|-------------:|
| Logistic regression         |   0.0509746 |   0.070878  | 0.619642 |  0.243142 | 0.635477 | 0.0900793 |    0.047054  |
| Histogram gradient boosting |   0.057217  |   0.0970458 | 0.55505  |  0.285535 | 0.70549  | 0.148326  |    0.0455265 |

The preferred model was selected using validation PR-AUC rather than
test performance.

**Selected model:** `Histogram gradient boosting`

Each model's F2 classification threshold was also selected using the
validation sample.

## Test Results

| Model                       |   Threshold |   Precision |   Recall |   Specificity |   F2Score |   ROCAUC |     PRAUC |   BrierScore |
|:----------------------------|------------:|------------:|---------:|--------------:|----------:|---------:|----------:|-------------:|
| Logistic regression         |   0.0509746 |   0.071419  | 0.624486 |      0.570545 |  0.245012 | 0.640154 | 0.0913082 |    0.0470114 |
| Histogram gradient boosting |   0.057217  |   0.0970844 | 0.55872  |      0.725159 |  0.286376 | 0.707009 | 0.144029  |    0.0455956 |

For the selected model:

- Threshold: 0.057217
- Precision: 9.71%
- Recall: 55.87%
- Specificity: 72.52%
- F2-score: 0.2864
- ROC-AUC: 0.7070
- PR-AUC: 0.1440
- Brier score: 0.04560

## Review-Capacity Analysis

When the highest-risk 10% of test policies were selected:

- Policies selected: 13,561
- Claims identified: 2,043
- Claim capture rate: 29.99%
- Precision: 15.07%
- Lift over random selection: 3.00

Capacity analysis may be more operationally useful than relying on one
fixed classification threshold.

## Permutation Importance

The five leading raw features were:

- `BonusMalus`
- `LogExposure`
- `VehAge`
- `VehBrand`
- `VehPower`

Permutation importance measures predictive reliance rather than
causation.

Correlated features may share importance, so a low value does not prove
that a variable has no relationship with claims.

## Subgroup Analysis

Subgroup evaluation was performed for:

- area;
- fuel type;
- driver-age band.

Groups with fewer than 1,000 policies or fewer than
30 claims were excluded from comparative
metrics.

Across eligible groups, recall ranged from approximately
46.18% to 77.05%.

Differences between groups should be investigated before the model is
used operationally. They may reflect sample size, claim prevalence,
data limitations, or differences in model fit.

## Limitations

- The data represents a historical French motor-insurance portfolio.
- Model performance does not establish causal relationships.
- The occurrence target is strongly imbalanced.
- Thresholds depend on the assumed relative importance of false
  positives and false negatives.
- Predicted probabilities may require additional calibration.
- Subgroup results do not by themselves establish fairness or unfairness.
- The model should support analysis rather than automate consequential
  insurance decisions.

## Recommended Model Use

The selected model may be used for:

- portfolio segmentation;
- risk-ranking research;
- identifying groups for further analysis;
- comparing expected and observed claim occurrence.

It should not be presented as a fraud model or used as the sole basis
for individual insurance decisions.
