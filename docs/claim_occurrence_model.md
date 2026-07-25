# Claim-Occurrence Model

## Objective

The model estimates the probability that a policy reports at least one
claim during its observed exposure period.

The target is `HasClaim`.

## Modelling Sample

| Sample | Rows | Claims | Claim rate |
|---|---:|---:|---:|
| Training | 406,807 | 20,436 | 5.02% |
| Validation | 135,603 | 6,812 | 5.02% |
| Test | 135,603 | 6,812 | 5.02% |

The samples were created using stratified random splitting so that the
claim rate remained similar across training, validation, and test data.

## Predictors

The model uses:

- logarithm of exposure;
- vehicle power;
- vehicle age;
- driver age;
- bonus-malus score;
- logarithm of population density;
- area;
- vehicle brand;
- fuel type;
- region.

Policy identifiers, claim counts, claim amounts, severity fields, and
other post-claim variables were excluded to prevent data leakage.

## Model

The first statistical model is logistic regression with L2
regularization.

Numerical variables are median-imputed and standardized. Categorical
variables are most-frequent-imputed and one-hot encoded.

## Class Imbalance

Only a small percentage of policies have claims. A model predicting no
claim for every policy can therefore obtain high accuracy while having
zero claim recall.

Accuracy is not used as the main selection criterion.

## Threshold Selection

The classification threshold was selected using only the validation
sample.

The selected threshold was:

`0.050975`

It maximized the F2-score, which gives recall greater importance than
precision.

This reflects a screening objective where failing to identify a policy
that has a claim may be more costly than reviewing an additional
policy.

## Test Performance

At the default threshold of 0.50:

- Precision: 0.600
- Recall: 0.000
- F2-score: 0.001

At the validation-selected threshold:

- Precision: 0.071
- Recall: 0.624
- Specificity: 0.571
- F1-score: 0.128
- F2-score: 0.245
- ROC-AUC: 0.640
- PR-AUC: 0.091
- Brier score: 0.04701
- False positives: 55,310
- False negatives: 2,558

## Strongest Positive Coefficients

| Feature | Coefficient | Odds ratio |
|---|---:|---:|
| numeric__LogExposure | 0.4380 | 1.5497 |
| numeric__BonusMalus | 0.3193 | 1.3762 |
| categorical__Region_R74 | 0.2739 | 1.3151 |
| categorical__Region_R53 | 0.2646 | 1.3030 |
| categorical__Region_R24 | 0.1987 | 1.2198 |
| categorical__Region_R82 | 0.1866 | 1.2051 |
| categorical__Region_R21 | 0.1595 | 1.1729 |
| categorical__Region_R22 | 0.1584 | 1.1716 |
| numeric__DrivAge | 0.1418 | 1.1523 |
| categorical__Region_R25 | 0.1372 | 1.1471 |

## Strongest Negative Coefficients

| Feature | Coefficient | Odds ratio |
|---|---:|---:|
| categorical__Region_R23 | -0.3319 | 0.7176 |
| categorical__Region_R83 | -0.2533 | 0.7762 |
| numeric__VehAge | -0.2145 | 0.8069 |
| categorical__Region_R73 | -0.1728 | 0.8413 |
| categorical__Region_R91 | -0.1373 | 0.8717 |
| categorical__Region_R72 | -0.1080 | 0.8976 |
| categorical__Region_R43 | -0.1042 | 0.9011 |
| categorical__Region_R31 | -0.0872 | 0.9165 |
| categorical__Region_R41 | -0.0630 | 0.9389 |
| categorical__VehBrand_B6 | -0.0580 | 0.9436 |

Positive coefficients are associated with higher estimated claim odds,
while negative coefficients are associated with lower estimated claim
odds, holding the other model variables constant.

Numerical variables were standardized, so their coefficients represent
a one-standard-deviation change. Categorical coefficients are relative
to the category omitted by one-hot encoding.

Coefficients describe statistical associations and should not be
interpreted as causal effects.

## Limitations

- The data represents a historical French insurance portfolio.
- The target is strongly imbalanced.
- The model predicts historical claim occurrence, not fraud.
- Exposure is known for the historical observation period.
- Logistic regression assumes an additive linear relationship on the
  log-odds scale.
- Interactions and nonlinear effects are not yet included.
- A lower classification threshold increases recall but also increases
  false positives.
- The model should support human analysis rather than automate
  consequential insurance decisions.

## Next Modelling Actions

Future work will:

1. compare logistic regression with tree-based models;
2. investigate nonlinear and interaction effects;
3. examine subgroup performance;
4. assess probability calibration more deeply;
5. build a claim-frequency model that explicitly accounts for exposure.
