# Exploratory Data Analysis Findings

## Portfolio Overview

- Total policies: 678,013
- Total exposure-years: 358,499.45
- Total reported claims: 36,102
- Policies with at least one claim: 34,060
- Claim occurrence rate: 5.02%
- Claims per 100 exposure-years: 10.07
- Policies with complete claim-cost outcomes: 668,896
- Average complete claim severity: 2,265.58
- Complete pure premium: 169.28

## Claim Occurrence

Most policies did not have a reported claim during their exposure
period. The claim-occurrence outcome is therefore imbalanced.

Future classification models should not be evaluated using accuracy
alone. Recall, precision, PR-AUC, ROC-AUC and calibration will also be
considered.

## Claim Amount Distribution

Individual claim amounts are strongly right-skewed. Most claims are
relatively small compared with a limited number of very expensive
claims.

Large claims were retained because extreme insurance losses may be
legitimate. Modelling stages will examine suitable distributions and
transformations rather than deleting extreme observations
automatically.

## Area Analysis

- Highest observed area claim frequency:
  `F` with
  13.91 claims per 100 exposure-years.
- Lowest observed area claim frequency:
  `A` with
  8.17 claims per 100 exposure-years.

These are unadjusted comparisons. Area groups may differ in driver,
vehicle, exposure and geographic characteristics.

## Driver Age Analysis

The driver-age band with the highest observed pure premium was
`Under 21`, with an estimated complete claim
cost of 3,216.91 per exposure-year.

This is an unadjusted descriptive relationship and should not be
interpreted as a causal effect.

## Bonus-Malus Analysis

The bonus-malus band with the highest observed claim frequency was
`150+`, with
56.77 claims per 100 exposure-years.

The relationship will be examined further during statistical and
predictive modelling.

## Regional Analysis

Among regions with at least 1,000 complete exposure-years, the region
with the highest observed pure premium was
`R21`, with a pure premium of
406.51.

The minimum exposure rule was used to avoid emphasizing highly unstable
rates based on very small samples.

## Frequency, Severity and Pure Premium

The analysis distinguishes:

- claim occurrence: whether a policy had a claim;
- claim frequency: number of claims per exposure-year;
- claim severity: average claim cost;
- pure premium: total claim cost per exposure-year.

Segments with high claim frequency do not necessarily have the highest
average severity.

## Data Completeness Limitation

Claim frequency is available for all structurally valid policies, but
severity information is incomplete for some policies.

Pure-premium and severity summaries use model-eligibility indicators
created during data preparation. Results should therefore be
interpreted as estimates based on the complete analytical sample.

## Interpretation Limitations

The analysis is descriptive and does not establish causality.

Differences between groups may reflect confounding factors and
interactions among driver, vehicle and geographic characteristics.

The data represents a historical French insurance portfolio and should
not be presented as current Canadian insurance experience.
