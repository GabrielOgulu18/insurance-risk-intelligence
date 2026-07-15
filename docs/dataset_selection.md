# Dataset Selection

## Selected Dataset

The project uses the French Motor Third-Party Liability Claims
dataset, commonly referred to as freMTPL2.

The dataset consists of two related tables:

1. `freMTPL2freq`, containing policy characteristics and claim counts.
2. `freMTPL2sev`, containing individual claim amounts and their
   corresponding policy identifiers.

## Data Sources

The datasets are downloaded from OpenML using scikit-learn.

| Dataset | OpenML Data ID |
|---|---:|
| freMTPL2freq | 41214 |
| freMTPL2sev | 41215 |

The file `src/download_data.py` provides a reproducible method for
downloading the data.

## Unit of Analysis

The primary unit of analysis is an automobile insurance policy.

Each row in the frequency dataset represents one policy. A policy may
have zero, one, or multiple claims.

Each row in the severity dataset represents one recorded claim. A
policy may therefore appear multiple times in the severity dataset.

The two tables are connected using `IDpol`.

## Reasons for Selection

The dataset was selected because it contains:

- insurance policy identifiers;
- policy exposure;
- claim counts;
- individual claim amounts;
- driver characteristics;
- vehicle characteristics;
- geographic characteristics;
- variables suitable for statistical and machine-learning models.

The dataset supports:

- claim occurrence modelling;
- claim-frequency modelling;
- claim-severity modelling;
- expected-loss modelling;
- high-loss policy classification;
- insurance portfolio analysis.

## Main Modelling Targets

The initial modelling targets will be:

1. `HasClaim`: whether a policy produced at least one claim.
2. `Frequency`: number of claims per unit of exposure.
3. `AverageClaimAmount`: average claim amount for policies with claims.
4. `PurePremium`: total claim amount per unit of exposure.
5. `HighLossPolicy`: whether a policy belongs to a high-loss group.

The exact definition of a high-loss policy will be selected after
exploratory data analysis.

## Variables Not Available

The original data does not include:

- customer names;
- claim dates;
- policy start dates;
- claim-processing time;
- claim status;
- fraud labels;
- earned premiums;
- operating expenses;
- budgets;
- detailed accident descriptions.

The project will not claim to predict insurance fraud.

Any synthetic financial or time variables added later will be clearly
identified and documented.

## Geographic and Temporal Limitations

The data comes from a French motor-insurance portfolio rather than a
Canadian insurer.

The dataset represents an earlier historical period and should not be
used to draw conclusions about current Canadian insurance customers.

Prairie Shield Insurance is a fictional organization used only to
frame the business problem.

## Privacy

Policy identifiers are anonymous. The dataset does not contain names,
addresses, telephone numbers, or other direct personal identifiers.

The analysis will still follow privacy-conscious practices and avoid
attempting to identify policyholders.

## Reproducibility

Raw data files are not committed to GitHub.

Users can reproduce the raw-data directory by installing the project
requirements and running:

```bash
python src/download_data.py