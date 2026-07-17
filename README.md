# Insurance Claims Risk Intelligence

## Project Overview

This project develops an end-to-end insurance claims analytics system
for a fictional Canadian automobile insurance company called Prairie
Shield Insurance.

The project will combine Python, SQL, statistical modelling, machine
learning, financial analysis, forecasting, and Power BI to support
insurance claims and risk-management decisions.

## Business Problem

Prairie Shield Insurance needs a data-driven system for:

- monitoring claims performance;
- identifying high-risk claims and policies;
- predicting claim costs;
- measuring financial and operational KPIs;
- forecasting future monthly claim costs;
- communicating findings through interactive dashboards.

The predictive models will be used as decision-support tools. They will
not automatically approve or reject insurance claims.

## Project Objectives

The main objectives are to:

1. Build a reproducible data-cleaning pipeline.
2. Store and analyze insurance data using SQL.
3. Conduct exploratory and financial analysis.
4. Predict insurance claim severity.
5. Classify claims according to risk.
6. Forecast monthly claim costs.
7. Explain the factors influencing model predictions.
8. Build an interactive Power BI dashboard.
9. Produce a non-technical executive summary.

## Planned Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- statsmodels
- SQL
- Power BI
- DAX
- Power Query
- Microsoft Excel
- Git and GitHub

## Completed

- Selected the freMTPL2 motor-insurance dataset.
- Created a reproducible OpenML download script.
- Documented the dataset variables and limitations.
- Defined the policy-level modelling targets.
- Completed the initial data-quality inspection.
- Examined data types, missing values, duplicates, and numerical ranges.
- Validated policy identifiers and categorical values.
- Compared policy claim counts with claim-level severity records.
- Generated a reproducible data-quality summary.

## Upcoming

- Resolve or document cross-table inconsistencies.
- Create cleaned interim datasets.
- Aggregate claim amounts to the policy level.
- Create derived insurance variables.
- Build the first combined analytical dataset.

## Repository Structure

```text
insurance-risk-intelligence/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── database/
├── docs/
├── notebooks/
├── powerbi/
├── reports/
├── src/
└── tests/

