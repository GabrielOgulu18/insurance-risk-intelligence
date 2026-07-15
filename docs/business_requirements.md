# Business Requirements Document

## Project Title

Insurance Claims Risk Intelligence: Predictive Modelling, Financial
Forecasting, and Power BI Reporting

## Organization

Prairie Shield Insurance is a fictional Canadian automobile insurance
company created for this portfolio project.

## Business Problem

Prairie Shield Insurance needs a data-driven system for monitoring
claims performance, identifying high-risk claims and policies,
estimating expected claim costs, and forecasting future insurance
losses.

The system should support claims managers, financial analysts, risk
analysts, and senior management.

## Project Objective

The objective is to develop an insurance claims analytics system that
provides:

1. Historical claims reporting.
2. Financial and operational KPIs.
3. Claim-cost predictions.
4. Claim risk scores.
5. Monthly claim-cost forecasts.
6. Interactive Power BI dashboards.
7. Explainable business recommendations.

## Primary Stakeholders

| Stakeholder | Main Need |
|---|---|
| Claims Manager | Identify claims requiring additional review |
| Financial Analyst | Monitor costs, budgets, and variances |
| Risk Analyst | Understand expected losses and risk drivers |
| Senior Management | Review high-level KPIs and trends |

## Business Decisions Supported

The system will support:

- claim prioritization;
- financial planning;
- risk monitoring;
- pricing review;
- claims-process improvement;
- resource allocation.

The risk model will support human decision-making. It will not
automatically approve or reject claims.

## Business Questions

### Claims Performance

- How many claims are reported each month?
- What is the total and average claim amount?
- Which policy segments generate the largest losses?
- Are claims increasing over time?
- Are claims affected by seasonality?

### Financial Performance

- What is the loss ratio?
- How do actual claims compare with budgeted claims?
- Which segments contribute most to total claim costs?

### Claims Operations

- How long does it take to process a claim?
- Which claims remain open for unusually long periods?
- Which claim types take the longest to process?

### Risk Analysis

- Which factors are associated with expensive claims?
- Which claims are likely to require additional review?
- What is the predicted cost of a claim?
- Which factors contribute most to a claim risk score?

### Forecasting

- What are total claim costs likely to be over the next three to six
  months?
- Which policy types are expected to drive future claim costs?

## Key Performance Indicators

The primary KPIs are:

- total claim cost;
- claim count;
- average claim severity;
- claim frequency;
- loss ratio;
- claim-cost variance;
- average processing time;
- open claim count;
- high-value claim count;
- high-risk claim rate.

## Project Scope

### Included

- customer, policy, claim, and financial data;
- data-quality analysis;
- SQL database development;
- exploratory analysis;
- financial KPI analysis;
- claim-cost prediction;
- claim-risk classification;
- monthly claim-cost forecasting;
- Power BI dashboard development;
- model explainability;
- executive reporting.

### Excluded

- automatic approval or rejection of real claims;
- legally valid insurance pricing;
- use of personally identifiable information;
- replacement of actuarial or claims professionals;
- production deployment.

## Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | Load customer, policy, claim, and financial data |
| FR-02 | Detect missing, duplicate, and invalid records |
| FR-03 | Link customers, policies, and claims |
| FR-04 | Calculate financial and claims KPIs |
| FR-05 | Analyze claims by date, region, and policy type |
| FR-06 | Predict claim cost |
| FR-07 | Assign a risk score to each claim |
| FR-08 | Forecast monthly claim costs |
| FR-09 | Compare actual, budgeted, and forecast costs |
| FR-10 | Provide dashboard filters and drill-down analysis |
| FR-11 | Explain the main model drivers |
| FR-12 | Produce a non-technical executive summary |

## Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-01 | The analysis must be reproducible |
| NFR-02 | The code must be clearly organized |
| NFR-03 | Dashboard labels must be clear |
| NFR-04 | No personal information may be exposed |
| NFR-05 | Model results must be explainable |
| NFR-06 | Multiple evaluation metrics must be used |
| NFR-07 | Assumptions and limitations must be documented |
| NFR-08 | Data-quality checks must run before modelling |
| NFR-09 | Synthetic data must be clearly identified |
| NFR-10 | Setup and reproduction instructions must be provided |

## Responsible Use

This model is a portfolio demonstration and decision-support tool.

Risk scores should not be used to automatically reject claims or make
consequential decisions about customers. Predictions should be reviewed
by a qualified human, and model limitations should be clearly
communicated.