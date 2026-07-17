"""Prepare the freMTPL2 data for policy-level analysis.

This script:

1. Loads the raw frequency and severity datasets.
2. Standardizes data types and categorical text.
3. Flags structurally invalid records.
4. Aggregates claim amounts to the policy level.
5. Audits claim-count consistency across the two datasets.
6. Builds a policy-level analytical dataset.
7. Saves reproducible reports and documentation.

The raw data files are never modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

FREQUENCY_PATH = RAW_DATA_DIR / "freMTPL2freq.csv"
SEVERITY_PATH = RAW_DATA_DIR / "freMTPL2sev.csv"

FREQUENCY_CLEAN_PATH = (
    INTERIM_DATA_DIR / "frequency_clean.csv"
)

SEVERITY_CLEAN_PATH = (
    INTERIM_DATA_DIR / "severity_clean.csv"
)

SEVERITY_BY_POLICY_PATH = (
    INTERIM_DATA_DIR / "severity_by_policy.csv"
)

CLAIM_COUNT_AUDIT_PATH = (
    INTERIM_DATA_DIR / "claim_count_audit.csv"
)

POLICY_ANALYTICS_PATH = (
    PROCESSED_DATA_DIR / "policy_analytics.csv"
)

PREPARATION_SUMMARY_PATH = (
    REPORTS_DIR / "data_preparation_summary.csv"
)

AUDIT_SUMMARY_PATH = (
    REPORTS_DIR / "claim_count_audit_summary.csv"
)

FINDINGS_PATH = (
    DOCS_DIR / "data_preparation_findings.md"
)


FREQUENCY_NUMERIC_COLUMNS = [
    "IDpol",
    "ClaimNb",
    "Exposure",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "Density",
]

FREQUENCY_CATEGORICAL_COLUMNS = [
    "Area",
    "VehBrand",
    "VehGas",
    "Region",
]


def create_directories() -> None:
    """Create all output directories when they do not exist."""

    for directory in [
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        DOCS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw insurance datasets."""

    if not FREQUENCY_PATH.exists():
        raise FileNotFoundError(
            f"Frequency data not found: {FREQUENCY_PATH}"
        )

    if not SEVERITY_PATH.exists():
        raise FileNotFoundError(
            f"Severity data not found: {SEVERITY_PATH}"
        )

    frequency = pd.read_csv(FREQUENCY_PATH)
    severity = pd.read_csv(SEVERITY_PATH)

    return frequency, severity


def clean_frequency_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize the policy-frequency dataset."""

    cleaned = dataframe.copy()

    for column in FREQUENCY_NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    for column in FREQUENCY_CATEGORICAL_COLUMNS:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
        )

    cleaned["ValidPolicyID"] = (
        cleaned["IDpol"].notna()
        & cleaned["IDpol"].gt(0)
    )

    cleaned["ValidClaimCount"] = (
        cleaned["ClaimNb"].notna()
        & cleaned["ClaimNb"].ge(0)
        & np.isclose(cleaned["ClaimNb"] % 1, 0)
    )

    cleaned["ValidExposure"] = (
        cleaned["Exposure"].notna()
        & cleaned["Exposure"].gt(0)
    )

    cleaned["ValidVehicleAge"] = (
        cleaned["VehAge"].isna()
        | cleaned["VehAge"].ge(0)
    )

    cleaned["ValidDriverAge"] = (
        cleaned["DrivAge"].isna()
        | cleaned["DrivAge"].ge(0)
    )

    cleaned["ValidBonusMalus"] = (
        cleaned["BonusMalus"].isna()
        | cleaned["BonusMalus"].ge(0)
    )

    cleaned["ValidDensity"] = (
        cleaned["Density"].isna()
        | cleaned["Density"].gt(0)
    )

    cleaned["CorePolicyRecordValid"] = (
        cleaned["ValidPolicyID"]
        & cleaned["ValidClaimCount"]
        & cleaned["ValidExposure"]
    )

    cleaned["AnyPredictorValidityIssue"] = ~(
        cleaned["ValidVehicleAge"]
        & cleaned["ValidDriverAge"]
        & cleaned["ValidBonusMalus"]
        & cleaned["ValidDensity"]
    )

    cleaned["IDpol"] = cleaned["IDpol"].astype(
        "Int64"
    )

    cleaned["ClaimNb"] = cleaned["ClaimNb"].astype(
        "Int64"
    )

    return cleaned


def clean_severity_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize the claim-severity dataset."""

    cleaned = dataframe.copy()

    cleaned["IDpol"] = pd.to_numeric(
        cleaned["IDpol"],
        errors="coerce",
    )

    cleaned["ClaimAmount"] = pd.to_numeric(
        cleaned["ClaimAmount"],
        errors="coerce",
    )

    cleaned["ValidPolicyID"] = (
        cleaned["IDpol"].notna()
        & cleaned["IDpol"].gt(0)
    )

    cleaned["ValidClaimAmount"] = (
        cleaned["ClaimAmount"].notna()
        & cleaned["ClaimAmount"].gt(0)
    )

    cleaned["SeverityRecordValid"] = (
        cleaned["ValidPolicyID"]
        & cleaned["ValidClaimAmount"]
    )

    cleaned["IDpol"] = cleaned["IDpol"].astype(
        "Int64"
    )

    return cleaned


def select_valid_frequency_records(
    frequency: pd.DataFrame,
) -> pd.DataFrame:
    """Select policy records valid for core analysis."""

    valid_frequency = frequency.loc[
        frequency["CorePolicyRecordValid"]
    ].copy()

    valid_frequency["IDpol"] = (
        valid_frequency["IDpol"].astype("int64")
    )

    valid_frequency["ClaimNb"] = (
        valid_frequency["ClaimNb"].astype("int64")
    )

    if valid_frequency["IDpol"].duplicated().any():
        duplicate_count = int(
            valid_frequency["IDpol"].duplicated().sum()
        )

        raise ValueError(
            "The valid frequency data contains "
            f"{duplicate_count:,} duplicate policy IDs."
        )

    return valid_frequency


def select_valid_severity_records(
    severity: pd.DataFrame,
) -> pd.DataFrame:
    """Select claim records valid for monetary analysis."""

    valid_severity = severity.loc[
        severity["SeverityRecordValid"]
    ].copy()

    valid_severity["IDpol"] = (
        valid_severity["IDpol"].astype("int64")
    )

    return valid_severity


def aggregate_severity(
    severity: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate individual claims to one row per policy."""

    severity_by_policy = (
        severity.groupby(
            "IDpol",
            as_index=False,
        )
        .agg(
            SeverityRowCount=(
                "ClaimAmount",
                "size",
            ),
            SeverityTotalClaimAmount=(
                "ClaimAmount",
                "sum",
            ),
            SeverityAverageClaimAmount=(
                "ClaimAmount",
                "mean",
            ),
            SeverityMinimumClaimAmount=(
                "ClaimAmount",
                "min",
            ),
            SeverityMaximumClaimAmount=(
                "ClaimAmount",
                "max",
            ),
        )
    )

    return severity_by_policy


def create_claim_count_audit(
    frequency: pd.DataFrame,
    severity: pd.DataFrame,
) -> pd.DataFrame:
    """Compare ClaimNb with the number of severity records."""

    frequency_counts = frequency[
        ["IDpol", "ClaimNb"]
    ].copy()

    severity_counts = (
        severity.groupby("IDpol")
        .size()
        .rename("SeverityRowCount")
        .reset_index()
    )

    audit = frequency_counts.merge(
        severity_counts,
        on="IDpol",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    audit["ClaimNb"] = (
        audit["ClaimNb"]
        .fillna(0)
        .astype("int64")
    )

    audit["SeverityRowCount"] = (
        audit["SeverityRowCount"]
        .fillna(0)
        .astype("int64")
    )

    audit["ClaimCountDifference"] = (
        audit["ClaimNb"]
        - audit["SeverityRowCount"]
    )

    audit["ClaimCountMatch"] = (
        audit["ClaimNb"]
        .eq(audit["SeverityRowCount"])
    )

    conditions = [
        audit["_merge"].eq("right_only"),
        (
            audit["ClaimNb"].eq(0)
            & audit["SeverityRowCount"].eq(0)
        ),
        (
            audit["ClaimNb"].gt(0)
            & audit["SeverityRowCount"].eq(0)
        ),
        (
            audit["ClaimNb"].gt(0)
            & audit["ClaimNb"].eq(
                audit["SeverityRowCount"]
            )
        ),
        audit["SeverityRowCount"].lt(
            audit["ClaimNb"]
        ),
        audit["SeverityRowCount"].gt(
            audit["ClaimNb"]
        ),
    ]

    statuses = [
        "severity_without_frequency",
        "no_claim_no_severity",
        "claim_without_severity",
        "matched_positive_claims",
        "fewer_severity_rows_than_claimnb",
        "more_severity_rows_than_claimnb",
    ]

    audit["AuditStatus"] = np.select(
        conditions,
        statuses,
        default="unclassified",
    )

    return audit


def create_policy_analytics(
    frequency: pd.DataFrame,
    severity_by_policy: pd.DataFrame,
) -> pd.DataFrame:
    """Create the first policy-level analytical dataset."""

    policy = frequency.merge(
        severity_by_policy,
        on="IDpol",
        how="left",
        validate="one_to_one",
    )

    policy["SeverityRowCount"] = (
        policy["SeverityRowCount"]
        .fillna(0)
        .astype("int64")
    )

    policy["SeverityTotalClaimAmount"] = (
        policy["SeverityTotalClaimAmount"]
        .fillna(0.0)
    )

    policy["HasClaim"] = (
        policy["ClaimNb"].gt(0).astype("int8")
    )

    policy["HasSeverityRecord"] = (
        policy["SeverityRowCount"]
        .gt(0)
        .astype("int8")
    )

    policy["ClaimCountDifference"] = (
        policy["ClaimNb"]
        - policy["SeverityRowCount"]
    )

    policy["ClaimCountMatch"] = (
        policy["ClaimNb"]
        .eq(policy["SeverityRowCount"])
    )

    conditions = [
        (
            policy["ClaimNb"].eq(0)
            & policy["SeverityRowCount"].eq(0)
        ),
        (
            policy["ClaimNb"].gt(0)
            & policy["SeverityRowCount"].eq(0)
        ),
        (
            policy["ClaimNb"].gt(0)
            & policy["ClaimNb"].eq(
                policy["SeverityRowCount"]
            )
        ),
        policy["SeverityRowCount"].lt(
            policy["ClaimNb"]
        ),
        policy["SeverityRowCount"].gt(
            policy["ClaimNb"]
        ),
    ]

    statuses = [
        "no_claim_no_severity",
        "claim_without_severity",
        "matched_positive_claims",
        "fewer_severity_rows_than_claimnb",
        "more_severity_rows_than_claimnb",
    ]

    policy["SeverityCompletenessStatus"] = (
        np.select(
            conditions,
            statuses,
            default="unclassified",
        )
    )

    policy["AnnualizedClaimFrequency"] = (
        policy["ClaimNb"]
        / policy["Exposure"]
    )

    policy["ObservedAverageClaimAmount"] = (
        policy["SeverityTotalClaimAmount"]
        / policy["SeverityRowCount"].replace(
            0,
            np.nan,
        )
    )

    policy["CompleteTotalClaimAmount"] = (
        np.where(
            policy["ClaimCountMatch"],
            policy["SeverityTotalClaimAmount"],
            np.nan,
        )
    )

    severity_model_eligible = (
        policy["ClaimNb"].gt(0)
        & policy["ClaimCountMatch"]
        & policy[
            "ObservedAverageClaimAmount"
        ].notna()
    )

    policy["CompleteAverageClaimAmount"] = (
        np.where(
            severity_model_eligible,
            (
                policy[
                    "SeverityTotalClaimAmount"
                ]
                / policy["ClaimNb"]
            ),
            np.nan,
        )
    )

    policy["CompletePurePremium"] = (
        np.where(
            policy["ClaimCountMatch"],
            (
                policy[
                    "SeverityTotalClaimAmount"
                ]
                / policy["Exposure"]
            ),
            np.nan,
        )
    )

    policy["FrequencyModelEligible"] = True

    policy["SeverityModelEligible"] = (
        severity_model_eligible
    )

    policy["PurePremiumModelEligible"] = (
        policy["ClaimCountMatch"]
    )

    return policy


def create_audit_summary(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize cross-table consistency statuses."""

    summary = (
        audit["AuditStatus"]
        .value_counts(dropna=False)
        .rename_axis("AuditStatus")
        .reset_index(name="PolicyCount")
    )

    summary["Percentage"] = (
        summary["PolicyCount"]
        / len(audit)
        * 100
    )

    return summary


def create_preparation_summary(
    raw_frequency: pd.DataFrame,
    raw_severity: pd.DataFrame,
    cleaned_frequency: pd.DataFrame,
    cleaned_severity: pd.DataFrame,
    valid_frequency: pd.DataFrame,
    valid_severity: pd.DataFrame,
    audit: pd.DataFrame,
    policy: pd.DataFrame,
) -> pd.DataFrame:
    """Create a high-level preparation report."""

    mismatch_count = int(
        (~audit["ClaimCountMatch"]).sum()
    )

    unmatched_severity_ids = int(
        audit["AuditStatus"]
        .eq("severity_without_frequency")
        .sum()
    )

    frequency_policy_mismatches = int(
        (
            (~policy["ClaimCountMatch"])
        ).sum()
    )

    summary = pd.DataFrame(
        {
            "Measure": [
                "Raw frequency rows",
                "Raw severity rows",
                "Valid frequency rows",
                "Invalid core frequency rows",
                "Valid severity rows",
                "Invalid severity rows",
                "Cleaned frequency missing values",
                "Cleaned severity missing values",
                "Cross-table mismatched policy IDs",
                "Severity IDs without frequency record",
                "Frequency policies with count mismatch",
                "Final policy analytics rows",
                "Frequency-model eligible policies",
                "Severity-model eligible policies",
                "Pure-premium-model eligible policies",
                "Claims reported by ClaimNb",
                "Valid severity claim rows",
            ],
            "Value": [
                len(raw_frequency),
                len(raw_severity),
                len(valid_frequency),
                int(
                    (
                        ~cleaned_frequency[
                            "CorePolicyRecordValid"
                        ]
                    ).sum()
                ),
                len(valid_severity),
                int(
                    (
                        ~cleaned_severity[
                            "SeverityRecordValid"
                        ]
                    ).sum()
                ),
                int(
                    cleaned_frequency
                    .isna()
                    .sum()
                    .sum()
                ),
                int(
                    cleaned_severity
                    .isna()
                    .sum()
                    .sum()
                ),
                mismatch_count,
                unmatched_severity_ids,
                frequency_policy_mismatches,
                len(policy),
                int(
                    policy[
                        "FrequencyModelEligible"
                    ].sum()
                ),
                int(
                    policy[
                        "SeverityModelEligible"
                    ].sum()
                ),
                int(
                    policy[
                        "PurePremiumModelEligible"
                    ].sum()
                ),
                int(
                    valid_frequency[
                        "ClaimNb"
                    ].sum()
                ),
                len(valid_severity),
            ],
        }
    )

    return summary


def save_outputs(
    cleaned_frequency: pd.DataFrame,
    cleaned_severity: pd.DataFrame,
    severity_by_policy: pd.DataFrame,
    audit: pd.DataFrame,
    policy: pd.DataFrame,
    preparation_summary: pd.DataFrame,
    audit_summary: pd.DataFrame,
) -> None:
    """Save all prepared datasets and reports."""

    cleaned_frequency.to_csv(
        FREQUENCY_CLEAN_PATH,
        index=False,
    )

    cleaned_severity.to_csv(
        SEVERITY_CLEAN_PATH,
        index=False,
    )

    severity_by_policy.to_csv(
        SEVERITY_BY_POLICY_PATH,
        index=False,
    )

    audit.to_csv(
        CLAIM_COUNT_AUDIT_PATH,
        index=False,
    )

    policy.to_csv(
        POLICY_ANALYTICS_PATH,
        index=False,
    )

    preparation_summary.to_csv(
        PREPARATION_SUMMARY_PATH,
        index=False,
    )

    audit_summary.to_csv(
        AUDIT_SUMMARY_PATH,
        index=False,
    )


def create_findings_document(
    raw_frequency: pd.DataFrame,
    raw_severity: pd.DataFrame,
    valid_frequency: pd.DataFrame,
    valid_severity: pd.DataFrame,
    audit: pd.DataFrame,
    policy: pd.DataFrame,
    audit_summary: pd.DataFrame,
) -> None:
    """Generate a Markdown summary of preparation decisions."""

    mismatch_count = int(
        (~audit["ClaimCountMatch"]).sum()
    )

    frequency_mismatch_count = int(
        (~policy["ClaimCountMatch"]).sum()
    )

    unmatched_severity_count = int(
        audit["AuditStatus"]
        .eq("severity_without_frequency")
        .sum()
    )

    status_table_lines = [
        "| Audit status | Policy count | Percentage |",
        "|---|---:|---:|",
    ]

    for row in audit_summary.itertuples(
        index=False
    ):
        status_table_lines.append(
            f"| {row.AuditStatus} "
            f"| {row.PolicyCount:,} "
            f"| {row.Percentage:.3f}% |"
        )

    status_table = "\n".join(
        status_table_lines
    )

    findings = f"""# Data Preparation Findings

## Purpose

This stage created the first policy-level analytical dataset while
preserving the original raw files and documenting inconsistencies
between the frequency and severity tables.

## Input Data

- Raw frequency records: {len(raw_frequency):,}
- Raw severity records: {len(raw_severity):,}
- Valid policy records: {len(valid_frequency):,}
- Valid severity records: {len(valid_severity):,}

## Cross-Table Consistency

- Claims reported through `ClaimNb`: {int(valid_frequency["ClaimNb"].sum()):,}
- Valid rows in the severity table: {len(valid_severity):,}
- Cross-table mismatched policy IDs: {mismatch_count:,}
- Frequency policies with count mismatches: {frequency_mismatch_count:,}
- Severity policy IDs without frequency records: {unmatched_severity_count:,}

The total cross-table mismatch count includes severity policy IDs that
do not exist in the frequency table. These severity-only IDs cannot be
included in the final policy-level analytical dataset.

## Audit Status Summary

{status_table}

## Cleaning Decisions

### Raw data preservation

The raw CSV files were not modified. Cleaned and derived files were
saved separately in `data/interim` and `data/processed`.

### Duplicate severity rows

Duplicate severity rows were retained. The dataset does not provide a
unique claim identifier, so identical policy IDs and claim amounts
cannot safely be classified as accidental duplicates.

### Missing severity information

A policy with `ClaimNb > 0` but no matching severity record is treated
as having incomplete loss information rather than a zero-dollar loss.

### Partial severity information

When the number of severity rows does not match `ClaimNb`, the observed
claim amount is preserved in `SeverityTotalClaimAmount`, but complete
loss variables are set to missing.

### Zero-claim policies

A policy with zero reported claims and zero severity rows receives a
complete total claim amount and pure premium of zero.

### Outliers

No numerical outliers were removed or capped during this step.

## Analytical Samples

### Frequency modelling

All structurally valid policy records are eligible for claim-occurrence
and claim-frequency modelling.

### Severity modelling

Only policies with at least one claim and a matching number of severity
records are eligible for the severity model.

### Pure-premium modelling

Only policies whose `ClaimNb` agrees with the number of severity rows
are treated as having a complete pure-premium outcome.

## Output Dataset

The policy-level analytical dataset contains {len(policy):,} rows.

Each row represents one policy and contains:

- original policy characteristics;
- claim occurrence;
- annualized claim frequency;
- observed severity summaries;
- claim-count consistency flags;
- complete loss variables where available;
- model-eligibility indicators.

## Important Limitation

The severity table does not contain all claims reported in the
frequency table. Pure-premium and severity results must therefore be
interpreted using the completeness and eligibility indicators.
"""

    FINDINGS_PATH.write_text(
        findings,
        encoding="utf-8",
    )


def validate_outputs(
    policy: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """Run final structural validation checks."""

    if not policy["IDpol"].is_unique:
        raise ValueError(
            "The policy analytical dataset contains "
            "duplicate policy IDs."
        )

    if policy["Exposure"].le(0).any():
        raise ValueError(
            "The policy analytical dataset contains "
            "non-positive exposure."
        )

    incomplete_mask = ~policy[
        "ClaimCountMatch"
    ]

    if policy.loc[
        incomplete_mask,
        "CompleteTotalClaimAmount",
    ].notna().any():
        raise ValueError(
            "Incomplete policies contain complete "
            "claim amounts."
        )

    zero_claim_complete = (
        policy["ClaimNb"].eq(0)
        & policy["ClaimCountMatch"]
    )

    if not policy.loc[
        zero_claim_complete,
        "CompleteTotalClaimAmount",
    ].eq(0).all():
        raise ValueError(
            "Some complete zero-claim policies do not "
            "have a zero claim amount."
        )

    unclassified_count = int(
        audit["AuditStatus"]
        .eq("unclassified")
        .sum()
    )

    if unclassified_count > 0:
        raise ValueError(
            f"{unclassified_count:,} audit records "
            "were not classified."
        )


def main() -> None:
    """Run the complete data-preparation pipeline."""

    create_directories()

    print("Loading raw data...")
    raw_frequency, raw_severity = (
        load_raw_data()
    )

    print("Cleaning frequency data...")
    cleaned_frequency = clean_frequency_data(
        raw_frequency
    )

    print("Cleaning severity data...")
    cleaned_severity = clean_severity_data(
        raw_severity
    )

    valid_frequency = (
        select_valid_frequency_records(
            cleaned_frequency
        )
    )

    valid_severity = (
        select_valid_severity_records(
            cleaned_severity
        )
    )

    print("Aggregating severity records...")
    severity_by_policy = aggregate_severity(
        valid_severity
    )

    print("Creating claim-count audit...")
    audit = create_claim_count_audit(
        valid_frequency,
        valid_severity,
    )

    print("Creating policy analytical dataset...")
    policy = create_policy_analytics(
        valid_frequency,
        severity_by_policy,
    )

    audit_summary = create_audit_summary(
        audit
    )

    preparation_summary = (
        create_preparation_summary(
            raw_frequency=raw_frequency,
            raw_severity=raw_severity,
            cleaned_frequency=cleaned_frequency,
            cleaned_severity=cleaned_severity,
            valid_frequency=valid_frequency,
            valid_severity=valid_severity,
            audit=audit,
            policy=policy,
        )
    )

    print("Validating outputs...")
    validate_outputs(
        policy,
        audit,
    )

    print("Saving prepared datasets...")
    save_outputs(
        cleaned_frequency=cleaned_frequency,
        cleaned_severity=cleaned_severity,
        severity_by_policy=severity_by_policy,
        audit=audit,
        policy=policy,
        preparation_summary=preparation_summary,
        audit_summary=audit_summary,
    )

    create_findings_document(
        raw_frequency=raw_frequency,
        raw_severity=raw_severity,
        valid_frequency=valid_frequency,
        valid_severity=valid_severity,
        audit=audit,
        policy=policy,
        audit_summary=audit_summary,
    )

    print("\nData preparation completed successfully.")
    print(f"Policy analytics shape: {policy.shape}")
    print(
        "Cross-table mismatched IDs:",
        f"{int((~audit['ClaimCountMatch']).sum()):,}",
    )
    print(
        "Frequency-policy mismatches:",
        f"{int((~policy['ClaimCountMatch']).sum()):,}",
    )
    print(
        "Severity IDs without frequency record:",
        f"{int(audit['AuditStatus'].eq('severity_without_frequency').sum()):,}",
    )
    print(
        "Severity-model eligible policies:",
        f"{int(policy['SeverityModelEligible'].sum()):,}",
    )
    print(
        "Pure-premium-model eligible policies:",
        f"{int(policy['PurePremiumModelEligible'].sum()):,}",
    )

    print("\nAudit summary:")
    print(audit_summary.to_string(index=False))

    print(
        "\nPolicy dataset saved to:",
        POLICY_ANALYTICS_PATH,
    )


if __name__ == "__main__":
    main()