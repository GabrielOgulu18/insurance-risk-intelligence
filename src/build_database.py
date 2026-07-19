"""Build the SQLite database for the insurance analytics project."""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
DATABASE_DIR = PROJECT_ROOT / "database"

POLICY_PATH = (
    PROCESSED_DATA_DIR / "policy_analytics.csv"
)

SEVERITY_PATH = (
    INTERIM_DATA_DIR / "severity_clean.csv"
)

AUDIT_PATH = (
    INTERIM_DATA_DIR / "claim_count_audit.csv"
)

SCHEMA_PATH = DATABASE_DIR / "schema.sql"

DATABASE_PATH = (
    DATABASE_DIR / "insurance_analytics.sqlite"
)


POLICY_COLUMNS = [
    "IDpol",
    "ClaimNb",
    "Exposure",
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
    "HasClaim",
    "HasSeverityRecord",
    "SeverityRowCount",
    "SeverityTotalClaimAmount",
    "SeverityAverageClaimAmount",
    "SeverityMinimumClaimAmount",
    "SeverityMaximumClaimAmount",
    "ClaimCountDifference",
    "ClaimCountMatch",
    "SeverityCompletenessStatus",
    "AnnualizedClaimFrequency",
    "ObservedAverageClaimAmount",
    "CompleteTotalClaimAmount",
    "CompleteAverageClaimAmount",
    "CompletePurePremium",
    "FrequencyModelEligible",
    "SeverityModelEligible",
    "PurePremiumModelEligible",
    "AnyPredictorValidityIssue",
]


BOOLEAN_POLICY_COLUMNS = [
    "HasClaim",
    "HasSeverityRecord",
    "ClaimCountMatch",
    "FrequencyModelEligible",
    "SeverityModelEligible",
    "PurePremiumModelEligible",
    "AnyPredictorValidityIssue",
]


def convert_to_boolean_integer(
    series: pd.Series,
) -> pd.Series:
    """Convert common Boolean values to SQLite-compatible integers."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8")

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
            }
        )
    )

    if mapped.isna().any():
        bad_values = sorted(
            series.loc[mapped.isna()]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Could not convert Boolean values: "
            f"{bad_values}"
        )

    return mapped.astype("int8")


def load_source_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load the prepared policy, severity, and audit datasets."""

    required_paths = [
        POLICY_PATH,
        SEVERITY_PATH,
        AUDIT_PATH,
        SCHEMA_PATH,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        missing_text = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Required files are missing:\n"
            f"{missing_text}"
        )

    policy = pd.read_csv(
        POLICY_PATH,
        low_memory=False,
    )

    severity = pd.read_csv(
        SEVERITY_PATH,
        low_memory=False,
    )

    audit = pd.read_csv(
        AUDIT_PATH,
        low_memory=False,
    )

    return policy, severity, audit


def prepare_policy_table(
    policy: pd.DataFrame,
) -> pd.DataFrame:
    """Select and standardize policy columns for SQLite."""

    missing_columns = (
        set(POLICY_COLUMNS)
        - set(policy.columns)
    )

    if missing_columns:
        raise ValueError(
            "Policy data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    output = policy[POLICY_COLUMNS].copy()

    output["IDpol"] = pd.to_numeric(
        output["IDpol"],
        errors="raise",
    ).astype("int64")

    output["ClaimNb"] = pd.to_numeric(
        output["ClaimNb"],
        errors="raise",
    ).astype("int64")

    output["SeverityRowCount"] = pd.to_numeric(
        output["SeverityRowCount"],
        errors="raise",
    ).astype("int64")

    output["ClaimCountDifference"] = pd.to_numeric(
        output["ClaimCountDifference"],
        errors="raise",
    ).astype("int64")

    for column in BOOLEAN_POLICY_COLUMNS:
        output[column] = convert_to_boolean_integer(
            output[column]
        )

    if not output["IDpol"].is_unique:
        raise ValueError(
            "Policy data contains duplicate policy IDs."
        )

    return output


def prepare_claim_tables(
    severity: pd.DataFrame,
    policy_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate matched claims from orphan claims."""

    required_columns = {
        "IDpol",
        "ClaimAmount",
        "SeverityRecordValid",
    }

    missing_columns = (
        required_columns - set(severity.columns)
    )

    if missing_columns:
        raise ValueError(
            "Severity data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    severity = severity.copy()

    severity["SeverityRecordValid"] = (
        convert_to_boolean_integer(
            severity["SeverityRecordValid"]
        )
    )

    severity["IDpol"] = pd.to_numeric(
        severity["IDpol"],
        errors="coerce",
    )

    severity["ClaimAmount"] = pd.to_numeric(
        severity["ClaimAmount"],
        errors="coerce",
    )

    valid_severity = severity.loc[
        severity["SeverityRecordValid"].eq(1)
    ].copy()

    valid_severity["IDpol"] = (
        valid_severity["IDpol"].astype("int64")
    )

    matched_mask = valid_severity[
        "IDpol"
    ].isin(policy_ids)

    matched_claims = valid_severity.loc[
        matched_mask,
        ["IDpol", "ClaimAmount"],
    ].copy()

    matched_claims.insert(
        0,
        "ClaimID",
        np.arange(
            1,
            len(matched_claims) + 1,
        ),
    )

    orphan_claims = valid_severity.loc[
        ~matched_mask,
        ["IDpol", "ClaimAmount"],
    ].copy()

    orphan_claims.insert(
        0,
        "OrphanClaimID",
        np.arange(
            1,
            len(orphan_claims) + 1,
        ),
    )

    return matched_claims, orphan_claims


def prepare_issue_table(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Create the claim-count issue table."""

    required_columns = {
        "IDpol",
        "ClaimNb",
        "SeverityRowCount",
        "ClaimCountDifference",
        "ClaimCountMatch",
        "AuditStatus",
        "_merge",
    }

    missing_columns = (
        required_columns - set(audit.columns)
    )

    if missing_columns:
        raise ValueError(
            "Audit data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    audit = audit.copy()

    audit["ClaimCountMatch"] = (
        convert_to_boolean_integer(
            audit["ClaimCountMatch"]
        )
    )

    issues = audit.loc[
        audit["ClaimCountMatch"].eq(0),
        [
            "IDpol",
            "ClaimNb",
            "SeverityRowCount",
            "ClaimCountDifference",
            "ClaimCountMatch",
            "AuditStatus",
            "_merge",
        ],
    ].copy()

    issues = issues.rename(
        columns={
            "_merge": "SourceStatus",
        }
    )

    integer_columns = [
        "IDpol",
        "ClaimNb",
        "SeverityRowCount",
        "ClaimCountDifference",
    ]

    for column in integer_columns:
        issues[column] = pd.to_numeric(
            issues[column],
            errors="raise",
        ).astype("int64")

    if not issues["IDpol"].is_unique:
        raise ValueError(
            "Claim-count issue table contains "
            "duplicate policy IDs."
        )

    return issues


def initialize_database(
    connection: sqlite3.Connection,
) -> None:
    """Create the SQLite schema."""

    schema_sql = SCHEMA_PATH.read_text(
        encoding="utf-8"
    )

    connection.executescript(schema_sql)


def load_dataframe(
    dataframe: pd.DataFrame,
    table_name: str,
    connection: sqlite3.Connection,
) -> None:
    """Append a DataFrame to a pre-created SQLite table."""

    print(
        f"Loading {len(dataframe):,} rows "
        f"into {table_name}..."
    )

    dataframe.to_sql(
        table_name,
        connection,
        if_exists="append",
        index=False,
        chunksize=20_000,
    )


def insert_metadata(
    connection: sqlite3.Connection,
    policy_rows: int,
    claim_rows: int,
    orphan_rows: int,
    issue_rows: int,
) -> None:
    """Record database build information."""

    metadata = [
        (
            "database_build_timestamp_utc",
            datetime.now(
                timezone.utc
            ).isoformat(),
        ),
        (
            "source_dataset",
            "freMTPL2",
        ),
        (
            "policy_rows",
            str(policy_rows),
        ),
        (
            "matched_claim_rows",
            str(claim_rows),
        ),
        (
            "orphan_claim_rows",
            str(orphan_rows),
        ),
        (
            "claim_count_issue_rows",
            str(issue_rows),
        ),
    ]

    connection.executemany(
        """
        INSERT INTO database_metadata (
            MetadataKey,
            MetadataValue
        )
        VALUES (?, ?)
        """,
        metadata,
    )


def validate_database(
    connection: sqlite3.Connection,
    expected_policy_rows: int,
    expected_claim_rows: int,
    expected_orphan_rows: int,
    expected_issue_rows: int,
) -> None:
    """Validate table counts and foreign keys."""

    expected_counts = {
        "policies": expected_policy_rows,
        "claims": expected_claim_rows,
        "orphan_claims": expected_orphan_rows,
        "claim_count_issues": expected_issue_rows,
    }

    for table_name, expected_count in expected_counts.items():
        actual_count = connection.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]

        if actual_count != expected_count:
            raise ValueError(
                f"{table_name} contains "
                f"{actual_count:,} rows; "
                f"expected {expected_count:,}."
            )

    foreign_key_issues = connection.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    if foreign_key_issues:
        raise ValueError(
            "Foreign-key validation failed: "
            f"{foreign_key_issues[:10]}"
        )

    duplicate_policies = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT IDpol
            FROM policies
            GROUP BY IDpol
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_policies > 0:
        raise ValueError(
            "Duplicate policy IDs were found "
            "in the database."
        )


def main() -> None:
    """Build and validate the insurance database."""

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading prepared source data...")

    policy, severity, audit = load_source_data()

    print("Preparing database tables...")

    policy_table = prepare_policy_table(
        policy
    )

    policy_ids = set(
        policy_table["IDpol"].tolist()
    )

    matched_claims, orphan_claims = (
        prepare_claim_tables(
            severity,
            policy_ids,
        )
    )

    issue_table = prepare_issue_table(
        audit
    )

    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    print("Creating SQLite database...")

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        initialize_database(connection)

        load_dataframe(
            policy_table,
            "policies",
            connection,
        )

        load_dataframe(
            matched_claims,
            "claims",
            connection,
        )

        load_dataframe(
            orphan_claims,
            "orphan_claims",
            connection,
        )

        load_dataframe(
            issue_table,
            "claim_count_issues",
            connection,
        )

        insert_metadata(
            connection=connection,
            policy_rows=len(policy_table),
            claim_rows=len(matched_claims),
            orphan_rows=len(orphan_claims),
            issue_rows=len(issue_table),
        )

        validate_database(
            connection=connection,
            expected_policy_rows=len(policy_table),
            expected_claim_rows=len(matched_claims),
            expected_orphan_rows=len(orphan_claims),
            expected_issue_rows=len(issue_table),
        )

        connection.execute("ANALYZE")

    database_size_mb = (
        DATABASE_PATH.stat().st_size
        / 1024
        / 1024
    )

    print("\nDatabase built successfully.")
    print(f"Policies: {len(policy_table):,}")
    print(
        "Matched claim rows:",
        f"{len(matched_claims):,}",
    )
    print(
        "Orphan claim rows:",
        f"{len(orphan_claims):,}",
    )
    print(
        "Unique orphan policy IDs:",
        f"{orphan_claims['IDpol'].nunique():,}",
    )
    print(
        "Claim-count issues:",
        f"{len(issue_table):,}",
    )
    print(
        "Database size:",
        f"{database_size_mb:,.2f} MB",
    )
    print(f"Database path: {DATABASE_PATH}")


if __name__ == "__main__":
    main()