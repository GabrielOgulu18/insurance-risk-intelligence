"""Export selected SQLite insurance analyses to CSV reports."""

from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "insurance_analytics.sqlite"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "sql"
)


REPORT_QUERIES = {
    "portfolio_kpis": """
        SELECT *
        FROM v_portfolio_kpis
    """,

    "area_metrics": """
        SELECT *
        FROM v_area_metrics
        ORDER BY Area
    """,

    "model_eligibility": """
        SELECT *
        FROM v_model_eligibility
    """,

    "claim_completeness": """
        SELECT *
        FROM v_claim_completeness
        ORDER BY policy_count DESC
    """,

    "claim_count_issues": """
        SELECT
            AuditStatus,
            COUNT(*) AS policy_count,
            SUM(ClaimCountDifference)
                AS total_count_difference
        FROM claim_count_issues
        GROUP BY AuditStatus
        ORDER BY policy_count DESC
    """,

    "top_regions_by_pure_premium": """
        SELECT
            Region,

            COUNT(*) AS policies,

            SUM(
                CASE
                    WHEN PurePremiumModelEligible = 1
                    THEN Exposure
                    ELSE 0
                END
            ) AS complete_loss_exposure,

            SUM(
                CASE
                    WHEN PurePremiumModelEligible = 1
                    THEN CompleteTotalClaimAmount
                    ELSE 0
                END
            )
            /
            NULLIF(
                SUM(
                    CASE
                        WHEN PurePremiumModelEligible = 1
                        THEN Exposure
                        ELSE 0
                    END
                ),
                0
            ) AS pure_premium

        FROM policies
        GROUP BY Region

        HAVING
            SUM(
                CASE
                    WHEN PurePremiumModelEligible = 1
                    THEN Exposure
                    ELSE 0
                END
            ) >= 1000

        ORDER BY pure_premium DESC
        LIMIT 10
    """,
}


def main() -> None:
    """Run SQL reports and export the results."""

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "Database not found. Run "
            "`python src/build_database.py` first."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:
        for report_name, query in REPORT_QUERIES.items():
            dataframe = pd.read_sql_query(
                query,
                connection,
            )

            output_path = (
                OUTPUT_DIR
                / f"{report_name}.csv"
            )

            dataframe.to_csv(
                output_path,
                index=False,
            )

            print(
                f"Saved {report_name}: "
                f"{len(dataframe):,} rows"
            )

    print(
        "\nSQL report export completed successfully."
    )
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()