-- =========================================================
-- 1. Overall portfolio KPIs
-- =========================================================

SELECT *
FROM v_portfolio_kpis;


-- =========================================================
-- 2. Table row counts
-- =========================================================

SELECT
    'policies' AS table_name,
    COUNT(*) AS row_count
FROM policies

UNION ALL

SELECT
    'claims',
    COUNT(*)
FROM claims

UNION ALL

SELECT
    'orphan_claims',
    COUNT(*)
FROM orphan_claims

UNION ALL

SELECT
    'claim_count_issues',
    COUNT(*)
FROM claim_count_issues;


-- =========================================================
-- 3. Claim-count distribution
-- =========================================================

SELECT
    CASE
        WHEN ClaimNb >= 4 THEN '4+'
        ELSE CAST(ClaimNb AS TEXT)
    END AS claim_count_group,

    COUNT(*) AS policy_count,

    100.0 * COUNT(*)
        / (
            SELECT COUNT(*)
            FROM policies
        ) AS percentage

FROM policies
GROUP BY
    CASE
        WHEN ClaimNb >= 4 THEN '4+'
        ELSE CAST(ClaimNb AS TEXT)
    END
ORDER BY
    CASE claim_count_group
        WHEN '0' THEN 0
        WHEN '1' THEN 1
        WHEN '2' THEN 2
        WHEN '3' THEN 3
        ELSE 4
    END;


-- =========================================================
-- 4. Area-level insurance metrics
-- =========================================================

SELECT *
FROM v_area_metrics
ORDER BY Area;


-- =========================================================
-- 5. Highest and lowest area claim frequency
-- =========================================================

SELECT
    Area,
    policies,
    exposure,
    claims,
    claims_per_100_exposure_years
FROM v_area_metrics
ORDER BY claims_per_100_exposure_years DESC;


-- =========================================================
-- 6. Model eligibility
-- =========================================================

SELECT *
FROM v_model_eligibility;


-- =========================================================
-- 7. Severity completeness
-- =========================================================

SELECT *
FROM v_claim_completeness
ORDER BY policy_count DESC;


-- =========================================================
-- 8. Claim-count issue types
-- =========================================================

SELECT
    AuditStatus,
    COUNT(*) AS policy_count,
    SUM(ClaimCountDifference) AS total_count_difference
FROM claim_count_issues
GROUP BY AuditStatus
ORDER BY policy_count DESC;


-- =========================================================
-- 9. Orphan claims
-- =========================================================

SELECT
    IDpol,
    COUNT(*) AS claim_rows,
    SUM(ClaimAmount) AS total_claim_amount
FROM orphan_claims
GROUP BY IDpol
ORDER BY total_claim_amount DESC;


-- =========================================================
-- 10. Driver-age band metrics
-- =========================================================

WITH driver_age_groups AS (
    SELECT
        CASE
            WHEN DrivAge < 21 THEN 'Under 21'
            WHEN DrivAge < 25 THEN '21-24'
            WHEN DrivAge < 30 THEN '25-29'
            WHEN DrivAge < 40 THEN '30-39'
            WHEN DrivAge < 50 THEN '40-49'
            WHEN DrivAge < 60 THEN '50-59'
            WHEN DrivAge < 70 THEN '60-69'
            WHEN DrivAge < 80 THEN '70-79'
            ELSE '80+'
        END AS driver_age_band,

        CASE
            WHEN DrivAge < 21 THEN 1
            WHEN DrivAge < 25 THEN 2
            WHEN DrivAge < 30 THEN 3
            WHEN DrivAge < 40 THEN 4
            WHEN DrivAge < 50 THEN 5
            WHEN DrivAge < 60 THEN 6
            WHEN DrivAge < 70 THEN 7
            WHEN DrivAge < 80 THEN 8
            ELSE 9
        END AS sort_order,

        *

    FROM policies
)

SELECT
    driver_age_band,
    COUNT(*) AS policies,
    SUM(Exposure) AS exposure,
    SUM(ClaimNb) AS claims,

    100.0 * SUM(ClaimNb)
        / NULLIF(SUM(Exposure), 0)
        AS claims_per_100_exposure_years,

    SUM(
        CASE
            WHEN SeverityModelEligible = 1
            THEN CompleteTotalClaimAmount
            ELSE 0
        END
    )
    /
    NULLIF(
        SUM(
            CASE
                WHEN SeverityModelEligible = 1
                THEN ClaimNb
                ELSE 0
            END
        ),
        0
    ) AS average_claim_severity,

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

FROM driver_age_groups
GROUP BY
    driver_age_band,
    sort_order
ORDER BY sort_order;


-- =========================================================
-- 11. Bonus-malus band metrics
-- =========================================================

WITH bonus_malus_groups AS (
    SELECT
        CASE
            WHEN BonusMalus < 60 THEN 'Below 60'
            WHEN BonusMalus < 70 THEN '60-69'
            WHEN BonusMalus < 80 THEN '70-79'
            WHEN BonusMalus < 90 THEN '80-89'
            WHEN BonusMalus < 100 THEN '90-99'
            WHEN BonusMalus < 110 THEN '100-109'
            WHEN BonusMalus < 130 THEN '110-129'
            WHEN BonusMalus < 150 THEN '130-149'
            ELSE '150+'
        END AS bonus_malus_band,

        CASE
            WHEN BonusMalus < 60 THEN 1
            WHEN BonusMalus < 70 THEN 2
            WHEN BonusMalus < 80 THEN 3
            WHEN BonusMalus < 90 THEN 4
            WHEN BonusMalus < 100 THEN 5
            WHEN BonusMalus < 110 THEN 6
            WHEN BonusMalus < 130 THEN 7
            WHEN BonusMalus < 150 THEN 8
            ELSE 9
        END AS sort_order,

        *

    FROM policies
)

SELECT
    bonus_malus_band,
    COUNT(*) AS policies,
    SUM(Exposure) AS exposure,
    SUM(ClaimNb) AS claims,

    100.0 * SUM(ClaimNb)
        / NULLIF(SUM(Exposure), 0)
        AS claims_per_100_exposure_years,

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

FROM bonus_malus_groups
GROUP BY
    bonus_malus_band,
    sort_order
ORDER BY sort_order;


-- =========================================================
-- 12. Regions with highest pure premium
--     Minimum 1,000 complete exposure-years
-- =========================================================

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
LIMIT 10;


-- =========================================================
-- 13. Policies with the largest complete losses
-- =========================================================

SELECT
    IDpol,
    ClaimNb,
    Exposure,
    Area,
    VehPower,
    VehAge,
    DrivAge,
    BonusMalus,
    VehBrand,
    VehGas,
    Region,
    CompleteTotalClaimAmount,
    CompletePurePremium

FROM policies

WHERE
    PurePremiumModelEligible = 1
    AND CompleteTotalClaimAmount > 0

ORDER BY CompleteTotalClaimAmount DESC
LIMIT 20;


-- =========================================================
-- 14. Database metadata
-- =========================================================

SELECT *
FROM database_metadata
ORDER BY MetadataKey;