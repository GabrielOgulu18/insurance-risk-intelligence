PRAGMA foreign_keys = ON;

DROP VIEW IF EXISTS v_portfolio_kpis;
DROP VIEW IF EXISTS v_area_metrics;
DROP VIEW IF EXISTS v_model_eligibility;
DROP VIEW IF EXISTS v_claim_completeness;

DROP TABLE IF EXISTS claims;
DROP TABLE IF EXISTS orphan_claims;
DROP TABLE IF EXISTS claim_count_issues;
DROP TABLE IF EXISTS policies;
DROP TABLE IF EXISTS database_metadata;

CREATE TABLE policies (
    IDpol INTEGER PRIMARY KEY,
    ClaimNb INTEGER NOT NULL CHECK (ClaimNb >= 0),
    Exposure REAL NOT NULL CHECK (Exposure > 0),
    Area TEXT,
    VehPower REAL,
    VehAge REAL,
    DrivAge REAL,
    BonusMalus REAL,
    VehBrand TEXT,
    VehGas TEXT,
    Density REAL,
    Region TEXT,

    HasClaim INTEGER NOT NULL CHECK (HasClaim IN (0, 1)),
    HasSeverityRecord INTEGER NOT NULL
        CHECK (HasSeverityRecord IN (0, 1)),

    SeverityRowCount INTEGER NOT NULL
        CHECK (SeverityRowCount >= 0),

    SeverityTotalClaimAmount REAL NOT NULL
        DEFAULT 0
        CHECK (SeverityTotalClaimAmount >= 0),

    SeverityAverageClaimAmount REAL,
    SeverityMinimumClaimAmount REAL,
    SeverityMaximumClaimAmount REAL,

    ClaimCountDifference INTEGER NOT NULL,

    ClaimCountMatch INTEGER NOT NULL
        CHECK (ClaimCountMatch IN (0, 1)),

    SeverityCompletenessStatus TEXT NOT NULL,

    AnnualizedClaimFrequency REAL NOT NULL
        CHECK (AnnualizedClaimFrequency >= 0),

    ObservedAverageClaimAmount REAL,
    CompleteTotalClaimAmount REAL,
    CompleteAverageClaimAmount REAL,
    CompletePurePremium REAL,

    FrequencyModelEligible INTEGER NOT NULL
        CHECK (FrequencyModelEligible IN (0, 1)),

    SeverityModelEligible INTEGER NOT NULL
        CHECK (SeverityModelEligible IN (0, 1)),

    PurePremiumModelEligible INTEGER NOT NULL
        CHECK (PurePremiumModelEligible IN (0, 1)),

    AnyPredictorValidityIssue INTEGER NOT NULL
        CHECK (AnyPredictorValidityIssue IN (0, 1))
);

CREATE TABLE claims (
    ClaimID INTEGER PRIMARY KEY,
    IDpol INTEGER NOT NULL,
    ClaimAmount REAL NOT NULL CHECK (ClaimAmount > 0),

    FOREIGN KEY (IDpol)
        REFERENCES policies(IDpol)
);

CREATE TABLE orphan_claims (
    OrphanClaimID INTEGER PRIMARY KEY,
    IDpol INTEGER NOT NULL,
    ClaimAmount REAL NOT NULL CHECK (ClaimAmount > 0)
);

CREATE TABLE claim_count_issues (
    IDpol INTEGER PRIMARY KEY,
    ClaimNb INTEGER NOT NULL CHECK (ClaimNb >= 0),
    SeverityRowCount INTEGER NOT NULL
        CHECK (SeverityRowCount >= 0),

    ClaimCountDifference INTEGER NOT NULL,

    ClaimCountMatch INTEGER NOT NULL
        CHECK (ClaimCountMatch IN (0, 1)),

    AuditStatus TEXT NOT NULL,
    SourceStatus TEXT NOT NULL
);

CREATE TABLE database_metadata (
    MetadataKey TEXT PRIMARY KEY,
    MetadataValue TEXT NOT NULL
);

CREATE INDEX idx_policies_area
    ON policies(Area);

CREATE INDEX idx_policies_region
    ON policies(Region);

CREATE INDEX idx_policies_driver_age
    ON policies(DrivAge);

CREATE INDEX idx_policies_bonus_malus
    ON policies(BonusMalus);

CREATE INDEX idx_policies_has_claim
    ON policies(HasClaim);

CREATE INDEX idx_policies_completeness
    ON policies(SeverityCompletenessStatus);

CREATE INDEX idx_policies_severity_eligible
    ON policies(SeverityModelEligible);

CREATE INDEX idx_policies_pure_premium_eligible
    ON policies(PurePremiumModelEligible);

CREATE INDEX idx_claims_policy
    ON claims(IDpol);

CREATE INDEX idx_claim_count_issues_status
    ON claim_count_issues(AuditStatus);

CREATE VIEW v_portfolio_kpis AS
SELECT
    COUNT(*) AS total_policies,

    SUM(Exposure) AS total_exposure_years,

    SUM(ClaimNb) AS total_reported_claims,

    SUM(HasClaim) AS policies_with_claims,

    1.0 * SUM(HasClaim)
        / NULLIF(COUNT(*), 0)
        AS claim_occurrence_rate,

    1.0 * SUM(ClaimNb)
        / NULLIF(SUM(Exposure), 0)
        AS claim_frequency,

    100.0 * SUM(ClaimNb)
        / NULLIF(SUM(Exposure), 0)
        AS claims_per_100_exposure_years,

    SUM(
        CASE
            WHEN PurePremiumModelEligible = 1
            THEN 1
            ELSE 0
        END
    ) AS complete_loss_policies,

    SUM(
        CASE
            WHEN PurePremiumModelEligible = 1
            THEN CompleteTotalClaimAmount
            ELSE 0
        END
    ) AS complete_claim_cost,

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

FROM policies;

CREATE VIEW v_area_metrics AS
SELECT
    Area,

    COUNT(*) AS policies,

    SUM(Exposure) AS exposure,

    SUM(ClaimNb) AS claims,

    SUM(HasClaim) AS policies_with_claim,

    1.0 * SUM(HasClaim)
        / NULLIF(COUNT(*), 0)
        AS claim_occurrence_rate,

    1.0 * SUM(ClaimNb)
        / NULLIF(SUM(Exposure), 0)
        AS claim_frequency,

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
    ) AS pure_premium,

    100.0 * SUM(
        CASE
            WHEN PurePremiumModelEligible = 1
            THEN 1
            ELSE 0
        END
    )
    / NULLIF(COUNT(*), 0)
        AS complete_loss_percentage

FROM policies
GROUP BY Area;

CREATE VIEW v_model_eligibility AS
SELECT
    'Frequency model' AS model_sample,
    SUM(FrequencyModelEligible) AS eligible_policies,
    100.0 * SUM(FrequencyModelEligible)
        / COUNT(*) AS eligible_percentage
FROM policies

UNION ALL

SELECT
    'Severity model',
    SUM(SeverityModelEligible),
    100.0 * SUM(SeverityModelEligible)
        / COUNT(*)
FROM policies

UNION ALL

SELECT
    'Pure premium model',
    SUM(PurePremiumModelEligible),
    100.0 * SUM(PurePremiumModelEligible)
        / COUNT(*)
FROM policies;

CREATE VIEW v_claim_completeness AS
SELECT
    SeverityCompletenessStatus AS completeness_status,
    COUNT(*) AS policy_count,
    100.0 * COUNT(*)
        / (
            SELECT COUNT(*)
            FROM policies
        ) AS percentage
FROM policies
GROUP BY SeverityCompletenessStatus;