"""Train and evaluate conditional claim-severity models.

The modelling sample contains policies with complete positive claim
information.

The target is average claim amount per claim:

    CompleteTotalClaimAmount / ClaimNb

ClaimNb is used as the sample weight because each row represents a
policy-level average over one or more claims.

The workflow compares:

1. A constant weighted-mean baseline.
2. Regularized Gamma regression.
3. Histogram gradient boosting with Gamma loss.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import (
    d2_tweedie_score,
    mean_gamma_deviance,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "policy_analytics.csv"
)

GAMMA_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "claim_severity_gamma.joblib"
)

BOOSTING_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "claim_severity_hist_gradient_boosting.joblib"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "claim_severity_test_predictions.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "claim_severity"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "modeling"
    / "claim_severity"
)

DOCUMENTATION_PATH = (
    PROJECT_ROOT
    / "docs"
    / "claim_severity_model.md"
)

RANDOM_STATE = 42
EPSILON = 1e-12

TARGET = "AverageClaimSeverity"

NUMERIC_FEATURES = [
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "LogDensity",
]

CATEGORICAL_FEATURES = [
    "Area",
    "VehBrand",
    "VehGas",
    "Region",
]

MODEL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


def convert_to_boolean(
    series: pd.Series,
) -> pd.Series:
    """Convert common Boolean representations."""

    if pd.api.types.is_bool_dtype(series):
        return series

    converted = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if converted.isna().any():
        bad_values = sorted(
            series.loc[converted.isna()]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Could not convert Boolean values: "
            f"{bad_values}"
        )

    return converted.astype(bool)


def create_directories() -> None:
    """Create output directories."""

    for directory in [
        GAMMA_MODEL_PATH.parent,
        REPORT_DIR,
        FIGURES_DIR,
        DOCUMENTATION_PATH.parent,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def load_modelling_data() -> pd.DataFrame:
    """Load the complete positive-claim severity sample."""

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            "Policy analytics data was not found. "
            "Run src/prepare_data.py first."
        )

    policy = pd.read_csv(
        POLICY_PATH,
        low_memory=False,
    )

    policy["SeverityModelEligible"] = (
        convert_to_boolean(
            policy["SeverityModelEligible"]
        )
    )

    model_data = policy.loc[
        policy["SeverityModelEligible"],
        [
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
            "CompleteTotalClaimAmount",
            "CompleteAverageClaimAmount",
        ],
    ].copy()

    numeric_columns = [
        "ClaimNb",
        "Exposure",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "Density",
        "CompleteTotalClaimAmount",
        "CompleteAverageClaimAmount",
    ]

    for column in numeric_columns:
        model_data[column] = pd.to_numeric(
            model_data[column],
            errors="coerce",
        )

    required_positive_columns = [
        "ClaimNb",
        "CompleteTotalClaimAmount",
        "CompleteAverageClaimAmount",
    ]

    for column in required_positive_columns:
        if model_data[column].isna().any():
            raise ValueError(
                f"{column} contains missing values."
            )

        if model_data[column].le(0).any():
            raise ValueError(
                f"{column} contains non-positive values."
            )

    if model_data["Density"].lt(0).any():
        raise ValueError(
            "Density contains negative values."
        )

    model_data["ClaimNb"] = (
        model_data["ClaimNb"]
        .astype("int64")
    )

    recomputed_average = (
        model_data["CompleteTotalClaimAmount"]
        / model_data["ClaimNb"]
    )

    averages_match = np.allclose(
        recomputed_average,
        model_data[
            "CompleteAverageClaimAmount"
        ],
        rtol=1e-8,
        atol=1e-6,
    )

    if not averages_match:
        raise ValueError(
            "CompleteAverageClaimAmount does not "
            "reconcile with total claim amount "
            "divided by ClaimNb."
        )

    model_data[TARGET] = recomputed_average

    model_data["LogDensity"] = np.log1p(
        model_data["Density"]
    )

    severity_rank = (
        model_data[TARGET]
        .rank(method="first")
    )

    model_data["SeverityStratum"] = pd.qcut(
        severity_rank,
        q=10,
        labels=False,
    )

    for column in CATEGORICAL_FEATURES:
        model_data[column] = (
            model_data[column]
            .astype("string")
            .fillna("Missing")
            .astype("category")
        )

    return model_data


def split_modelling_data(
    model_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create stratified 60/20/20 severity samples."""

    development, test = train_test_split(
        model_data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=model_data[
            "SeverityStratum"
        ],
    )

    train, validation = train_test_split(
        development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=development[
            "SeverityStratum"
        ],
    )

    return train, validation, test


def create_split_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize sample sizes and severity outcomes."""

    rows = []

    for sample_name, dataframe in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        total_claims = dataframe[
            "ClaimNb"
        ].sum()

        total_claim_amount = dataframe[
            "CompleteTotalClaimAmount"
        ].sum()

        rows.append(
            {
                "Sample": sample_name,
                "Policies": len(dataframe),
                "Claims": total_claims,
                "TotalClaimAmount": (
                    total_claim_amount
                ),
                "WeightedMeanSeverity": (
                    total_claim_amount
                    / total_claims
                ),
                "MedianPolicyAverageSeverity": (
                    dataframe[TARGET].median()
                ),
                "MaximumPolicyAverageSeverity": (
                    dataframe[TARGET].max()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_gamma_pipeline() -> Pipeline:
    """Create the regularized Gamma-regression pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    regressor = GammaRegressor(
        alpha=1e-4,
        solver="lbfgs",
        max_iter=2_000,
        tol=1e-8,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regressor",
                regressor,
            ),
        ]
    )


def build_boosting_model(
) -> HistGradientBoostingRegressor:
    """Create the nonlinear Gamma boosting model."""

    return HistGradientBoostingRegressor(
        loss="gamma",
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=5.0,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        tol=1e-7,
        random_state=RANDOM_STATE,
    )


def positive_predictions(
    predictions: np.ndarray,
) -> np.ndarray:
    """Ensure predictions are strictly positive."""

    return np.clip(
        np.asarray(predictions),
        EPSILON,
        None,
    )


def calculate_severity_metrics(
    model_name: str,
    sample_name: str,
    dataframe: pd.DataFrame,
    predicted_severity: np.ndarray,
) -> dict:
    """Calculate severity and financial metrics."""

    observed_severity = (
        dataframe[TARGET]
        .to_numpy()
    )

    claim_weights = (
        dataframe["ClaimNb"]
        .to_numpy()
    )

    predicted_severity = positive_predictions(
        predicted_severity
    )

    observed_total_amount = float(
        dataframe[
            "CompleteTotalClaimAmount"
        ].sum()
    )

    predicted_total_amount = float(
        np.sum(
            predicted_severity
            * claim_weights
        )
    )

    observed_mean_severity = float(
        np.average(
            observed_severity,
            weights=claim_weights,
        )
    )

    predicted_mean_severity = float(
        np.average(
            predicted_severity,
            weights=claim_weights,
        )
    )

    weighted_gamma_deviance = (
        mean_gamma_deviance(
            observed_severity,
            predicted_severity,
            sample_weight=claim_weights,
        )
    )

    d2_gamma = d2_tweedie_score(
        observed_severity,
        predicted_severity,
        sample_weight=claim_weights,
        power=2,
    )

    weighted_mae = float(
        np.average(
            np.abs(
                observed_severity
                - predicted_severity
            ),
            weights=claim_weights,
        )
    )

    weighted_rmse = float(
        np.sqrt(
            np.average(
                (
                    observed_severity
                    - predicted_severity
                ) ** 2,
                weights=claim_weights,
            )
        )
    )

    weighted_bias = float(
        np.average(
            predicted_severity
            - observed_severity,
            weights=claim_weights,
        )
    )

    return {
        "Model": model_name,
        "Sample": sample_name,
        "WeightedGammaDeviance": (
            weighted_gamma_deviance
        ),
        "D2Gamma": d2_gamma,
        "WeightedMAE": weighted_mae,
        "WeightedRMSE": weighted_rmse,
        "WeightedBias": weighted_bias,
        "Policies": len(dataframe),
        "Claims": int(
            claim_weights.sum()
        ),
        "ObservedClaimAmount": (
            observed_total_amount
        ),
        "PredictedClaimAmount": (
            predicted_total_amount
        ),
        "PredictedToObservedAmountRatio": (
            predicted_total_amount
            / observed_total_amount
        ),
        "ObservedMeanSeverity": (
            observed_mean_severity
        ),
        "PredictedMeanSeverity": (
            predicted_mean_severity
        ),
        "MeanSeverityDifference": (
            predicted_mean_severity
            - observed_mean_severity
        ),
    }


def create_calibration_deciles(
    dataframe: pd.DataFrame,
    predicted_severity: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Create observed-versus-predicted severity deciles."""

    calibration_data = dataframe[
        [
            "IDpol",
            "ClaimNb",
            "CompleteTotalClaimAmount",
            TARGET,
        ]
    ].copy()

    calibration_data[
        "PredictedSeverity"
    ] = positive_predictions(
        predicted_severity
    )

    calibration_data[
        "PredictedClaimAmount"
    ] = (
        calibration_data[
            "PredictedSeverity"
        ]
        * calibration_data["ClaimNb"]
    )

    predicted_rank = (
        calibration_data[
            "PredictedSeverity"
        ]
        .rank(method="first")
    )

    calibration_data["RiskDecile"] = pd.qcut(
        predicted_rank,
        q=10,
        labels=range(1, 11),
    )

    summary = (
        calibration_data.groupby(
            "RiskDecile",
            observed=True,
        )
        .agg(
            Policies=("IDpol", "size"),
            Claims=("ClaimNb", "sum"),
            ObservedClaimAmount=(
                "CompleteTotalClaimAmount",
                "sum",
            ),
            PredictedClaimAmount=(
                "PredictedClaimAmount",
                "sum",
            ),
            MinimumPredictedSeverity=(
                "PredictedSeverity",
                "min",
            ),
            MaximumPredictedSeverity=(
                "PredictedSeverity",
                "max",
            ),
        )
        .reset_index()
    )

    summary["ObservedSeverity"] = (
        summary["ObservedClaimAmount"]
        / summary["Claims"]
    )

    summary["PredictedSeverity"] = (
        summary["PredictedClaimAmount"]
        / summary["Claims"]
    )

    summary["ObservedToExpectedRatio"] = (
        summary["ObservedClaimAmount"]
        / summary[
            "PredictedClaimAmount"
        ].replace(0, np.nan)
    )

    summary.insert(
        0,
        "Model",
        model_name,
    )

    return summary


def add_severity_bands(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create readable segment bands."""

    output = dataframe.copy()

    output["DrivAgeBand"] = pd.cut(
        output["DrivAge"],
        bins=[
            0,
            21,
            25,
            30,
            40,
            50,
            60,
            70,
            80,
            np.inf,
        ],
        labels=[
            "Under 21",
            "21-24",
            "25-29",
            "30-39",
            "40-49",
            "50-59",
            "60-69",
            "70-79",
            "80+",
        ],
        right=False,
    )

    output["VehAgeBand"] = pd.cut(
        output["VehAge"],
        bins=[
            0,
            1,
            3,
            6,
            11,
            16,
            21,
            np.inf,
        ],
        labels=[
            "0",
            "1-2",
            "3-5",
            "6-10",
            "11-15",
            "16-20",
            "21+",
        ],
        right=False,
    )

    output["BonusMalusBand"] = pd.cut(
        output["BonusMalus"],
        bins=[
            0,
            60,
            70,
            80,
            90,
            100,
            110,
            130,
            150,
            np.inf,
        ],
        labels=[
            "Below 60",
            "60-69",
            "70-79",
            "80-89",
            "90-99",
            "100-109",
            "110-129",
            "130-149",
            "150+",
        ],
        right=False,
    )

    return output


def create_segment_calibration(
    dataframe: pd.DataFrame,
    predicted_severity: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Summarize severity calibration by portfolio segment."""

    analysis_data = add_severity_bands(
        dataframe
    )

    analysis_data[
        "PredictedSeverity"
    ] = positive_predictions(
        predicted_severity
    )

    analysis_data[
        "PredictedClaimAmount"
    ] = (
        analysis_data[
            "PredictedSeverity"
        ]
        * analysis_data["ClaimNb"]
    )

    group_columns = [
        "Area",
        "VehGas",
        "VehBrand",
        "DrivAgeBand",
        "VehAgeBand",
        "BonusMalusBand",
    ]

    outputs = []

    for group_column in group_columns:
        summary = (
            analysis_data.groupby(
                group_column,
                observed=True,
                dropna=False,
            )
            .agg(
                Policies=("IDpol", "size"),
                Claims=("ClaimNb", "sum"),
                ObservedClaimAmount=(
                    "CompleteTotalClaimAmount",
                    "sum",
                ),
                PredictedClaimAmount=(
                    "PredictedClaimAmount",
                    "sum",
                ),
            )
            .reset_index()
            .rename(
                columns={
                    group_column: "Segment"
                }
            )
        )

        summary.insert(
            0,
            "SegmentType",
            group_column,
        )

        summary.insert(
            0,
            "Model",
            model_name,
        )

        summary["Segment"] = (
            summary["Segment"]
            .astype(str)
        )

        summary["ObservedSeverity"] = (
            summary["ObservedClaimAmount"]
            / summary["Claims"]
        )

        summary["PredictedSeverity"] = (
            summary["PredictedClaimAmount"]
            / summary["Claims"]
        )

        summary[
            "ObservedToExpectedRatio"
        ] = (
            summary["ObservedClaimAmount"]
            / summary[
                "PredictedClaimAmount"
            ].replace(0, np.nan)
        )

        summary[
            "EligibleForComparison"
        ] = (
            summary["Policies"].ge(100)
            & summary["Claims"].ge(100)
        )

        outputs.append(summary)

    return pd.concat(
        outputs,
        ignore_index=True,
    )


def extract_gamma_coefficients(
    model: Pipeline,
) -> pd.DataFrame:
    """Extract Gamma coefficients and severity ratios."""

    feature_names = (
        model.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )

    coefficients = (
        model.named_steps[
            "regressor"
        ].coef_
    )

    output = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
        }
    )

    output["SeverityRatio"] = np.exp(
        np.clip(
            output["Coefficient"],
            -20,
            20,
        )
    )

    output["AbsoluteCoefficient"] = (
        output["Coefficient"].abs()
    )

    return output.sort_values(
        "AbsoluteCoefficient",
        ascending=False,
    )


def create_tail_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the severity distribution and loss concentration."""

    quantiles = [
        0.00,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        0.995,
        0.999,
        1.00,
    ]

    quantile_names = [
        "Minimum",
        "25th percentile",
        "Median",
        "75th percentile",
        "90th percentile",
        "95th percentile",
        "99th percentile",
        "99.5th percentile",
        "99.9th percentile",
        "Maximum",
    ]

    severity_quantiles = (
        dataframe[TARGET]
        .quantile(quantiles)
        .to_numpy()
    )

    rows = []

    for name, value in zip(
        quantile_names,
        severity_quantiles,
    ):
        rows.append(
            {
                "Metric": (
                    f"Policy average severity: {name}"
                ),
                "Value": value,
            }
        )

    ordered_losses = dataframe.sort_values(
        "CompleteTotalClaimAmount",
        ascending=False,
    )

    total_amount = ordered_losses[
        "CompleteTotalClaimAmount"
    ].sum()

    for percentage in [
        0.01,
        0.05,
        0.10,
    ]:
        number_of_policies = max(
            1,
            int(
                np.ceil(
                    len(ordered_losses)
                    * percentage
                )
            ),
        )

        selected_amount = (
            ordered_losses.head(
                number_of_policies
            )[
                "CompleteTotalClaimAmount"
            ].sum()
        )

        rows.append(
            {
                "Metric": (
                    f"Top {percentage:.0%} "
                    "policy share of claim amount"
                ),
                "Value": (
                    selected_amount
                    / total_amount
                ),
            }
        )

    return pd.DataFrame(rows)


def create_largest_error_report(
    dataframe: pd.DataFrame,
    predicted_severity: np.ndarray,
) -> pd.DataFrame:
    """Identify policies contributing the largest financial errors."""

    output = dataframe[
        [
            "IDpol",
            "ClaimNb",
            "Area",
            "VehPower",
            "VehAge",
            "DrivAge",
            "BonusMalus",
            "VehBrand",
            "VehGas",
            "Region",
            "CompleteTotalClaimAmount",
            TARGET,
        ]
    ].copy()

    output["PredictedSeverity"] = (
        positive_predictions(
            predicted_severity
        )
    )

    output["PredictedClaimAmount"] = (
        output["PredictedSeverity"]
        * output["ClaimNb"]
    )

    output["SeverityError"] = (
        output["PredictedSeverity"]
        - output[TARGET]
    )

    output["AbsoluteSeverityError"] = (
        output["SeverityError"].abs()
    )

    output["FinancialError"] = (
        output["PredictedClaimAmount"]
        - output[
            "CompleteTotalClaimAmount"
        ]
    )

    output["AbsoluteFinancialError"] = (
        output["FinancialError"].abs()
    )

    return output.nlargest(
        25,
        "AbsoluteFinancialError",
    )


def save_deviance_comparison(
    test_metrics: pd.DataFrame,
) -> None:
    """Save the test Gamma-deviance comparison."""

    plot_data = test_metrics.sort_values(
        "WeightedGammaDeviance"
    )

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        plot_data["Model"],
        plot_data[
            "WeightedGammaDeviance"
        ],
    )

    plt.title(
        "Claim-Weighted Test Gamma Deviance"
    )

    plt.xlabel("Model")
    plt.ylabel("Mean Gamma deviance")
    plt.xticks(rotation=20, ha="right")

    for bar, value in zip(
        bars,
        plot_data[
            "WeightedGammaDeviance"
        ],
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.5f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "test_gamma_deviance_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_calibration_scatter(
    calibration: pd.DataFrame,
) -> None:
    """Save predicted-versus-observed decile calibration."""

    maximum_value = max(
        calibration[
            "ObservedSeverity"
        ].max(),
        calibration[
            "PredictedSeverity"
        ].max(),
    )

    plt.figure(figsize=(7, 6))

    plt.scatter(
        calibration[
            "PredictedSeverity"
        ],
        calibration[
            "ObservedSeverity"
        ],
    )

    for row in calibration.itertuples(
        index=False
    ):
        plt.annotate(
            str(row.RiskDecile),
            (
                row.PredictedSeverity,
                row.ObservedSeverity,
            ),
        )

    plt.plot(
        [0, maximum_value],
        [0, maximum_value],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.xlabel("Predicted average severity")
    plt.ylabel("Observed average severity")

    plt.title(
        "Claim-Severity Calibration by Risk Decile"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "severity_calibration_deciles.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_decile_severity_figure(
    calibration: pd.DataFrame,
) -> None:
    """Save observed and predicted severity by decile."""

    ordered = calibration.sort_values(
        "RiskDecile"
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        ordered["RiskDecile"].astype(int),
        ordered["ObservedSeverity"],
        marker="o",
        label="Observed",
    )

    plt.plot(
        ordered["RiskDecile"].astype(int),
        ordered["PredictedSeverity"],
        marker="o",
        label="Predicted",
    )

    plt.xlabel(
        "Predicted-severity decile "
        "(1 = lowest, 10 = highest)"
    )

    plt.ylabel("Average claim severity")

    plt.title(
        "Observed and Predicted Severity "
        "by Risk Decile"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "observed_predicted_severity_by_decile.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_distribution_figure(
    dataframe: pd.DataFrame,
) -> None:
    """Save the positive severity distribution."""

    severity = dataframe[TARGET]

    logarithmic_bins = np.logspace(
        np.log10(severity.min()),
        np.log10(severity.max()),
        60,
    )

    plt.figure(figsize=(9, 5))

    plt.hist(
        severity,
        bins=logarithmic_bins,
    )

    plt.xscale("log")

    plt.xlabel(
        "Average claim severity — logarithmic scale"
    )

    plt.ylabel("Number of policies")

    plt.title(
        "Distribution of Policy-Level "
        "Average Claim Severity"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "claim_severity_distribution_log.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_bonus_malus_figure(
    segment_calibration: pd.DataFrame,
) -> None:
    """Save observed and predicted severity by bonus-malus band."""

    plot_data = segment_calibration.loc[
        segment_calibration[
            "SegmentType"
        ].eq("BonusMalusBand")
        & segment_calibration[
            "EligibleForComparison"
        ]
    ].copy()

    positions = np.arange(
        len(plot_data)
    )

    width = 0.38

    plt.figure(figsize=(10, 5))

    plt.bar(
        positions - width / 2,
        plot_data["ObservedSeverity"],
        width=width,
        label="Observed",
    )

    plt.bar(
        positions + width / 2,
        plot_data["PredictedSeverity"],
        width=width,
        label="Predicted",
    )

    plt.xticks(
        positions,
        plot_data["Segment"],
        rotation=45,
        ha="right",
    )

    plt.xlabel("Bonus-malus band")
    plt.ylabel("Average claim severity")

    plt.title(
        "Observed and Predicted Severity "
        "by Bonus-Malus Band"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "bonus_malus_severity_calibration.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def create_documentation(
    split_summary: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    selected_model: str,
    calibration: pd.DataFrame,
    tail_summary: pd.DataFrame,
    gamma_iterations: int,
    boosting_iterations: int,
) -> None:
    """Generate severity-model documentation."""

    selected_test = test_metrics.loc[
        test_metrics["Model"].eq(
            selected_model
        )
    ].iloc[0]

    lowest_decile = (
        calibration.sort_values(
            "RiskDecile"
        )
        .iloc[0]
    )

    highest_decile = (
        calibration.sort_values(
            "RiskDecile"
        )
        .iloc[-1]
    )

    top_one_percent_share = (
        tail_summary.loc[
            tail_summary["Metric"].eq(
                "Top 1% policy share of claim amount"
            ),
            "Value",
        ].iloc[0]
    )

    validation_table = (
        validation_metrics[
            [
                "Model",
                "WeightedGammaDeviance",
                "D2Gamma",
                "ObservedClaimAmount",
                "PredictedClaimAmount",
                "PredictedToObservedAmountRatio",
            ]
        ]
        .to_markdown(index=False)
    )

    test_table = (
        test_metrics[
            [
                "Model",
                "WeightedGammaDeviance",
                "D2Gamma",
                "WeightedMAE",
                "WeightedRMSE",
                "ObservedClaimAmount",
                "PredictedClaimAmount",
                "PredictedToObservedAmountRatio",
            ]
        ]
        .to_markdown(index=False)
    )

    documentation = f"""# Claim-Severity Model

## Objective

The severity model estimates average claim amount conditional on a
policy having complete positive claim information.

The target is:

`CompleteTotalClaimAmount / ClaimNb`

Policies are weighted by `ClaimNb`, so policies representing multiple
claims contribute proportionally more information.

## Data Splits

{split_summary.to_markdown(index=False)}

Severity deciles were used for stratification so that the highly skewed
target distribution remained broadly similar across training,
validation, and test samples.

## Models

### Constant Severity Baseline

The baseline predicts the training sample's claim-weighted mean severity
for every policy.

### Gamma Regression

A regularized Gamma generalized linear model was fitted using a log
link.

Numerical predictors were standardized and categorical predictors were
one-hot encoded.

The model completed {gamma_iterations:,} optimization iterations.

### Histogram Gradient Boosting

The nonlinear model used Gamma loss and native categorical-variable
handling.

It completed {boosting_iterations:,} boosting iterations.

## Validation Results

{validation_table}

The preferred model was selected using the lowest validation
claim-weighted Gamma deviance.

**Selected model:** `{selected_model}`

## Test Results

{test_table}

For the selected model:

- Claim-weighted Gamma deviance:
  {selected_test["WeightedGammaDeviance"]:.6f}
- Gamma D2:
  {selected_test["D2Gamma"]:.4f}
- Weighted MAE:
  {selected_test["WeightedMAE"]:,.2f}
- Weighted RMSE:
  {selected_test["WeightedRMSE"]:,.2f}
- Observed claim amount:
  {selected_test["ObservedClaimAmount"]:,.2f}
- Predicted claim amount:
  {selected_test["PredictedClaimAmount"]:,.2f}
- Predicted-to-observed amount ratio:
  {selected_test["PredictedToObservedAmountRatio"]:.4f}
- Observed mean severity:
  {selected_test["ObservedMeanSeverity"]:,.2f}
- Predicted mean severity:
  {selected_test["PredictedMeanSeverity"]:,.2f}

## Risk-Decile Calibration

The lowest predicted-severity decile had:

- Observed severity:
  {lowest_decile["ObservedSeverity"]:,.2f}
- Predicted severity:
  {lowest_decile["PredictedSeverity"]:,.2f}

The highest predicted-severity decile had:

- Observed severity:
  {highest_decile["ObservedSeverity"]:,.2f}
- Predicted severity:
  {highest_decile["PredictedSeverity"]:,.2f}

The top 1% of test policies accounted for approximately
{top_one_percent_share:.2%} of the observed test claim amount.

## Interpretation

The severity model is conditional on having a claim.

It does not predict whether a policy will have a claim and should not be
applied directly to zero-claim policies as an expected policy cost.

Expected loss cost will later be estimated by combining:

`Predicted claim frequency × Predicted claim severity`

## Extreme Claims

Large claims were retained.

They may represent legitimate insurance losses and are financially
important. The Gamma loss and logarithmic-link models are designed for
positive, right-skewed targets, but extreme losses can still produce
substantial prediction error.

The largest-error report is included for audit and sensitivity analysis,
not as an automatic removal list.

## Limitations

- Severity is available only for complete positive-claim policies.
- The severity sample is much smaller than the frequency sample.
- Large claims create substantial random variation.
- The dataset does not contain detailed accident or damage information.
- Group-level calibration can differ from overall calibration.
- Predictive associations should not be interpreted as causal.
- The model should not be the sole basis for consequential insurance
  decisions.

## Next Step

Frequency and severity predictions will be combined to estimate expected
claim cost per exposure-year.

The two-part estimate will then be compared with a direct Tweedie
pure-premium model.
"""

    DOCUMENTATION_PATH.write_text(
        documentation,
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete severity-modelling workflow."""

    create_directories()

    print("Loading severity-modelling data...")

    model_data = load_modelling_data()

    print("Creating stratified severity splits...")

    train, validation, test = (
        split_modelling_data(
            model_data
        )
    )

    split_summary = create_split_summary(
        train,
        validation,
        test,
    )

    split_summary.to_csv(
        REPORT_DIR
        / "split_summary.csv",
        index=False,
    )

    X_train = train[MODEL_FEATURES]
    y_train = train[TARGET]
    training_claim_weights = train[
        "ClaimNb"
    ]

    X_validation = validation[
        MODEL_FEATURES
    ]

    X_test = test[MODEL_FEATURES]

    training_mean_severity = float(
        np.average(
            y_train,
            weights=training_claim_weights,
        )
    )

    print(
        "Training regularized Gamma regression..."
    )

    gamma_model = build_gamma_pipeline()

    gamma_model.fit(
        X_train,
        y_train,
        regressor__sample_weight=(
            training_claim_weights
        ),
    )

    print(
        "Training histogram gradient boosting "
        "with Gamma loss..."
    )

    boosting_model = build_boosting_model()

    boosting_model.fit(
        X_train,
        y_train,
        sample_weight=training_claim_weights,
    )

    baseline_validation_severity = np.full(
        len(validation),
        training_mean_severity,
    )

    gamma_validation_severity = (
        gamma_model.predict(
            X_validation
        )
    )

    boosting_validation_severity = (
        boosting_model.predict(
            X_validation
        )
    )

    validation_metrics = pd.DataFrame(
        [
            calculate_severity_metrics(
                model_name=(
                    "Constant severity baseline"
                ),
                sample_name="Validation",
                dataframe=validation,
                predicted_severity=(
                    baseline_validation_severity
                ),
            ),
            calculate_severity_metrics(
                model_name="Gamma regression",
                sample_name="Validation",
                dataframe=validation,
                predicted_severity=(
                    gamma_validation_severity
                ),
            ),
            calculate_severity_metrics(
                model_name=(
                    "Histogram gradient boosting"
                ),
                sample_name="Validation",
                dataframe=validation,
                predicted_severity=(
                    boosting_validation_severity
                ),
            ),
        ]
    )

    validation_metrics.to_csv(
        REPORT_DIR
        / "validation_severity_model_comparison.csv",
        index=False,
    )

    selected_model = (
        validation_metrics.sort_values(
           [
              "WeightedGammaDeviance",
              "D2Gamma",
           ],
           ascending=[True, False],
        )
        .iloc[0]["Model"]
    )

    print(
        "Validation-selected severity model:",
        selected_model,
    )

    baseline_test_severity = np.full(
        len(test),
        training_mean_severity,
    )

    gamma_test_severity = (
        gamma_model.predict(
            X_test
        )
    )

    boosting_test_severity = (
        boosting_model.predict(
            X_test
        )
    )

    test_metrics = pd.DataFrame(
        [
            calculate_severity_metrics(
                model_name=(
                    "Constant severity baseline"
                ),
                sample_name="Test",
                dataframe=test,
                predicted_severity=(
                    baseline_test_severity
                ),
            ),
            calculate_severity_metrics(
                model_name="Gamma regression",
                sample_name="Test",
                dataframe=test,
                predicted_severity=(
                    gamma_test_severity
                ),
            ),
            calculate_severity_metrics(
                model_name=(
                    "Histogram gradient boosting"
                ),
                sample_name="Test",
                dataframe=test,
                predicted_severity=(
                    boosting_test_severity
                ),
            ),
        ]
    )

    test_metrics.to_csv(
        REPORT_DIR
        / "test_severity_model_comparison.csv",
        index=False,
    )

    if selected_model == "Constant severity baseline":
       selected_test_severity = (
        baseline_test_severity
        )
    elif selected_model == "Gamma regression":
        selected_test_severity = (
        gamma_test_severity
        )
    else:
       selected_test_severity = (
        boosting_test_severity
        )

    calibration = create_calibration_deciles(
        dataframe=test,
        predicted_severity=(
            selected_test_severity
        ),
        model_name=selected_model,
    )

    calibration.to_csv(
        REPORT_DIR
        / "severity_calibration_deciles.csv",
        index=False,
    )

    segment_calibration = (
        create_segment_calibration(
            dataframe=test,
            predicted_severity=(
                selected_test_severity
            ),
            model_name=selected_model,
        )
    )

    segment_calibration.to_csv(
        REPORT_DIR
        / "segment_severity_calibration.csv",
        index=False,
    )

    gamma_coefficients = (
        extract_gamma_coefficients(
            gamma_model
        )
    )

    gamma_coefficients.to_csv(
        REPORT_DIR
        / "gamma_coefficients.csv",
        index=False,
    )

    tail_summary = create_tail_summary(
        test
    )

    tail_summary.to_csv(
        REPORT_DIR
        / "severity_tail_summary.csv",
        index=False,
    )

    largest_errors = (
        create_largest_error_report(
            dataframe=test,
            predicted_severity=(
                selected_test_severity
            ),
        )
    )

    largest_errors.to_csv(
        REPORT_DIR
        / "largest_severity_errors.csv",
        index=False,
    )

    selected_summary = test_metrics.loc[
        test_metrics["Model"].eq(
            selected_model
        )
    ].copy()

    selected_summary.to_csv(
        REPORT_DIR
        / "selected_severity_model_summary.csv",
        index=False,
    )

    predictions = test[
        [
            "IDpol",
            "ClaimNb",
            "CompleteTotalClaimAmount",
            TARGET,
        ]
    ].copy()

    predictions[
        "BaselinePredictedSeverity"
    ] = baseline_test_severity

    predictions[
        "GammaPredictedSeverity"
    ] = positive_predictions(
        gamma_test_severity
    )

    predictions[
        "BoostingPredictedSeverity"
    ] = positive_predictions(
        boosting_test_severity
    )

    predictions[
        "SelectedModel"
    ] = selected_model

    predictions[
        "SelectedPredictedSeverity"
    ] = positive_predictions(
        selected_test_severity
    )

    predictions[
        "SelectedPredictedClaimAmount"
    ] = (
        predictions[
            "SelectedPredictedSeverity"
        ]
        * predictions["ClaimNb"]
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("Saving severity models...")

    joblib.dump(
        gamma_model,
        GAMMA_MODEL_PATH,
    )

    joblib.dump(
        boosting_model,
        BOOSTING_MODEL_PATH,
    )

    print("Creating severity figures...")

    save_deviance_comparison(
        test_metrics
    )

    save_calibration_scatter(
        calibration
    )

    save_decile_severity_figure(
        calibration
    )

    save_distribution_figure(
        test
    )

    save_bonus_malus_figure(
        segment_calibration
    )

    gamma_iterations = int(
        gamma_model.named_steps[
            "regressor"
        ].n_iter_
    )

    boosting_iterations = int(
        boosting_model.n_iter_
    )

    create_documentation(
        split_summary=split_summary,
        validation_metrics=(
            validation_metrics
        ),
        test_metrics=test_metrics,
        selected_model=selected_model,
        calibration=calibration,
        tail_summary=tail_summary,
        gamma_iterations=gamma_iterations,
        boosting_iterations=(
            boosting_iterations
        ),
    )

    print(
        "\nStep 11 completed successfully."
    )

    print(
        "Training policies:",
        f"{len(train):,}",
    )

    print(
        "Validation policies:",
        f"{len(validation):,}",
    )

    print(
        "Test policies:",
        f"{len(test):,}",
    )

    print(
        "Training claims:",
        f"{train['ClaimNb'].sum():,}",
    )

    print(
        "Validation claims:",
        f"{validation['ClaimNb'].sum():,}",
    )

    print(
        "Test claims:",
        f"{test['ClaimNb'].sum():,}",
    )

    print(
        "Training weighted mean severity:",
        f"{training_mean_severity:,.2f}",
    )

    print(
        "Gamma-regression iterations:",
        gamma_iterations,
    )

    print(
        "Gradient-boosting iterations:",
        boosting_iterations,
    )

    print(
        "Validation-selected severity model:",
        selected_model,
    )

    print("\nValidation results:")

    print(
        validation_metrics.to_string(
            index=False
        )
    )

    print("\nTest results:")

    print(
        test_metrics.to_string(
            index=False
        )
    )

    print(
        "\nHighest predicted-severity decile:"
    )

    print(
        calibration.tail(1).to_string(
            index=False
        )
    )

    print(
        "\nReports saved to:",
        REPORT_DIR,
    )


if __name__ == "__main__":
    main()