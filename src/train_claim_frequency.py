"""Train and evaluate exposure-adjusted claim-frequency models.

The workflow compares:

1. A constant portfolio-frequency baseline.
2. Regularized Poisson regression.
3. Histogram gradient boosting with Poisson loss.

The target is ClaimNb divided by Exposure. Exposure is passed as the
sample weight, following the standard scaled-Poisson frequency
formulation.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    d2_tweedie_score,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "policy_analytics.csv"
)

POISSON_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "claim_frequency_poisson.joblib"
)

BOOSTING_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "claim_frequency_hist_gradient_boosting.joblib"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "claim_frequency_test_predictions.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "claim_frequency"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "modeling"
    / "claim_frequency"
)

DOCUMENTATION_PATH = (
    PROJECT_ROOT
    / "docs"
    / "claim_frequency_model.md"
)

RANDOM_STATE = 42
EPSILON = 1e-12

TARGET_COUNT = "ClaimNb"
TARGET_RATE = "ClaimFrequency"

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
    """Create all model-output directories."""

    for directory in [
        POISSON_MODEL_PATH.parent,
        REPORT_DIR,
        FIGURES_DIR,
        DOCUMENTATION_PATH.parent,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def load_modelling_data() -> pd.DataFrame:
    """Load and prepare the frequency-modelling data."""

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            "Policy analytics data was not found. "
            "Run src/prepare_data.py first."
        )

    policy = pd.read_csv(
        POLICY_PATH,
        low_memory=False,
    )

    policy["FrequencyModelEligible"] = (
        convert_to_boolean(
            policy["FrequencyModelEligible"]
        )
    )

    model_data = policy.loc[
        policy["FrequencyModelEligible"],
        [
            "IDpol",
            "Exposure",
            "ClaimNb",
            "Area",
            "VehPower",
            "VehAge",
            "DrivAge",
            "BonusMalus",
            "VehBrand",
            "VehGas",
            "Density",
            "Region",
        ],
    ].copy()

    numeric_columns = [
        "Exposure",
        "ClaimNb",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "Density",
    ]

    for column in numeric_columns:
        model_data[column] = pd.to_numeric(
            model_data[column],
            errors="coerce",
        )

    if model_data["Exposure"].isna().any():
        raise ValueError(
            "Exposure contains missing values."
        )

    if model_data["Exposure"].le(0).any():
        raise ValueError(
            "Exposure contains non-positive values."
        )

    if model_data["ClaimNb"].isna().any():
        raise ValueError(
            "ClaimNb contains missing values."
        )

    if model_data["ClaimNb"].lt(0).any():
        raise ValueError(
            "ClaimNb contains negative values."
        )

    if model_data["Density"].lt(0).any():
        raise ValueError(
            "Density contains negative values."
        )

    model_data["ClaimNb"] = (
        model_data["ClaimNb"]
        .astype("int64")
    )

    model_data["LogDensity"] = np.log1p(
        model_data["Density"]
    )

    model_data[TARGET_RATE] = (
        model_data["ClaimNb"]
        / model_data["Exposure"]
    )

    model_data["ClaimCountStratum"] = np.select(
        [
            model_data["ClaimNb"].eq(0),
            model_data["ClaimNb"].eq(1),
        ],
        [
            "0 claims",
            "1 claim",
        ],
        default="2+ claims",
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
    """Create stratified 60/20/20 frequency samples."""

    development, test = train_test_split(
        model_data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=model_data[
            "ClaimCountStratum"
        ],
    )

    train, validation = train_test_split(
        development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=development[
            "ClaimCountStratum"
        ],
    )

    return train, validation, test


def create_split_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the frequency-modelling samples."""

    rows = []

    for sample_name, dataframe in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        total_exposure = dataframe[
            "Exposure"
        ].sum()

        total_claims = dataframe[
            "ClaimNb"
        ].sum()

        rows.append(
            {
                "Sample": sample_name,
                "Policies": len(dataframe),
                "Exposure": total_exposure,
                "Claims": total_claims,
                "ZeroClaimPolicies": int(
                    dataframe[
                        "ClaimNb"
                    ].eq(0).sum()
                ),
                "OneClaimPolicies": int(
                    dataframe[
                        "ClaimNb"
                    ].eq(1).sum()
                ),
                "MultipleClaimPolicies": int(
                    dataframe[
                        "ClaimNb"
                    ].ge(2).sum()
                ),
                "ObservedFrequency": (
                    total_claims
                    / total_exposure
                ),
                "ClaimsPer100ExposureYears": (
                    100
                    * total_claims
                    / total_exposure
                ),
            }
        )

    return pd.DataFrame(rows)


def build_poisson_pipeline() -> Pipeline:
    """Create the regularized Poisson-regression pipeline."""

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

    regressor = PoissonRegressor(
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
    """Create the nonlinear Poisson boosting model."""

    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
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
    """Ensure predictions satisfy Poisson metric requirements."""

    return np.clip(
        np.asarray(predictions),
        EPSILON,
        None,
    )


def calculate_frequency_metrics(
    model_name: str,
    sample_name: str,
    dataframe: pd.DataFrame,
    predicted_frequency: np.ndarray,
) -> dict:
    """Calculate rate, count, deviance, and calibration metrics."""

    observed_count = (
        dataframe["ClaimNb"]
        .to_numpy()
    )

    exposure = (
        dataframe["Exposure"]
        .to_numpy()
    )

    observed_frequency = (
        observed_count / exposure
    )

    predicted_frequency = (
        positive_predictions(
            predicted_frequency
        )
    )

    predicted_count = (
        predicted_frequency
        * exposure
    )

    observed_total_claims = float(
        observed_count.sum()
    )

    predicted_total_claims = float(
        predicted_count.sum()
    )

    observed_portfolio_frequency = (
        observed_total_claims
        / exposure.sum()
    )

    predicted_portfolio_frequency = (
        predicted_total_claims
        / exposure.sum()
    )

    weighted_deviance = (
        mean_poisson_deviance(
            observed_frequency,
            predicted_frequency,
            sample_weight=exposure,
        )
    )

    count_deviance = (
        mean_poisson_deviance(
            observed_count,
            predicted_count,
        )
    )

    d2_score = d2_tweedie_score(
        observed_frequency,
        predicted_frequency,
        sample_weight=exposure,
        power=1,
    )

    weighted_frequency_mae = np.average(
        np.abs(
            observed_frequency
            - predicted_frequency
        ),
        weights=exposure,
    )

    count_mae = mean_absolute_error(
        observed_count,
        predicted_count,
    )

    count_rmse = np.sqrt(
        mean_squared_error(
            observed_count,
            predicted_count,
        )
    )

    pearson_chi_square = np.sum(
        (
            observed_count
            - predicted_count
        ) ** 2
        / np.clip(
            predicted_count,
            EPSILON,
            None,
        )
    )

    pearson_dispersion = (
        pearson_chi_square
        / len(dataframe)
    )

    return {
        "Model": model_name,
        "Sample": sample_name,
        "ExposureWeightedPoissonDeviance": (
            weighted_deviance
        ),
        "CountMeanPoissonDeviance": (
            count_deviance
        ),
        "D2Poisson": d2_score,
        "WeightedFrequencyMAE": (
            weighted_frequency_mae
        ),
        "CountMAE": count_mae,
        "CountRMSE": count_rmse,
        "ObservedClaims": (
            observed_total_claims
        ),
        "PredictedClaims": (
            predicted_total_claims
        ),
        "PredictedToObservedClaimsRatio": (
            predicted_total_claims
            / observed_total_claims
        ),
        "ObservedFrequency": (
            observed_portfolio_frequency
        ),
        "PredictedFrequency": (
            predicted_portfolio_frequency
        ),
        "FrequencyDifference": (
            predicted_portfolio_frequency
            - observed_portfolio_frequency
        ),
        "PearsonDispersionDiagnostic": (
            pearson_dispersion
        ),
    }


def create_calibration_deciles(
    dataframe: pd.DataFrame,
    predicted_frequency: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Create observed-versus-predicted risk deciles."""

    calibration_data = dataframe[
        [
            "IDpol",
            "Exposure",
            "ClaimNb",
        ]
    ].copy()

    calibration_data[
        "PredictedFrequency"
    ] = positive_predictions(
        predicted_frequency
    )

    calibration_data[
        "PredictedClaims"
    ] = (
        calibration_data[
            "PredictedFrequency"
        ]
        * calibration_data["Exposure"]
    )

    ranked_frequency = (
        calibration_data[
            "PredictedFrequency"
        ]
        .rank(
            method="first"
        )
    )

    calibration_data["RiskDecile"] = pd.qcut(
        ranked_frequency,
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
            Exposure=("Exposure", "sum"),
            ObservedClaims=("ClaimNb", "sum"),
            PredictedClaims=(
                "PredictedClaims",
                "sum",
            ),
            MinimumPredictedFrequency=(
                "PredictedFrequency",
                "min",
            ),
            MaximumPredictedFrequency=(
                "PredictedFrequency",
                "max",
            ),
            MeanPredictedFrequency=(
                "PredictedFrequency",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["ObservedFrequency"] = (
        summary["ObservedClaims"]
        / summary["Exposure"]
    )

    summary["PredictedFrequency"] = (
        summary["PredictedClaims"]
        / summary["Exposure"]
    )

    summary["ObservedToExpectedRatio"] = (
        summary["ObservedClaims"]
        / summary[
            "PredictedClaims"
        ].replace(0, np.nan)
    )

    summary.insert(
        0,
        "Model",
        model_name,
    )

    return summary


def add_frequency_bands(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create driver-age and bonus-malus bands."""

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
    predicted_frequency: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Summarize observed and predicted frequency by segment."""

    analysis_data = add_frequency_bands(
        dataframe
    )

    analysis_data[
        "PredictedFrequency"
    ] = positive_predictions(
        predicted_frequency
    )

    analysis_data[
        "PredictedClaims"
    ] = (
        analysis_data[
            "PredictedFrequency"
        ]
        * analysis_data["Exposure"]
    )

    group_columns = [
        "Area",
        "VehGas",
        "DrivAgeBand",
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
                Exposure=("Exposure", "sum"),
                ObservedClaims=(
                    "ClaimNb",
                    "sum",
                ),
                PredictedClaims=(
                    "PredictedClaims",
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

        summary["ObservedFrequency"] = (
            summary["ObservedClaims"]
            / summary["Exposure"]
        )

        summary["PredictedFrequency"] = (
            summary["PredictedClaims"]
            / summary["Exposure"]
        )

        summary[
            "ObservedToExpectedRatio"
        ] = (
            summary["ObservedClaims"]
            / summary[
                "PredictedClaims"
            ].replace(0, np.nan)
        )

        summary[
            "EligibleForComparison"
        ] = summary["Exposure"].ge(500)

        outputs.append(summary)

    return pd.concat(
        outputs,
        ignore_index=True,
    )


def extract_poisson_coefficients(
    model: Pipeline,
) -> pd.DataFrame:
    """Extract Poisson coefficients and multiplicative rate ratios."""

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

    output["RateRatio"] = np.exp(
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


def save_deviance_comparison(
    test_metrics: pd.DataFrame,
) -> None:
    """Save the test Poisson-deviance comparison."""

    plot_data = test_metrics.sort_values(
        "ExposureWeightedPoissonDeviance"
    )

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        plot_data["Model"],
        plot_data[
            "ExposureWeightedPoissonDeviance"
        ],
    )

    plt.title(
        "Exposure-Weighted Test Poisson Deviance"
    )

    plt.xlabel("Model")
    plt.ylabel("Mean Poisson deviance")
    plt.xticks(rotation=20, ha="right")

    for bar, value in zip(
        bars,
        plot_data[
            "ExposureWeightedPoissonDeviance"
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
        / "test_poisson_deviance_comparison.png",
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
            "ObservedFrequency"
        ].max(),
        calibration[
            "PredictedFrequency"
        ].max(),
    )

    plt.figure(figsize=(7, 6))

    plt.scatter(
        calibration[
            "PredictedFrequency"
        ],
        calibration[
            "ObservedFrequency"
        ],
    )

    for row in calibration.itertuples(
        index=False
    ):
        plt.annotate(
            str(row.RiskDecile),
            (
                row.PredictedFrequency,
                row.ObservedFrequency,
            ),
        )

    plt.plot(
        [0, maximum_value],
        [0, maximum_value],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.xlabel("Predicted claim frequency")
    plt.ylabel("Observed claim frequency")
    plt.title(
        "Claim-Frequency Calibration by Risk Decile"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "frequency_calibration_deciles.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_decile_frequency_figure(
    calibration: pd.DataFrame,
) -> None:
    """Save observed and predicted frequency by decile."""

    ordered = calibration.sort_values(
        "RiskDecile"
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        ordered["RiskDecile"].astype(int),
        (
            ordered["ObservedFrequency"]
            * 100
        ),
        marker="o",
        label="Observed",
    )

    plt.plot(
        ordered["RiskDecile"].astype(int),
        (
            ordered["PredictedFrequency"]
            * 100
        ),
        marker="o",
        label="Predicted",
    )

    plt.xlabel(
        "Predicted-frequency decile "
        "(1 = lowest, 10 = highest)"
    )

    plt.ylabel(
        "Claims per 100 exposure-years"
    )

    plt.title(
        "Observed and Predicted Frequency by Risk Decile"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "observed_predicted_frequency_by_decile.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_bonus_malus_figure(
    segment_calibration: pd.DataFrame,
) -> None:
    """Save observed and predicted bonus-malus frequency."""

    plot_data = segment_calibration.loc[
        segment_calibration[
            "SegmentType"
        ].eq("BonusMalusBand")
        & segment_calibration[
            "EligibleForComparison"
        ]
    ].copy()

    plt.figure(figsize=(10, 5))

    positions = np.arange(
        len(plot_data)
    )

    width = 0.38

    plt.bar(
        positions - width / 2,
        plot_data[
            "ObservedFrequency"
        ] * 100,
        width=width,
        label="Observed",
    )

    plt.bar(
        positions + width / 2,
        plot_data[
            "PredictedFrequency"
        ] * 100,
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
    plt.ylabel(
        "Claims per 100 exposure-years"
    )

    plt.title(
        "Observed and Predicted Frequency "
        "by Bonus-Malus Band"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "bonus_malus_frequency_calibration.png",
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
    poisson_iterations: int,
    boosting_iterations: int,
) -> None:
    """Generate claim-frequency model documentation."""

    selected_test = test_metrics.loc[
        test_metrics["Model"].eq(
            selected_model
        )
    ].iloc[0]

    highest_decile = (
        calibration.sort_values(
            "RiskDecile"
        )
        .iloc[-1]
    )

    lowest_decile = (
        calibration.sort_values(
            "RiskDecile"
        )
        .iloc[0]
    )

    validation_table = (
        validation_metrics[
            [
                "Model",
                "ExposureWeightedPoissonDeviance",
                "D2Poisson",
                "ObservedClaims",
                "PredictedClaims",
                "PredictedToObservedClaimsRatio",
            ]
        ]
        .to_markdown(index=False)
    )

    test_table = (
        test_metrics[
            [
                "Model",
                "ExposureWeightedPoissonDeviance",
                "D2Poisson",
                "CountMAE",
                "CountRMSE",
                "ObservedClaims",
                "PredictedClaims",
                "PredictedToObservedClaimsRatio",
            ]
        ]
        .to_markdown(index=False)
    )

    if (
        selected_test[
            "PearsonDispersionDiagnostic"
        ] > 1.2
    ):
        dispersion_text = (
            "The diagnostic is above one, indicating that "
            "residual variation remains larger than the "
            "basic conditional Poisson variance assumption."
        )
    else:
        dispersion_text = (
            "The diagnostic is close to one, suggesting that "
            "residual variation is reasonably consistent with "
            "the fitted Poisson mean structure."
        )

    documentation = f"""# Claim-Frequency Model

## Objective

The frequency model estimates the expected number of claims per
exposure-year.

The modelling target is:

`ClaimNb / Exposure`

Exposure is used as the sample weight. Predicted policy claim counts are
calculated by multiplying predicted annualized frequency by policy
exposure.

## Data Splits

{split_summary.to_markdown(index=False)}

The split was stratified using zero-claim, one-claim, and multiple-claim
policy categories.

## Models

### Constant Frequency Baseline

The baseline assigns every policy the training portfolio's overall
claim frequency.

### Poisson Regression

A regularized log-linear Poisson regression was fitted using numerical
standardization and categorical one-hot encoding.

The model completed {poisson_iterations:,} optimization iterations.

### Histogram Gradient Boosting

The nonlinear model used Poisson loss and native categorical-variable
handling.

It completed {boosting_iterations:,} boosting iterations.

## Validation Results

{validation_table}

The preferred model was selected using the lowest validation
exposure-weighted Poisson deviance.

**Selected model:** `{selected_model}`

## Test Results

{test_table}

For the selected model:

- Exposure-weighted Poisson deviance:
  {selected_test["ExposureWeightedPoissonDeviance"]:.6f}
- Poisson D2:
  {selected_test["D2Poisson"]:.4f}
- Observed claims:
  {selected_test["ObservedClaims"]:,.0f}
- Predicted claims:
  {selected_test["PredictedClaims"]:,.2f}
- Predicted-to-observed claims ratio:
  {selected_test["PredictedToObservedClaimsRatio"]:.4f}
- Observed claim frequency:
  {selected_test["ObservedFrequency"]:.6f}
- Predicted claim frequency:
  {selected_test["PredictedFrequency"]:.6f}
- Pearson dispersion diagnostic:
  {selected_test["PearsonDispersionDiagnostic"]:.4f}

{dispersion_text}

The Pearson value is used as an empirical diagnostic rather than a
formal hypothesis test, particularly for the nonlinear model.

## Risk-Decile Calibration

The lowest predicted-frequency decile had:

- Observed frequency:
  {lowest_decile["ObservedFrequency"]:.6f}
- Predicted frequency:
  {lowest_decile["PredictedFrequency"]:.6f}

The highest predicted-frequency decile had:

- Observed frequency:
  {highest_decile["ObservedFrequency"]:.6f}
- Predicted frequency:
  {highest_decile["PredictedFrequency"]:.6f}

A useful model should generally assign higher observed claim frequencies
to higher predicted-risk deciles.

## Interpretation

The frequency model estimates expected claim counts, not whether a claim
is certain to occur.

For example, an expected count of 0.10 means approximately one expected
claim per ten comparable exposure-years, not that a specific policy will
experience one-tenth of a claim.

## Limitations

- The data represents a historical French motor-insurance portfolio.
- The model assumes historical relationships are informative for the
  evaluation sample.
- The Poisson assumption may not capture all remaining variation.
- Segment-level deviations can remain even when overall calibration is
  strong.
- Predictive relationships should not be interpreted as causal.
- The model should not be the sole basis for consequential insurance
  decisions.

## Next Steps

The next modelling stage will estimate claim severity using complete
positive-claim policies.

Frequency and severity predictions will later be combined:

`Expected claim frequency × Expected claim severity`

to estimate expected claim cost per exposure-year.
"""

    DOCUMENTATION_PATH.write_text(
        documentation,
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete claim-frequency workflow."""

    create_directories()

    print("Loading frequency-modelling data...")
    model_data = load_modelling_data()

    print("Creating stratified frequency splits...")
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
    y_train = train[TARGET_RATE]
    training_exposure = train["Exposure"]

    X_validation = validation[
        MODEL_FEATURES
    ]

    X_test = test[MODEL_FEATURES]

    training_portfolio_frequency = (
        train["ClaimNb"].sum()
        / train["Exposure"].sum()
    )

    print("Training regularized Poisson regression...")

    poisson_model = build_poisson_pipeline()

    poisson_model.fit(
        X_train,
        y_train,
        regressor__sample_weight=(
            training_exposure
        ),
    )

    print(
        "Training histogram gradient boosting "
        "with Poisson loss..."
    )

    boosting_model = build_boosting_model()

    boosting_model.fit(
        X_train,
        y_train,
        sample_weight=training_exposure,
    )

    baseline_validation_frequency = np.full(
        len(validation),
        training_portfolio_frequency,
    )

    poisson_validation_frequency = (
        poisson_model.predict(
            X_validation
        )
    )

    boosting_validation_frequency = (
        boosting_model.predict(
            X_validation
        )
    )

    validation_metrics = pd.DataFrame(
        [
            calculate_frequency_metrics(
                model_name=(
                    "Constant frequency baseline"
                ),
                sample_name="Validation",
                dataframe=validation,
                predicted_frequency=(
                    baseline_validation_frequency
                ),
            ),
            calculate_frequency_metrics(
                model_name=(
                    "Poisson regression"
                ),
                sample_name="Validation",
                dataframe=validation,
                predicted_frequency=(
                    poisson_validation_frequency
                ),
            ),
            calculate_frequency_metrics(
                model_name=(
                    "Histogram gradient boosting"
                ),
                sample_name="Validation",
                dataframe=validation,
                predicted_frequency=(
                    boosting_validation_frequency
                ),
            ),
        ]
    )

    validation_metrics.to_csv(
        REPORT_DIR
        / "validation_frequency_model_comparison.csv",
        index=False,
    )

    trained_validation_metrics = (
        validation_metrics.loc[
            ~validation_metrics["Model"].eq(
                "Constant frequency baseline"
            )
        ]
    )

    selected_model = (
        trained_validation_metrics.sort_values(
            [
                "ExposureWeightedPoissonDeviance",
                "D2Poisson",
            ],
            ascending=[True, False],
        )
        .iloc[0]["Model"]
    )

    print(
        "Validation-selected frequency model:",
        selected_model,
    )

    baseline_test_frequency = np.full(
        len(test),
        training_portfolio_frequency,
    )

    poisson_test_frequency = (
        poisson_model.predict(
            X_test
        )
    )

    boosting_test_frequency = (
        boosting_model.predict(
            X_test
        )
    )

    test_metrics = pd.DataFrame(
        [
            calculate_frequency_metrics(
                model_name=(
                    "Constant frequency baseline"
                ),
                sample_name="Test",
                dataframe=test,
                predicted_frequency=(
                    baseline_test_frequency
                ),
            ),
            calculate_frequency_metrics(
                model_name=(
                    "Poisson regression"
                ),
                sample_name="Test",
                dataframe=test,
                predicted_frequency=(
                    poisson_test_frequency
                ),
            ),
            calculate_frequency_metrics(
                model_name=(
                    "Histogram gradient boosting"
                ),
                sample_name="Test",
                dataframe=test,
                predicted_frequency=(
                    boosting_test_frequency
                ),
            ),
        ]
    )

    test_metrics.to_csv(
        REPORT_DIR
        / "test_frequency_model_comparison.csv",
        index=False,
    )

    if selected_model == "Poisson regression":
        selected_test_frequency = (
            poisson_test_frequency
        )
    else:
        selected_test_frequency = (
            boosting_test_frequency
        )

    calibration = create_calibration_deciles(
        dataframe=test,
        predicted_frequency=(
            selected_test_frequency
        ),
        model_name=selected_model,
    )

    calibration.to_csv(
        REPORT_DIR
        / "frequency_calibration_deciles.csv",
        index=False,
    )

    segment_calibration = (
        create_segment_calibration(
            dataframe=test,
            predicted_frequency=(
                selected_test_frequency
            ),
            model_name=selected_model,
        )
    )

    segment_calibration.to_csv(
        REPORT_DIR
        / "segment_frequency_calibration.csv",
        index=False,
    )

    coefficients = (
        extract_poisson_coefficients(
            poisson_model
        )
    )

    coefficients.to_csv(
        REPORT_DIR
        / "poisson_coefficients.csv",
        index=False,
    )

    selected_summary = test_metrics.loc[
        test_metrics["Model"].eq(
            selected_model
        )
    ].copy()

    selected_summary.to_csv(
        REPORT_DIR
        / "selected_frequency_model_summary.csv",
        index=False,
    )

    predictions = test[
        [
            "IDpol",
            "Exposure",
            "ClaimNb",
        ]
    ].copy()

    predictions[
        "ObservedFrequency"
    ] = (
        predictions["ClaimNb"]
        / predictions["Exposure"]
    )

    predictions[
        "BaselinePredictedFrequency"
    ] = baseline_test_frequency

    predictions[
        "PoissonPredictedFrequency"
    ] = positive_predictions(
        poisson_test_frequency
    )

    predictions[
        "BoostingPredictedFrequency"
    ] = positive_predictions(
        boosting_test_frequency
    )

    predictions[
        "SelectedModel"
    ] = selected_model

    predictions[
        "SelectedPredictedFrequency"
    ] = positive_predictions(
        selected_test_frequency
    )

    predictions[
        "SelectedExpectedClaims"
    ] = (
        predictions[
            "SelectedPredictedFrequency"
        ]
        * predictions["Exposure"]
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("Saving frequency models...")

    joblib.dump(
        poisson_model,
        POISSON_MODEL_PATH,
    )

    joblib.dump(
        boosting_model,
        BOOSTING_MODEL_PATH,
    )

    print("Creating frequency-model figures...")

    save_deviance_comparison(
        test_metrics
    )

    save_calibration_scatter(
        calibration
    )

    save_decile_frequency_figure(
        calibration
    )

    save_bonus_malus_figure(
        segment_calibration
    )

    poisson_iterations = int(
        poisson_model.named_steps[
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
        poisson_iterations=(
            poisson_iterations
        ),
        boosting_iterations=(
            boosting_iterations
        ),
    )

    print(
        "\nStep 10 completed successfully."
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
        "Training portfolio frequency:",
        f"{training_portfolio_frequency:.6f}",
    )

    print(
        "Poisson-regression iterations:",
        poisson_iterations,
    )

    print(
        "Gradient-boosting iterations:",
        boosting_iterations,
    )

    print(
        "Validation-selected frequency model:",
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
        "\nHighest-risk calibration decile:"
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