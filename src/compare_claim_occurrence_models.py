"""Compare logistic regression with histogram gradient boosting.

The workflow:

1. Recreates the Step 8 stratified data splits.
2. Loads the converged logistic-regression pipeline.
3. Trains a histogram gradient-boosting classifier.
4. Selects thresholds using validation F2-score.
5. Selects the preferred model using validation PR-AUC.
6. Evaluates both models on the untouched test sample.
7. Performs operational-capacity and subgroup analyses.
8. Calculates test-sample permutation feature importance.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "policy_analytics.csv"
)

LOGISTIC_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "claim_occurrence_logistic.joblib"
)

TREE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "claim_occurrence_hist_gradient_boosting.joblib"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "claim_occurrence_model_comparison_predictions.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "claim_occurrence_comparison"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "modeling"
    / "claim_occurrence_comparison"
)

DOCUMENTATION_PATH = (
    PROJECT_ROOT
    / "docs"
    / "claim_occurrence_model_comparison.md"
)

RANDOM_STATE = 42
TARGET = "HasClaim"

NUMERIC_FEATURES = [
    "LogExposure",
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

CAPACITY_LEVELS = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

MINIMUM_SUBGROUP_ROWS = 1_000
MINIMUM_SUBGROUP_CLAIMS = 30


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
    """Create the output directories."""

    for directory in [
        TREE_MODEL_PATH.parent,
        REPORT_DIR,
        FIGURES_DIR,
        DOCUMENTATION_PATH.parent,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def load_modelling_data() -> pd.DataFrame:
    """Load the same occurrence-modelling sample used in Step 8."""

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            "Policy analytics data was not found."
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
            "Area",
            "VehPower",
            "VehAge",
            "DrivAge",
            "BonusMalus",
            "VehBrand",
            "VehGas",
            "Density",
            "Region",
            TARGET,
        ],
    ].copy()

    model_data[TARGET] = pd.to_numeric(
        model_data[TARGET],
        errors="raise",
    ).astype("int8")

    numeric_source_columns = [
        "Exposure",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "Density",
    ]

    for column in numeric_source_columns:
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

    if model_data["Density"].lt(0).any():
        raise ValueError(
            "Density contains negative values."
        )

    model_data["LogExposure"] = np.log(
        model_data["Exposure"]
    )

    model_data["LogDensity"] = np.log1p(
        model_data["Density"]
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
    """Recreate the Step 8 stratified 60/20/20 split."""

    development, test = train_test_split(
        model_data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=model_data[TARGET],
    )

    train, validation = train_test_split(
        development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=development[TARGET],
    )

    return train, validation, test


def select_f2_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Select the probability threshold maximizing F2."""

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    threshold_precision = precision[:-1]
    threshold_recall = recall[:-1]

    numerator = (
        5
        * threshold_precision
        * threshold_recall
    )

    denominator = (
        4 * threshold_precision
        + threshold_recall
    )

    f2_scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )

    best_index = int(
        np.nanargmax(f2_scores)
    )

    return (
        float(thresholds[best_index]),
        float(f2_scores[best_index]),
    )


def calculate_metrics(
    model_name: str,
    sample_name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate classification and probability metrics."""

    predictions = (
        probabilities >= threshold
    ).astype("int8")

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "Model": model_name,
        "Sample": sample_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Specificity": specificity,
        "F1Score": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F2Score": fbeta_score(
            y_true,
            predictions,
            beta=2,
            zero_division=0,
        ),
        "ROCAUC": roc_auc_score(
            y_true,
            probabilities,
        ),
        "PRAUC": average_precision_score(
            y_true,
            probabilities,
        ),
        "BrierScore": brier_score_loss(
            y_true,
            probabilities,
        ),
        "TrueNegatives": int(tn),
        "FalsePositives": int(fp),
        "FalseNegatives": int(fn),
        "TruePositives": int(tp),
        "PredictedPositiveRate": float(
            predictions.mean()
        ),
        "MeanPredictedProbability": float(
            probabilities.mean()
        ),
        "ObservedClaimRate": float(
            y_true.mean()
        ),
    }


def create_capacity_analysis(
    model_name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Evaluate performance at fixed review-capacity levels."""

    y_array = np.asarray(y_true)
    probability_array = np.asarray(
        probabilities
    )

    ranked_indices = np.argsort(
        -probability_array
    )

    total_claims = int(
        y_array.sum()
    )

    portfolio_claim_rate = float(
        y_array.mean()
    )

    rows = []

    for capacity in CAPACITY_LEVELS:
        selected_count = max(
            1,
            int(
                np.ceil(
                    len(y_array)
                    * capacity
                )
            ),
        )

        selected_indices = ranked_indices[
            :selected_count
        ]

        selected_claims = int(
            y_array[
                selected_indices
            ].sum()
        )

        precision_at_capacity = (
            selected_claims
            / selected_count
        )

        claim_capture_rate = (
            selected_claims
            / total_claims
            if total_claims > 0
            else np.nan
        )

        lift = (
            precision_at_capacity
            / portfolio_claim_rate
            if portfolio_claim_rate > 0
            else np.nan
        )

        threshold_at_capacity = float(
            probability_array[
                selected_indices
            ].min()
        )

        rows.append(
            {
                "Model": model_name,
                "Capacity": capacity,
                "SelectedPolicies": (
                    selected_count
                ),
                "SelectedClaims": (
                    selected_claims
                ),
                "PrecisionAtCapacity": (
                    precision_at_capacity
                ),
                "ClaimCaptureRate": (
                    claim_capture_rate
                ),
                "Lift": lift,
                "ThresholdAtCapacity": (
                    threshold_at_capacity
                ),
            }
        )

    return pd.DataFrame(rows)


def create_driver_age_band(
    series: pd.Series,
) -> pd.Series:
    """Create the driver-age bands used in Step 6."""

    return pd.cut(
        series,
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


def create_subgroup_metrics(
    test_data: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
) -> pd.DataFrame:
    """Evaluate the selected model across policy subgroups."""

    analysis_data = test_data[
        [
            TARGET,
            "Area",
            "VehGas",
            "DrivAge",
        ]
    ].copy()

    analysis_data[
        "PredictedProbability"
    ] = probabilities

    analysis_data[
        "PredictedPositive"
    ] = (
        probabilities >= threshold
    ).astype("int8")

    analysis_data["DrivAgeBand"] = (
        create_driver_age_band(
            analysis_data["DrivAge"]
        )
    )

    group_columns = [
        "Area",
        "VehGas",
        "DrivAgeBand",
    ]

    rows = []

    for group_column in group_columns:
        group_values = (
            analysis_data[group_column]
            .astype("string")
            .fillna("Missing")
        )

        temporary = analysis_data.copy()
        temporary["GroupValue"] = (
            group_values
        )

        for group_value, group in (
            temporary.groupby(
                "GroupValue",
                dropna=False,
            )
        ):
            number_of_rows = len(group)
            number_of_claims = int(
                group[TARGET].sum()
            )

            if (
                number_of_rows
                < MINIMUM_SUBGROUP_ROWS
                or number_of_claims
                < MINIMUM_SUBGROUP_CLAIMS
            ):
                continue

            predictions = group[
                "PredictedPositive"
            ]

            tn, fp, fn, tp = confusion_matrix(
                group[TARGET],
                predictions,
                labels=[0, 1],
            ).ravel()

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0
            )

            specificity = (
                tn / (tn + fp)
                if (tn + fp) > 0
                else np.nan
            )

            observed_rate = float(
                group[TARGET].mean()
            )

            mean_probability = float(
                group[
                    "PredictedProbability"
                ].mean()
            )

            rows.append(
                {
                    "Model": model_name,
                    "GroupType": group_column,
                    "GroupValue": str(
                        group_value
                    ),
                    "Policies": number_of_rows,
                    "Claims": number_of_claims,
                    "ObservedClaimRate": (
                        observed_rate
                    ),
                    "MeanPredictedProbability": (
                        mean_probability
                    ),
                    "CalibrationDifference": (
                        mean_probability
                        - observed_rate
                    ),
                    "SelectionRate": float(
                        predictions.mean()
                    ),
                    "Precision": precision,
                    "Recall": recall,
                    "Specificity": specificity,
                    "ROCAUC": roc_auc_score(
                        group[TARGET],
                        group[
                            "PredictedProbability"
                        ],
                    ),
                    "PRAUC": (
                        average_precision_score(
                            group[TARGET],
                            group[
                                "PredictedProbability"
                            ],
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def create_permutation_importance(
    model: HistGradientBoostingClassifier,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate permutation importance on a test subsample."""

    sample_size = min(
        30_000,
        len(test_data),
    )

    sample_indices, _ = train_test_split(
        test_data.index,
        train_size=sample_size,
        random_state=RANDOM_STATE,
        stratify=test_data[TARGET],
    )

    importance_sample = test_data.loc[
        sample_indices
    ]

    X_importance = importance_sample[
        MODEL_FEATURES
    ]

    y_importance = importance_sample[
        TARGET
    ]

    result = permutation_importance(
        model,
        X_importance,
        y_importance,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_table = pd.DataFrame(
        {
            "Feature": MODEL_FEATURES,
            "ImportanceMean": (
                result.importances_mean
            ),
            "ImportanceStandardDeviation": (
                result.importances_std
            ),
        }
    )

    return importance_table.sort_values(
        "ImportanceMean",
        ascending=False,
    )


def save_roc_comparison(
    y_true: pd.Series,
    model_probabilities: dict[str, np.ndarray],
) -> None:
    """Save the ROC comparison figure."""

    plt.figure(figsize=(8, 6))

    for model_name, probabilities in (
        model_probabilities.items()
    ):
        false_positive_rate, true_positive_rate, _ = (
            roc_curve(
                y_true,
                probabilities,
            )
        )

        auc = roc_auc_score(
            y_true,
            probabilities,
        )

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=(
                f"{model_name}: "
                f"AUC = {auc:.3f}"
            ),
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random ranking",
    )

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Claim-Occurrence ROC Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "roc_curve_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_pr_comparison(
    y_true: pd.Series,
    model_probabilities: dict[str, np.ndarray],
) -> None:
    """Save the precision-recall comparison figure."""

    plt.figure(figsize=(8, 6))

    for model_name, probabilities in (
        model_probabilities.items()
    ):
        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                probabilities,
            )
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"{model_name}: "
                f"PR-AUC = {pr_auc:.3f}"
            ),
        )

    prevalence = float(
        y_true.mean()
    )

    plt.axhline(
        prevalence,
        linestyle="--",
        label=(
            f"Claim-rate baseline = "
            f"{prevalence:.3f}"
        ),
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "Claim-Occurrence "
        "Precision-Recall Comparison"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "precision_recall_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_calibration_comparison(
    y_true: pd.Series,
    model_probabilities: dict[str, np.ndarray],
) -> None:
    """Save the probability-calibration comparison."""

    plt.figure(figsize=(8, 6))

    for model_name, probabilities in (
        model_probabilities.items()
    ):
        observed_rate, predicted_rate = (
            calibration_curve(
                y_true,
                probabilities,
                n_bins=10,
                strategy="quantile",
            )
        )

        plt.plot(
            predicted_rate,
            observed_rate,
            marker="o",
            label=model_name,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed claim rate")
    plt.title(
        "Claim-Occurrence "
        "Calibration Comparison"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "calibration_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_capacity_comparison(
    capacity_analysis: pd.DataFrame,
) -> None:
    """Save the claim-capture comparison."""

    plt.figure(figsize=(8, 6))

    for model_name, group in (
        capacity_analysis.groupby("Model")
    ):
        ordered = group.sort_values(
            "Capacity"
        )

        plt.plot(
            ordered["Capacity"] * 100,
            ordered["ClaimCaptureRate"] * 100,
            marker="o",
            label=model_name,
        )

    plt.plot(
        [0, 50],
        [0, 50],
        linestyle="--",
        label="Random selection",
    )

    plt.xlabel("Portfolio reviewed (%)")
    plt.ylabel("Claims captured (%)")
    plt.title(
        "Claim Capture at Fixed Review Capacity"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "capacity_capture_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_importance_figure(
    importance_table: pd.DataFrame,
) -> None:
    """Save the permutation-importance figure."""

    plot_data = (
        importance_table.head(10)
        .sort_values(
            "ImportanceMean",
            ascending=True,
        )
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        plot_data["Feature"],
        plot_data["ImportanceMean"],
        xerr=plot_data[
            "ImportanceStandardDeviation"
        ],
    )

    plt.xlabel(
        "Decrease in test PR-AUC "
        "after permutation"
    )
    plt.ylabel("Feature")
    plt.title(
        "Histogram Gradient Boosting "
        "Permutation Importance"
    )
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "permutation_importance.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def create_documentation(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    capacity_analysis: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    importance_table: pd.DataFrame,
    selected_model: str,
    tree_iterations: int,
) -> None:
    """Generate model-comparison documentation."""

    selected_test = test_metrics.loc[
        test_metrics["Model"].eq(
            selected_model
        )
    ].iloc[0]

    ten_percent = capacity_analysis.loc[
        capacity_analysis["Model"].eq(
            selected_model
        )
        & capacity_analysis[
            "Capacity"
        ].eq(0.10)
    ].iloc[0]

    top_features = importance_table.head(
        5
    )["Feature"].tolist()

    subgroup_recall_minimum = (
        subgroup_metrics[
            "Recall"
        ].min()
        if not subgroup_metrics.empty
        else np.nan
    )

    subgroup_recall_maximum = (
        subgroup_metrics[
            "Recall"
        ].max()
        if not subgroup_metrics.empty
        else np.nan
    )

    validation_markdown = (
        validation_metrics[
            [
                "Model",
                "Threshold",
                "Precision",
                "Recall",
                "F2Score",
                "ROCAUC",
                "PRAUC",
                "BrierScore",
            ]
        ]
        .to_markdown(index=False)
    )

    test_markdown = (
        test_metrics[
            [
                "Model",
                "Threshold",
                "Precision",
                "Recall",
                "Specificity",
                "F2Score",
                "ROCAUC",
                "PRAUC",
                "BrierScore",
            ]
        ]
        .to_markdown(index=False)
    )

    documentation = f"""# Claim-Occurrence Model Comparison

## Objective

This analysis compares the Step 8 logistic-regression model with a
histogram gradient-boosting classifier.

The comparison uses the same training, validation, and test samples as
the original model.

## Models

### Logistic Regression

The logistic model provides a transparent statistical baseline with
additive effects on the log-odds scale.

### Histogram Gradient Boosting

The gradient-boosting model can represent nonlinear relationships and
interactions among driver, vehicle, geographic, and exposure variables.

The fitted model completed {tree_iterations:,} boosting iterations.

## Validation Results

{validation_markdown}

The preferred model was selected using validation PR-AUC rather than
test performance.

**Selected model:** `{selected_model}`

Each model's F2 classification threshold was also selected using the
validation sample.

## Test Results

{test_markdown}

For the selected model:

- Threshold: {selected_test["Threshold"]:.6f}
- Precision: {selected_test["Precision"]:.2%}
- Recall: {selected_test["Recall"]:.2%}
- Specificity: {selected_test["Specificity"]:.2%}
- F2-score: {selected_test["F2Score"]:.4f}
- ROC-AUC: {selected_test["ROCAUC"]:.4f}
- PR-AUC: {selected_test["PRAUC"]:.4f}
- Brier score: {selected_test["BrierScore"]:.5f}

## Review-Capacity Analysis

When the highest-risk 10% of test policies were selected:

- Policies selected: {int(ten_percent["SelectedPolicies"]):,}
- Claims identified: {int(ten_percent["SelectedClaims"]):,}
- Claim capture rate: {ten_percent["ClaimCaptureRate"]:.2%}
- Precision: {ten_percent["PrecisionAtCapacity"]:.2%}
- Lift over random selection: {ten_percent["Lift"]:.2f}

Capacity analysis may be more operationally useful than relying on one
fixed classification threshold.

## Permutation Importance

The five leading raw features were:

{chr(10).join(f"- `{feature}`" for feature in top_features)}

Permutation importance measures predictive reliance rather than
causation.

Correlated features may share importance, so a low value does not prove
that a variable has no relationship with claims.

## Subgroup Analysis

Subgroup evaluation was performed for:

- area;
- fuel type;
- driver-age band.

Groups with fewer than {MINIMUM_SUBGROUP_ROWS:,} policies or fewer than
{MINIMUM_SUBGROUP_CLAIMS:,} claims were excluded from comparative
metrics.

Across eligible groups, recall ranged from approximately
{subgroup_recall_minimum:.2%} to {subgroup_recall_maximum:.2%}.

Differences between groups should be investigated before the model is
used operationally. They may reflect sample size, claim prevalence,
data limitations, or differences in model fit.

## Limitations

- The data represents a historical French motor-insurance portfolio.
- Model performance does not establish causal relationships.
- The occurrence target is strongly imbalanced.
- Thresholds depend on the assumed relative importance of false
  positives and false negatives.
- Predicted probabilities may require additional calibration.
- Subgroup results do not by themselves establish fairness or unfairness.
- The model should support analysis rather than automate consequential
  insurance decisions.

## Recommended Model Use

The selected model may be used for:

- portfolio segmentation;
- risk-ranking research;
- identifying groups for further analysis;
- comparing expected and observed claim occurrence.

It should not be presented as a fraud model or used as the sole basis
for individual insurance decisions.
"""

    DOCUMENTATION_PATH.write_text(
        documentation,
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete Step 9 workflow."""

    create_directories()

    print("Loading occurrence-modelling data...")
    model_data = load_modelling_data()

    print("Recreating Step 8 data splits...")
    train, validation, test = (
        split_modelling_data(
            model_data
        )
    )

    if not LOGISTIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The Step 8 logistic model was not found. "
            "Run train_claim_occurrence.py first."
        )

    print("Loading converged logistic model...")
    logistic_model = joblib.load(
        LOGISTIC_MODEL_PATH
    )

    X_train = train[MODEL_FEATURES]
    y_train = train[TARGET]

    X_validation = validation[
        MODEL_FEATURES
    ]
    y_validation = validation[TARGET]

    X_test = test[MODEL_FEATURES]
    y_test = test[TARGET]

    print(
        "Generating logistic-regression "
        "validation probabilities..."
    )

    logistic_validation_probability = (
        logistic_model.predict_proba(
            X_validation
        )[:, 1]
    )

    print("Building histogram gradient boosting...")

    tree_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_iter=250,
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

    print(
        "Training histogram gradient boosting. "
        "This may take several minutes..."
    )

    tree_model.fit(
        X_train,
        y_train,
    )

    tree_validation_probability = (
        tree_model.predict_proba(
            X_validation
        )[:, 1]
    )

    logistic_threshold, _ = (
        select_f2_threshold(
            y_validation,
            logistic_validation_probability,
        )
    )

    tree_threshold, _ = (
        select_f2_threshold(
            y_validation,
            tree_validation_probability,
        )
    )

    validation_metrics = pd.DataFrame(
        [
            calculate_metrics(
                model_name=(
                    "Logistic regression"
                ),
                sample_name="Validation",
                y_true=y_validation,
                probabilities=(
                    logistic_validation_probability
                ),
                threshold=logistic_threshold,
            ),
            calculate_metrics(
                model_name=(
                    "Histogram gradient boosting"
                ),
                sample_name="Validation",
                y_true=y_validation,
                probabilities=(
                    tree_validation_probability
                ),
                threshold=tree_threshold,
            ),
        ]
    )

    validation_metrics.to_csv(
        REPORT_DIR
        / "validation_model_comparison.csv",
        index=False,
    )

    selected_model = (
        validation_metrics.sort_values(
            [
                "PRAUC",
                "ROCAUC",
            ],
            ascending=False,
        )
        .iloc[0]["Model"]
    )

    print(
        "Validation-selected model:",
        selected_model,
    )

    print(
        "Generating untouched test probabilities..."
    )

    logistic_test_probability = (
        logistic_model.predict_proba(
            X_test
        )[:, 1]
    )

    tree_test_probability = (
        tree_model.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = pd.DataFrame(
        [
            calculate_metrics(
                model_name=(
                    "Logistic regression"
                ),
                sample_name="Test",
                y_true=y_test,
                probabilities=(
                    logistic_test_probability
                ),
                threshold=logistic_threshold,
            ),
            calculate_metrics(
                model_name=(
                    "Histogram gradient boosting"
                ),
                sample_name="Test",
                y_true=y_test,
                probabilities=(
                    tree_test_probability
                ),
                threshold=tree_threshold,
            ),
        ]
    )

    test_metrics.to_csv(
        REPORT_DIR
        / "test_model_comparison.csv",
        index=False,
    )

    capacity_analysis = pd.concat(
        [
            create_capacity_analysis(
                model_name=(
                    "Logistic regression"
                ),
                y_true=y_test,
                probabilities=(
                    logistic_test_probability
                ),
            ),
            create_capacity_analysis(
                model_name=(
                    "Histogram gradient boosting"
                ),
                y_true=y_test,
                probabilities=(
                    tree_test_probability
                ),
            ),
        ],
        ignore_index=True,
    )

    capacity_analysis.to_csv(
        REPORT_DIR
        / "capacity_analysis.csv",
        index=False,
    )

    if selected_model == (
        "Histogram gradient boosting"
    ):
        selected_probability = (
            tree_test_probability
        )
        selected_threshold = (
            tree_threshold
        )
    else:
        selected_probability = (
            logistic_test_probability
        )
        selected_threshold = (
            logistic_threshold
        )

    subgroup_metrics = (
        create_subgroup_metrics(
            test_data=test,
            probabilities=selected_probability,
            threshold=selected_threshold,
            model_name=selected_model,
        )
    )

    subgroup_metrics.to_csv(
        REPORT_DIR
        / "subgroup_metrics.csv",
        index=False,
    )

    print(
        "Calculating permutation importance..."
    )

    importance_table = (
        create_permutation_importance(
            model=tree_model,
            test_data=test,
        )
    )

    importance_table.to_csv(
        REPORT_DIR
        / "permutation_importance.csv",
        index=False,
    )

    selected_model_summary = (
        test_metrics.loc[
            test_metrics["Model"].eq(
                selected_model
            )
        ].copy()
    )

    selected_model_summary.to_csv(
        REPORT_DIR
        / "selected_model_summary.csv",
        index=False,
    )

    predictions = test[
        [
            "IDpol",
            "Exposure",
            TARGET,
        ]
    ].copy()

    predictions[
        "LogisticProbability"
    ] = logistic_test_probability

    predictions[
        "GradientBoostingProbability"
    ] = tree_test_probability

    predictions[
        "SelectedModel"
    ] = selected_model

    predictions[
        "SelectedModelProbability"
    ] = selected_probability

    predictions[
        "SelectedModelPrediction"
    ] = (
        selected_probability
        >= selected_threshold
    ).astype("int8")

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("Saving histogram gradient-boosting model...")

    joblib.dump(
        tree_model,
        TREE_MODEL_PATH,
    )

    model_probabilities = {
        "Logistic regression": (
            logistic_test_probability
        ),
        "Histogram gradient boosting": (
            tree_test_probability
        ),
    }

    print("Creating comparison figures...")

    save_roc_comparison(
        y_true=y_test,
        model_probabilities=(
            model_probabilities
        ),
    )

    save_pr_comparison(
        y_true=y_test,
        model_probabilities=(
            model_probabilities
        ),
    )

    save_calibration_comparison(
        y_true=y_test,
        model_probabilities=(
            model_probabilities
        ),
    )

    save_capacity_comparison(
        capacity_analysis
    )

    save_importance_figure(
        importance_table
    )

    create_documentation(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        capacity_analysis=capacity_analysis,
        subgroup_metrics=subgroup_metrics,
        importance_table=importance_table,
        selected_model=selected_model,
        tree_iterations=int(
            tree_model.n_iter_
        ),
    )

    print(
        "\nStep 9 completed successfully."
    )

    print(
        "Training rows:",
        f"{len(train):,}",
    )

    print(
        "Validation rows:",
        f"{len(validation):,}",
    )

    print(
        "Test rows:",
        f"{len(test):,}",
    )

    print(
        "Gradient-boosting iterations:",
        tree_model.n_iter_,
    )

    print(
        "Logistic validation threshold:",
        f"{logistic_threshold:.6f}",
    )

    print(
        "Gradient-boosting validation threshold:",
        f"{tree_threshold:.6f}",
    )

    print(
        "Validation-selected model:",
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

    print("\nTop permutation features:")
    print(
        importance_table.head(
            10
        ).to_string(
            index=False
        )
    )

    print(
        "\nReports saved to:",
        REPORT_DIR,
    )


if __name__ == "__main__":
    main()