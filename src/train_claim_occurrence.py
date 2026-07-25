"""Train and evaluate the claim-occurrence logistic regression model.

The model predicts whether an insurance policy reports at least one
claim during its observed exposure period.

The workflow includes:

1. Leakage-safe feature selection.
2. Stratified training, validation, and test splits.
3. Numerical and categorical preprocessing.
4. A no-claim baseline.
5. Logistic regression.
6. Validation-based threshold selection.
7. Test-set evaluation.
8. Probability calibration and coefficient analysis.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "policy_analytics.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "claim_occurrence_test_predictions.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
    / "claim_occurrence_logistic.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "claim_occurrence"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "modeling"
    / "claim_occurrence"
)

DOCS_DIR = PROJECT_ROOT / "docs"

DOCUMENTATION_PATH = (
    DOCS_DIR
    / "claim_occurrence_model.md"
)

RANDOM_STATE = 42

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

TARGET = "HasClaim"


def convert_to_boolean(series: pd.Series) -> pd.Series:
    """Convert common Boolean representations to True and False."""

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
        MODEL_DIR,
        REPORT_DIR,
        FIGURES_DIR,
        DOCS_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def load_modelling_data() -> pd.DataFrame:
    """Load and prepare the claim-occurrence modelling sample."""

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            "Policy analytics file not found. Run "
            "`python src/prepare_data.py` first."
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

    if not model_data[TARGET].isin([0, 1]).all():
        raise ValueError(
            "HasClaim contains values other than zero and one."
        )

    if model_data["Exposure"].le(0).any():
        raise ValueError(
            "Exposure contains non-positive values."
        )

    model_data["LogExposure"] = np.log(
        model_data["Exposure"]
    )

    model_data["LogDensity"] = np.log1p(
        model_data["Density"]
    )

    return model_data


def split_modelling_data(
    model_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create stratified 60/20/20 data splits."""

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


def create_split_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize sample sizes and class prevalence."""

    rows = []

    for name, dataframe in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        positive_count = int(
            dataframe[TARGET].sum()
        )

        rows.append(
            {
                "Sample": name,
                "Rows": len(dataframe),
                "Claims": positive_count,
                "NoClaims": (
                    len(dataframe)
                    - positive_count
                ),
                "ClaimRate": (
                    positive_count
                    / len(dataframe)
                ),
            }
        )

    return pd.DataFrame(rows)


def build_model_pipeline() -> Pipeline:
    """Create the preprocessing and logistic-regression pipeline."""

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

    classifier = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def select_f2_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """Select the validation threshold that maximizes F2-score."""

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

    best_threshold = float(
        thresholds[best_index]
    )

    threshold_results = pd.DataFrame(
        {
            "Threshold": thresholds,
            "Precision": threshold_precision,
            "Recall": threshold_recall,
            "F2Score": f2_scores,
        }
    )

    return best_threshold, threshold_results


def calculate_metrics(
    model_name: str,
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

    negative_predictive_value = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    return {
        "Model": model_name,
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
        "NegativePredictiveValue": (
            negative_predictive_value
        ),
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
        "PredictedPositiveRate": (
            predictions.mean()
        ),
    }


def create_threshold_grid_report(
    y_true: pd.Series,
    probabilities: np.ndarray,
    selected_threshold: float,
) -> pd.DataFrame:
    """Evaluate a compact collection of possible thresholds."""

    threshold_grid = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.01,
                    0.25,
                    49,
                ),
                np.array(
                    [
                        selected_threshold,
                        0.50,
                    ]
                ),
            ]
        )
    )

    rows = []

    for threshold in threshold_grid:
        predictions = (
            probabilities >= threshold
        ).astype("int8")

        rows.append(
            {
                "Threshold": threshold,
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
                "PredictedPositiveRate": (
                    predictions.mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def save_roc_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> None:
    """Save the test-set ROC curve."""

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_true,
            probabilities,
        )
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Logistic regression: AUC = {roc_auc:.3f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random ranking",
    )

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Claim-Occurrence ROC Curve")
    plt.legend()
    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "claim_occurrence_roc_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_precision_recall_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> None:
    """Save the test-set precision-recall curve."""

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

    prevalence = float(
        y_true.mean()
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        recall,
        precision,
        label=f"Logistic regression: PR-AUC = {pr_auc:.3f}",
    )

    plt.axhline(
        prevalence,
        linestyle="--",
        label=f"Claim-rate baseline = {prevalence:.3f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Claim-Occurrence Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "claim_occurrence_precision_recall_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_confusion_matrix(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    """Save the confusion matrix at the selected threshold."""

    predictions = (
        probabilities >= threshold
    ).astype("int8")

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "No claim",
            "Claim",
        ],
    )

    display.plot(
        values_format="d",
    )

    plt.title(
        "Claim-Occurrence Confusion Matrix\n"
        f"Threshold = {threshold:.4f}"
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "claim_occurrence_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_calibration_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> None:
    """Save the probability calibration curve."""

    observed_rate, predicted_probability = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        predicted_probability,
        observed_rate,
        marker="o",
        label="Logistic regression",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed claim rate")
    plt.title("Claim-Occurrence Probability Calibration")
    plt.legend()
    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "claim_occurrence_calibration.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def extract_coefficients(
    model: Pipeline,
) -> pd.DataFrame:
    """Extract logistic-regression coefficients."""

    feature_names = (
        model.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )

    coefficients = (
        model.named_steps[
            "classifier"
        ].coef_[0]
    )

    coefficient_table = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
        }
    )

    coefficient_table["OddsRatio"] = np.exp(
        np.clip(
            coefficient_table["Coefficient"],
            -20,
            20,
        )
    )

    coefficient_table["AbsoluteCoefficient"] = (
        coefficient_table[
            "Coefficient"
        ].abs()
    )

    return coefficient_table.sort_values(
        "AbsoluteCoefficient",
        ascending=False,
    )


def markdown_coefficient_table(
    dataframe: pd.DataFrame,
    number_of_rows: int = 10,
) -> str:
    """Convert selected coefficient rows to Markdown."""

    lines = [
        "| Feature | Coefficient | Odds ratio |",
        "|---|---:|---:|",
    ]

    for row in dataframe.head(
        number_of_rows
    ).itertuples(index=False):
        lines.append(
            f"| {row.Feature} "
            f"| {row.Coefficient:.4f} "
            f"| {row.OddsRatio:.4f} |"
        )

    return "\n".join(lines)


def create_documentation(
    split_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_threshold: float,
    coefficients: pd.DataFrame,
) -> None:
    """Generate the claim-occurrence model report."""

    selected = metrics.loc[
        metrics["Model"].eq(
            "Logistic regression — selected threshold"
        )
    ].iloc[0]

    default = metrics.loc[
        metrics["Model"].eq(
            "Logistic regression — threshold 0.50"
        )
    ].iloc[0]

    strongest_positive = (
        coefficients.loc[
            coefficients["Coefficient"] > 0
        ]
        .sort_values(
            "Coefficient",
            ascending=False,
        )
    )

    strongest_negative = (
        coefficients.loc[
            coefficients["Coefficient"] < 0
        ]
        .sort_values(
            "Coefficient",
            ascending=True,
        )
    )

    positive_table = markdown_coefficient_table(
        strongest_positive
    )

    negative_table = markdown_coefficient_table(
        strongest_negative
    )

    split_lines = [
        "| Sample | Rows | Claims | Claim rate |",
        "|---|---:|---:|---:|",
    ]

    for row in split_summary.itertuples(
        index=False
    ):
        split_lines.append(
            f"| {row.Sample} "
            f"| {row.Rows:,} "
            f"| {row.Claims:,} "
            f"| {row.ClaimRate:.2%} |"
        )

    split_table = "\n".join(
        split_lines
    )

    documentation = f"""# Claim-Occurrence Model

## Objective

The model estimates the probability that a policy reports at least one
claim during its observed exposure period.

The target is `HasClaim`.

## Modelling Sample

{split_table}

The samples were created using stratified random splitting so that the
claim rate remained similar across training, validation, and test data.

## Predictors

The model uses:

- logarithm of exposure;
- vehicle power;
- vehicle age;
- driver age;
- bonus-malus score;
- logarithm of population density;
- area;
- vehicle brand;
- fuel type;
- region.

Policy identifiers, claim counts, claim amounts, severity fields, and
other post-claim variables were excluded to prevent data leakage.

## Model

The first statistical model is logistic regression with L2
regularization.

Numerical variables are median-imputed and standardized. Categorical
variables are most-frequent-imputed and one-hot encoded.

## Class Imbalance

Only a small percentage of policies have claims. A model predicting no
claim for every policy can therefore obtain high accuracy while having
zero claim recall.

Accuracy is not used as the main selection criterion.

## Threshold Selection

The classification threshold was selected using only the validation
sample.

The selected threshold was:

`{selected_threshold:.6f}`

It maximized the F2-score, which gives recall greater importance than
precision.

This reflects a screening objective where failing to identify a policy
that has a claim may be more costly than reviewing an additional
policy.

## Test Performance

At the default threshold of 0.50:

- Precision: {default["Precision"]:.3f}
- Recall: {default["Recall"]:.3f}
- F2-score: {default["F2Score"]:.3f}

At the validation-selected threshold:

- Precision: {selected["Precision"]:.3f}
- Recall: {selected["Recall"]:.3f}
- Specificity: {selected["Specificity"]:.3f}
- F1-score: {selected["F1Score"]:.3f}
- F2-score: {selected["F2Score"]:.3f}
- ROC-AUC: {selected["ROCAUC"]:.3f}
- PR-AUC: {selected["PRAUC"]:.3f}
- Brier score: {selected["BrierScore"]:.5f}
- False positives: {int(selected["FalsePositives"]):,}
- False negatives: {int(selected["FalseNegatives"]):,}

## Strongest Positive Coefficients

{positive_table}

## Strongest Negative Coefficients

{negative_table}

Positive coefficients are associated with higher estimated claim odds,
while negative coefficients are associated with lower estimated claim
odds, holding the other model variables constant.

Numerical variables were standardized, so their coefficients represent
a one-standard-deviation change. Categorical coefficients are relative
to the category omitted by one-hot encoding.

Coefficients describe statistical associations and should not be
interpreted as causal effects.

## Limitations

- The data represents a historical French insurance portfolio.
- The target is strongly imbalanced.
- The model predicts historical claim occurrence, not fraud.
- Exposure is known for the historical observation period.
- Logistic regression assumes an additive linear relationship on the
  log-odds scale.
- Interactions and nonlinear effects are not yet included.
- A lower classification threshold increases recall but also increases
  false positives.
- The model should support human analysis rather than automate
  consequential insurance decisions.

## Next Modelling Actions

Future work will:

1. compare logistic regression with tree-based models;
2. investigate nonlinear and interaction effects;
3. examine subgroup performance;
4. assess probability calibration more deeply;
5. build a claim-frequency model that explicitly accounts for exposure.
"""

    DOCUMENTATION_PATH.write_text(
        documentation,
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete claim-occurrence modelling workflow."""

    create_directories()

    print("Loading modelling data...")
    model_data = load_modelling_data()

    print("Creating stratified data splits...")
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
        REPORT_DIR / "split_summary.csv",
        index=False,
    )

    X_train = train[MODEL_FEATURES]
    y_train = train[TARGET]

    X_validation = validation[
        MODEL_FEATURES
    ]
    y_validation = validation[TARGET]

    X_test = test[MODEL_FEATURES]
    y_test = test[TARGET]

    print("Building logistic-regression pipeline...")
    model = build_model_pipeline()

    print(
        "Training logistic regression. "
        "This may take several minutes..."
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    selected_threshold, validation_thresholds = (
        select_f2_threshold(
            y_validation,
            validation_probabilities,
        )
    )

    validation_thresholds.to_csv(
        REPORT_DIR
        / "validation_threshold_results.csv",
        index=False,
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    baseline_probability = float(
        y_train.mean()
    )

    baseline_probabilities = np.full(
        len(y_test),
        baseline_probability,
    )

    metrics = pd.DataFrame(
        [
            calculate_metrics(
                model_name="No-claim baseline",
                y_true=y_test,
                probabilities=baseline_probabilities,
                threshold=0.50,
            ),
            calculate_metrics(
                model_name=(
                    "Logistic regression — threshold 0.50"
                ),
                y_true=y_test,
                probabilities=test_probabilities,
                threshold=0.50,
            ),
            calculate_metrics(
                model_name=(
                    "Logistic regression — selected threshold"
                ),
                y_true=y_test,
                probabilities=test_probabilities,
                threshold=selected_threshold,
            ),
        ]
    )

    metrics.to_csv(
        REPORT_DIR
        / "claim_occurrence_metrics.csv",
        index=False,
    )

    threshold_grid = (
        create_threshold_grid_report(
            y_true=y_test,
            probabilities=test_probabilities,
            selected_threshold=selected_threshold,
        )
    )

    threshold_grid.to_csv(
        REPORT_DIR
        / "test_threshold_comparison.csv",
        index=False,
    )

    coefficients = extract_coefficients(
        model
    )

    coefficients.to_csv(
        REPORT_DIR
        / "logistic_coefficients.csv",
        index=False,
    )

    test_predictions = test[
        [
            "IDpol",
            "Exposure",
            TARGET,
        ]
    ].copy()

    test_predictions[
        "PredictedClaimProbability"
    ] = test_probabilities

    test_predictions[
        "PredictedClaimDefaultThreshold"
    ] = (
        test_probabilities >= 0.50
    ).astype("int8")

    test_predictions[
        "PredictedClaimSelectedThreshold"
    ] = (
        test_probabilities
        >= selected_threshold
    ).astype("int8")

    test_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    prediction_sample = (
        test_predictions.sample(
            n=min(
                2_000,
                len(test_predictions),
            ),
            random_state=RANDOM_STATE,
        )
        .sort_values(
            "PredictedClaimProbability",
            ascending=False,
        )
    )

    prediction_sample.to_csv(
        REPORT_DIR
        / "prediction_sample.csv",
        index=False,
    )

    print("Saving model...")
    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("Creating model figures...")
    save_roc_curve(
        y_test,
        test_probabilities,
    )

    save_precision_recall_curve(
        y_test,
        test_probabilities,
    )

    save_confusion_matrix(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    save_calibration_curve(
        y_test,
        test_probabilities,
    )

    create_documentation(
        split_summary=split_summary,
        metrics=metrics,
        selected_threshold=selected_threshold,
        coefficients=coefficients,
    )

    classifier = model.named_steps[
        "classifier"
    ]

    print(
        "\nClaim-occurrence model completed successfully."
    )

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    print(
        "Selected F2 threshold:",
        f"{selected_threshold:.6f}",
    )

    print(
        "Logistic-regression iterations:",
        classifier.n_iter_.tolist(),
    )

    print("\nTest metrics:")
    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\nModel saved to:",
        MODEL_PATH,
    )

    print(
        "Reports saved to:",
        REPORT_DIR,
    )


if __name__ == "__main__":
    main()