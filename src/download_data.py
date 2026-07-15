"""Download the freMTPL2 insurance datasets from OpenML.

The script downloads:

1. freMTPL2freq: policy characteristics and claim counts.
2. freMTPL2sev: policy identifiers and individual claim amounts.

The original datasets are saved without analytical transformations in
data/raw.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FREQUENCY_DATA_ID = 41214
SEVERITY_DATA_ID = 41215

EXPECTED_FREQUENCY_COLUMNS = {
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
}

EXPECTED_SEVERITY_COLUMNS = {
    "IDpol",
    "ClaimAmount",
}


def download_openml_dataset(
    data_id: int,
    dataset_name: str,
    expected_columns: set[str],
) -> pd.DataFrame:
    """Download and validate a dataset from OpenML."""

    print(f"Downloading {dataset_name}...")

    dataset = fetch_openml(
        data_id=data_id,
        as_frame=True,
    )

    dataframe = dataset.data.copy()

    if dataframe.empty:
        raise ValueError(f"{dataset_name} was downloaded but contains no rows.")

    missing_columns = expected_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing expected columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe["IDpol"] = pd.to_numeric(
        dataframe["IDpol"],
        errors="raise",
    ).astype("int64")

    return dataframe


def save_dataset(dataframe: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame in the raw-data directory."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DATA_DIR / filename
    dataframe.to_csv(output_path, index=False)

    return output_path


def main() -> None:
    """Download, validate and save both freMTPL2 datasets."""

    frequency_data = download_openml_dataset(
        data_id=FREQUENCY_DATA_ID,
        dataset_name="freMTPL2freq",
        expected_columns=EXPECTED_FREQUENCY_COLUMNS,
    )

    severity_data = download_openml_dataset(
        data_id=SEVERITY_DATA_ID,
        dataset_name="freMTPL2sev",
        expected_columns=EXPECTED_SEVERITY_COLUMNS,
    )

    frequency_path = save_dataset(
        frequency_data,
        "freMTPL2freq.csv",
    )

    severity_path = save_dataset(
        severity_data,
        "freMTPL2sev.csv",
    )

    print("\nDownload completed successfully.")
    print(f"Frequency data shape: {frequency_data.shape}")
    print(f"Severity data shape:  {severity_data.shape}")
    print(f"Unique policies:      {frequency_data['IDpol'].nunique():,}")
    print(f"Claim records:        {len(severity_data):,}")
    print(f"\nSaved frequency data to:\n{frequency_path}")
    print(f"\nSaved severity data to:\n{severity_path}")


if __name__ == "__main__":
    main()