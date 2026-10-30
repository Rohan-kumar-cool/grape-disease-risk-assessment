
from pathlib import Path
import pandas as pd

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_FILE = (
    BASE_DIR / "environmental_data" / "environmental_cleaned.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "environmental_data" / "anfis"
    / "environmental_anfis_600.csv"
)

SAMPLE_SIZE = 600
RANDOM_SEED = 42


def main():
    # Load original tab-separated data without modifying it.
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_FILE}"
        )

    df = pd.read_csv(SOURCE_FILE, sep=None, engine="python")

    required = [
        "date",
        "air_temp",
        "air_hum",
        "pluvm",
        "down_mildew",
        "powdery_mildew",
    ]

    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select relevant columns into a separate DataFrame.
    data = df[required].copy()

    # Convert measurement and label columns to numeric.
    numeric_columns = [
        "air_temp",
        "air_hum",
        "pluvm",
        "down_mildew",
        "powdery_mildew",
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    # Validate disease-associated labels.
    for col in ["down_mildew", "powdery_mildew"]:
        if data[col].isna().any():
            raise ValueError(
                f"{col} contains missing or non-numeric values."
            )

        if not data[col].isin([0, 1]).all():
            raise ValueError(
                f"{col} must contain only 0 and 1."
            )

    # Parse dates and identify missing values.
    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    if data.isna().any().any():
        print("Missing values found:")
        print(data.isna().sum())
        raise ValueError(
            "Resolve missing data explicitly before sampling. "
            "No records have been silently imputed."
        )

    if len(data) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(data)} records available; "
            f"{SAMPLE_SIZE} required."
        )

    # Combined label: positive if either original label is positive.
    data["combined_target"] = data[
        ["down_mildew", "powdery_mildew"]
    ].max(axis=1).astype(int)

    # Reproducible sample of genuine records; no replacement.
    sampled = data.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED,
        replace=False,
    )

    sampled = sampled.sort_values("date").reset_index(drop=True)

    # Provenance: these are sampled original observations,
    # not newly generated or synthetic weather records.
    sampled["record_type"] = "sampled_original_observation"
    sampled["target_definition"] = (
        "max(down_mildew, powdery_mildew)"
    )

    # Save to a new file, preserving the source.
    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    sampled.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset preparation successful.")
    print("Original row count:", len(df))
    print("Prepared row count:", len(sampled))
    print("Output file:", OUTPUT_FILE)

    print("\nCombined target counts:")
    print(sampled["combined_target"].value_counts().sort_index())

    print("\nOriginal label combinations:")
    print(
        pd.crosstab(
            sampled["down_mildew"],
            sampled["powdery_mildew"],
        )
    )

    print("\nFirst five records:")
    print(sampled.head().to_string(index=False))


if __name__ == "__main__":
    main()
