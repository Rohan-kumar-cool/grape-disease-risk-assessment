
from pathlib import Path
import pandas as pd

FOLDERS = [
    Path("environmental_data/source_original"),
    Path("environmental_data/source_synthetic"),
]

LABELS = ["down_mildew", "powdery_mildew"]
WEATHER = [
    "air_hum",
    "air_hum_min",
    "air_hum_max",
    "air_temp",
    "air_temp_min",
    "air_temp_max",
    "pluvm",
]

def read_dataset(path):
    # The supplied source files are tab-separated despite their .csv extension.
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    return df

for folder in FOLDERS:
    print("\n" + "=" * 70)
    print("FOLDER:", folder)
    print("=" * 70)

    files = sorted(folder.glob("*.csv"))

    if not files:
        print("No CSV files found. Check the folder path and copied files.")
        continue

    for path in files:
        try:
            df = read_dataset(path)
            print(f"\nFILE: {path.name}")
            print("Rows:", len(df))
            print("Columns:", df.columns.tolist())

            if "date" in df.columns:
                dates = pd.to_datetime(df["date"], errors="coerce")
                print("Invalid dates:", int(dates.isna().sum()))
                print("Unique dates:", int(dates.nunique()))
                print("Duplicate dates:", int(dates.duplicated().sum()))
                print("Date range:", dates.min(), "to", dates.max())

            for col in LABELS:
                if col in df.columns:
                    print(f"{col} counts:")
                    print(df[col].value_counts(dropna=False).sort_index().to_dict())

            present_weather = [c for c in WEATHER if c in df.columns]
            if present_weather:
                print("Weather summary:")
                print(df[present_weather].describe().round(3).to_string())

            missing = df.isna().sum()
            print("Columns with missing values:")
            print(missing[missing > 0].to_dict())

        except Exception as exc:
            print("Could not read this file:", repr(exc))

print("\nSource dataset comparison complete.")
