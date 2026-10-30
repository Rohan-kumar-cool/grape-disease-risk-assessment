
from pathlib import Path
import numpy as np
import pandas as pd

original_dir = Path("environmental_data/source_original")
synthetic_dir = Path("environmental_data/source_synthetic")

weather_cols = [
    "air_hum", "air_hum_min", "air_hum_max",
    "air_temp", "air_temp_min", "air_temp_max", "pluvm"
]

year_map = {
    0: 2020,
    1: 2021,
    2: 2022,
    3: 2023,
    4: 2024,
}

def read_tsv(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

original = pd.concat(
    [read_tsv(p) for p in sorted(original_dir.glob("*.csv"))],
    ignore_index=True
).drop_duplicates(subset=["date"]).set_index("date")

for index, year in year_map.items():
    path = synthetic_dir / (
        f"Temporal enviromental factors_Vine_noised_{index}_form.csv"
    )

    if not path.exists():
        print(f"\nMissing file: {path.name}")
        continue

    noisy = read_tsv(path).set_index("date")
    baseline = original[original.index.year == year]

    common = noisy.index.intersection(baseline.index)
    noisy = noisy.loc[common]
    baseline = baseline.loc[common]

    print(f"\n{'=' * 60}")
    print(f"Year: {year} | Matched dates: {len(common)}")

    for col in weather_cols:
        if col not in noisy.columns or col not in baseline.columns:
            continue

        a = pd.to_numeric(baseline[col], errors="coerce")
        b = pd.to_numeric(noisy[col], errors="coerce")

        valid = a.notna() & b.notna()
        if valid.sum() == 0:
            continue

        difference = b[valid] - a[valid]

        print(
            f"{col}: "
            f"mean absolute change={difference.abs().mean():.4f}, "
            f"maximum absolute change={difference.abs().max():.4f}, "
            f"correlation={a[valid].corr(b[valid]):.4f}"
        )

print("\nWeather-noise comparison complete.")
