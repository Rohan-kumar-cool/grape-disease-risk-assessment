
from pathlib import Path
import numpy as np
import pandas as pd

original_dir = Path("environmental_data/source_original")
synthetic_dir = Path("environmental_data/source_synthetic")

weather_cols = [
    "air_hum", "air_hum_min", "air_hum_max",
    "air_temp", "air_temp_min", "air_temp_max", "pluvm"
]

year_map = {0: 2020, 1: 2021, 2: 2022, 3: 2023, 4: 2024}

def read_tsv(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

original = pd.concat(
    [read_tsv(p) for p in sorted(original_dir.glob("*.csv"))],
    ignore_index=True
).drop_duplicates(subset=["date"]).set_index("date")

results = []

for i, year in year_map.items():
    path = synthetic_dir / (
        f"Temporal enviromental factors_Vine_noised_{i}_form.csv"
    )
    if not path.exists():
        continue

    noisy = read_tsv(path).set_index("date")
    baseline = original[original.index.year == year]
    common = noisy.index.intersection(baseline.index)

    for col in weather_cols:
        if col not in noisy.columns or col not in baseline.columns:
            continue

        a = pd.to_numeric(baseline.loc[common, col], errors="coerce")
        b = pd.to_numeric(noisy.loc[common, col], errors="coerce")
        valid = a.notna() & b.notna()

        if not valid.any():
            continue

        difference = (b[valid] - a[valid]).abs()
        value_range = a[valid].max() - a[valid].min()

        results.append({
            "year": year,
            "feature": col,
            "mean_abs_change": difference.mean(),
            "max_abs_change": difference.max(),
            "original_range": value_range,
            "mean_change_pct_of_range": (
                100 * difference.mean() / value_range
                if value_range > 0 else np.nan
            )
        })

summary = pd.DataFrame(results)

print("Mean perturbation as percentage of original yearly range:")
print(
    summary.groupby("feature")["mean_change_pct_of_range"]
    .mean()
    .round(4)
    .to_string()
)

print("\nFull summary:")
print(summary.round(4).to_string(index=False))

output = Path("outputs/metrics")
output.mkdir(parents=True, exist_ok=True)
summary.to_csv(output / "noise_scale_summary.csv", index=False)

print("\nSaved: outputs/metrics/noise_scale_summary.csv")
