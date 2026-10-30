
from pathlib import Path
import pandas as pd

original_dir = Path("environmental_data/source_original")
synthetic_dir = Path("environmental_data/source_synthetic")

label_cols = ["down_mildew", "powdery_mildew"]

def read_tsv(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

original_files = sorted(original_dir.glob("*.csv"))
original = pd.concat(
    [read_tsv(path) for path in original_files],
    ignore_index=True
)

original = original.drop_duplicates(subset=["date"])
original_by_date = original.set_index("date")

for path in sorted(synthetic_dir.glob("*noised*form.csv")):
    synthetic = read_tsv(path)

    if "synth" in path.name.lower():
        continue

    year_number = path.name.split("_noised_")[1].split("_")[0]
    year_map = {
        "0": 2020,
        "1": 2021,
        "2": 2022,
        "3": 2023,
        "4": 2024,
    }
    year = year_map.get(year_number)

    if year is None:
        continue

    original_year = original[
        original["date"].dt.year == year
    ].set_index("date")

    synthetic = synthetic.set_index("date")

    print("\nFile:", path.name)
    print("Year:", year)

    common_dates = synthetic.index.intersection(original_year.index)
    print("Matching dates:", len(common_dates))

    for col in label_cols:
        if col not in synthetic.columns or col not in original_year.columns:
            continue

        a = synthetic.loc[common_dates, col]
        b = original_year.loc[common_dates, col]

        same = a.astype(str).to_numpy() == b.astype(str).to_numpy()

        print(f"{col}: {same.sum()}/{len(same)} labels match")

print("\nYearly noise comparison complete.")
