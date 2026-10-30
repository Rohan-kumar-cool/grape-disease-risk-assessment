from pathlib import Path
import pandas as pd

DATA_PATH = Path("environmental_data/environmental_cleaned.csv")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"], errors="coerce")

disease_cols = ["down_mildew", "powdery_mildew"]

for col in disease_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = (
    df.dropna(subset=["date"])
    .sort_values("date")
    .drop_duplicates(subset=["date"])
    .reset_index(drop=True)
)

for disease in disease_cols:
    previous = df[disease].shift(1)
    valid = df[disease].notna() & previous.notna()

    current = df.loc[valid, disease].astype(int)
    previous = previous.loc[valid].astype(int)

    changes = current.ne(previous)

    print(f"\n{'=' * 45}")
    print(f"Disease indicator: {disease}")
    print("Evaluated consecutive observations:", len(current))
    print("Label changes:", int(changes.sum()))
    print("Label changes (%):", round(changes.mean() * 100, 2))

    print("\nTransition counts (previous label → current label):")
    print(pd.crosstab(
        previous,
        current,
        rownames=["Previous"],
        colnames=["Current"],
        dropna=False
    ))

    transition_dates = df.loc[valid].loc[
        changes.to_numpy(), ["date", disease]
    ]

    print("\nFirst 15 label-change dates:")
    print(transition_dates.head(15).to_string(index=False))
