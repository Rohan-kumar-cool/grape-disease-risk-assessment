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
    df.dropna(subset=["date", *disease_cols])
    .sort_values("date")
    .drop_duplicates(subset=["date"])
    .reset_index(drop=True)
)

print("LABEL AGREEMENT")
print(pd.crosstab(
    df["down_mildew"],
    df["powdery_mildew"],
    rownames=["Downy Mildew"],
    colnames=["Powdery Mildew"]
))

agreement = (
    df["down_mildew"] == df["powdery_mildew"]
).mean()

print(f"\nOverall label agreement: {agreement * 100:.2f}%")

down_change = df["down_mildew"].ne(df["down_mildew"].shift(1))
powdery_change = df["powdery_mildew"].ne(
    df["powdery_mildew"].shift(1)
)

valid = df.index > 0

print("\nTRANSITION OVERLAP")
print("Downy changes:", int(down_change.loc[valid].sum()))
print("Powdery changes:", int(powdery_change.loc[valid].sum()))
print(
    "Both change on the same transition:",
    int((down_change & powdery_change).loc[valid].sum())
)
print(
    "Only Downy changes:",
    int((down_change & ~powdery_change).loc[valid].sum())
)
print(
    "Only Powdery changes:",
    int((~down_change & powdery_change).loc[valid].sum())
)

print("\nDATES WHERE THE LABELS DISAGREE")
print(
    df.loc[
        df["down_mildew"] != df["powdery_mildew"],
        ["date", "down_mildew", "powdery_mildew"]
    ].to_string(index=False)
)
