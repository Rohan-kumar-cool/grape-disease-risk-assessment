
from pathlib import Path
import pandas as pd

DATA_PATH = Path("environmental_data/environmental_cleaned.csv")
OUTPUT_DIR = Path("outputs/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("File:", DATA_PATH.resolve())
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head().to_string(index=False))

print("\n========== DATA TYPES ==========")
print(df.dtypes.to_string())

print("\n========== MISSING VALUES ==========")
print(df.isna().sum().to_string())

print("\n========== DUPLICATE ROWS ==========")
print("Exact duplicate rows:", df.duplicated().sum())

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print("\n========== DATE CHECK ==========")
    print("Invalid dates:", df["date"].isna().sum())
    print("Earliest date:", df["date"].min())
    print("Latest date:", df["date"].max())
    print("Duplicate dates:", df["date"].duplicated().sum())

    dates = df["date"].dropna().sort_values()
    gaps = dates.diff().dt.days
    print("Date gaps greater than 1 day:", int((gaps > 1).sum()))

print("\n========== DISEASE LABEL COUNTS ==========")
for col in ["down_mildew", "powdery_mildew"]:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).sort_index().to_string())

if {"down_mildew", "powdery_mildew"}.issubset(df.columns):
    print("\n========== LABEL OVERLAP ==========")
    print(
        pd.crosstab(
            df["down_mildew"],
            df["powdery_mildew"],
            margins=True
        ).to_string()
    )

print("\n========== NUMERIC FEATURE SUMMARY ==========")
print(df.describe(include="number").T.to_string())

report_path = OUTPUT_DIR / "environmental_audit.txt"
with report_path.open("w", encoding="utf-8") as file:
    file.write(f"Dataset: {DATA_PATH.resolve()}\n\n")
    file.write(f"Shape: {df.shape}\n\n")
    file.write("Columns:\n")
    file.write("\n".join(df.columns.astype(str)))
    file.write("\n\nMissing values:\n")
    file.write(df.isna().sum().to_string())
    file.write("\n\nNumeric summary:\n")
    file.write(df.describe(include="number").T.to_string())

print(f"\nAudit summary saved to: {report_path.resolve()}")
print("\nAudit complete.")
