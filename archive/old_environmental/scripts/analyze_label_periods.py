from pathlib import Path
import pandas as pd

DATA_PATH = Path("environmental_data/environmental_cleaned.csv")
OUTPUT_PATH = Path("outputs/metrics/label_periods.csv")

df = pd.read_csv(DATA_PATH, sep=None, engine="python")
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"], errors="coerce")

diseases = ["down_mildew", "powdery_mildew"]

for col in diseases:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = (
    df.dropna(subset=["date"])
    .sort_values("date")
    .drop_duplicates(subset=["date"])
    .reset_index(drop=True)
)

periods = []

for disease in diseases:
    positive = df[disease].eq(1)
    group = positive.ne(positive.shift()).cumsum()

    positive_rows = df.loc[positive].copy()
    positive_rows["period_id"] = group.loc[positive].values

    for _, part in positive_rows.groupby("period_id"):
        start = part["date"].min()
        end = part["date"].max()

        periods.append({
            "disease": disease,
            "start_date": start,
            "end_date": end,
            "observations": len(part),
            "calendar_span_days": (end - start).days + 1,
            "contains_date_gaps": (
                part["date"].diff().dropna().dt.days.gt(1).any()
            ),
        })

result = pd.DataFrame(periods).sort_values(
    ["disease", "start_date"]
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)

for disease in diseases:
    part = result[result["disease"] == disease]
    print(f"\n{'=' * 50}")
    print(f"{disease}: {len(part)} positive periods")
    print("Median observations per period:", part["observations"].median())
    print("Longest period by observations:")
    print(
        part.sort_values("observations", ascending=False)
        .head(10)
        .to_string(index=False)
    )

print("\nSaved period details to:", OUTPUT_PATH)
