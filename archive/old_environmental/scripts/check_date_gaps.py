
from pathlib import Path
import pandas as pd

path = Path("environmental_data/environmental_cleaned.csv")
df = pd.read_csv(path, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

df["days_since_previous"] = df["date"].diff().dt.days

gaps = df.loc[
    df["days_since_previous"] > 1,
    ["date", "days_since_previous"]
].copy()

gaps["previous_date"] = df["date"].shift(1).loc[gaps.index]

print("Number of gaps:", len(gaps))
print("\nDate gaps:")
print(
    gaps[["previous_date", "date", "days_since_previous"]]
    .to_string(index=False)
)

print("\nMissing calendar dates:")
missing_dates = []

for i in range(1, len(df)):
    previous_date = df.loc[i - 1, "date"]
    current_date = df.loc[i, "date"]

    for offset in range(1, (current_date - previous_date).days):
        missing_dates.append(
            previous_date + pd.Timedelta(days=offset)
        )

if missing_dates:
    print(pd.Series(missing_dates).dt.strftime("%Y-%m-%d").to_string(index=False))
else:
    print("No missing calendar dates.")

print("\nDate coverage by year:")
print(df.groupby(df["date"].dt.year)["date"].agg(["min", "max", "count"]))
