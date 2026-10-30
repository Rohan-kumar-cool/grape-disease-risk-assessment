import pandas as pd

df = pd.read_csv("environmental_data/environmental_cleaned.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

targets = ["down_mildew", "powdery_mildew"]

print("Rows:", len(df))
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())

for target in targets:
    print("\n" + "=" * 50)
    print("TARGET:", target)

    previous = df[target].shift(1)
    current = df[target]
    valid = previous.notna()

    print("\nConsecutive-row transition table:")
    print(pd.crosstab(
        previous[valid],
        current[valid],
        rownames=["Previous row"],
        colnames=["Current row"]
    ))

    same_rate = (previous[valid] == current[valid]).mean()
    print(f"\nSame label as previous row: {same_rate:.4f}")

    prev_positive = previous[valid] == 1
    if prev_positive.sum() > 0:
        recall = ((current[valid] == 1) & prev_positive).sum() / prev_positive.sum()
        print(f"Recall when previous row was positive: {recall:.4f}")

    print("Positive labels:", int(current.sum()))

# Check whether consecutive records are consecutive calendar days
gaps = df["date"].diff().dt.days
print("\n" + "=" * 50)
print("Calendar-day gaps between records:")
print(gaps.value_counts().sort_index().head(15))
print("Gaps greater than 1 day:", int((gaps > 1).sum()))
