
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

DATA_PATH = Path("environmental_data/environmental_cleaned.csv")
OUTPUT_PATH = Path("outputs/metrics/persistence_baseline_results.csv")

disease_cols = ["down_mildew", "powdery_mildew"]

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in disease_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = (
    df.dropna(subset=["date"])
    .sort_values("date")
    .drop_duplicates(subset=["date"])
    .reset_index(drop=True)
)

split_index = int(len(df) * 0.8)
df["previous_label"] = None

results = []

for disease in disease_cols:
    previous = df[disease].shift(1)
    previous_date = df["date"].shift(1)

    test_index = df.index[split_index:]
    test = df.loc[test_index].copy()

    test["previous_label"] = previous.loc[test_index]
    test["previous_date"] = previous_date.loc[test_index]

    # Evaluate only where a previous observed label is available.
    valid = (
        test[disease].notna()
        & test["previous_label"].notna()
    )

    evaluated = test.loc[valid].copy()

    if evaluated.empty:
        print(f"No valid persistence rows for {disease}.")
        continue

    y_true = evaluated[disease].astype(int)
    y_pred = evaluated["previous_label"].astype(int)

    row = {
        "disease": disease,
        "model": "previous_observation_persistence",
        "test_rows_available": len(test),
        "evaluated_rows": len(evaluated),
        "coverage": len(evaluated) / len(test),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "consecutive_calendar_days_pct": (
            (evaluated["date"] - evaluated["previous_date"])
            .dt.days.eq(1)
            .mean() * 100
        ),
    }

    results.append(row)

    print(f"\n{disease}")
    for key, value in row.items():
        if key not in ("disease", "model"):
            print(f"{key}: {value:.4f}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

print("\nSaved results to:", OUTPUT_PATH)
