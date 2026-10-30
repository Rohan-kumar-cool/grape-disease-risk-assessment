
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = "environmental_data/environmental_cleaned.csv"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

weather_cols = [
    "air_hum",
    "air_hum_min",
    "air_hum_max",
    "air_temp",
    "air_temp_min",
    "air_temp_max",
    "pluvm",
]

targets = ["down_mildew", "powdery_mildew"]
horizons = [1, 3, 7]

# Build features using weather records available up to yesterday.
# Do not use today's or future weather to predict a future label.
for col in weather_cols:
    previous = df[col].shift(1)
    df[f"{col}_previous_day"] = previous
    df[f"{col}_7day_mean"] = previous.rolling(7).mean()
    df[f"{col}_7day_max"] = previous.rolling(7).max()
    df[f"{col}_7day_min"] = previous.rolling(7).min()

feature_cols = [
    c for c in df.columns
    if c.endswith((
        "_previous_day", "_7day_mean", "_7day_max", "_7day_min"
    ))
]

results = []

for horizon in horizons:
    for target in targets:
        data = df.copy()

        # Target is the indicator observed H rows into the future.
        # We will verify date gaps separately.
        data["future_target"] = data[target].shift(-horizon)
        data = data.dropna(
            subset=feature_cols + ["future_target"]
        ).reset_index(drop=True)

        split_at = int(len(data) * 0.8)
        train = data.iloc[:split_at]
        test = data.iloc[split_at:]

        y_train = train["future_target"].astype(int)
        y_test = test["future_target"].astype(int)

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print(f"Skipping {target}, horizon {horizon}: one class only")
            continue

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000
            ),
        )

        model.fit(train[feature_cols], y_train)
        probabilities = model.predict_proba(test[feature_cols])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results.append({
            "target": target,
            "horizon_rows": horizon,
            "test_start": test["date"].min().date(),
            "test_end": test["date"].max().date(),
            "test_rows": len(test),
            "positive_prevalence": round(y_test.mean(), 4),
            "average_precision": round(
                average_precision_score(y_test, probabilities), 4
            ),
            "balanced_accuracy": round(
                balanced_accuracy_score(y_test, predictions), 4
            ),
            "precision": round(
                precision_score(y_test, predictions, zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_test, predictions, zero_division=0), 4
            ),
            "f1": round(
                f1_score(y_test, predictions, zero_division=0), 4
            ),
        })

result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

output_path = "environmental_data/early_warning_results.csv"
result_df.to_csv(output_path, index=False)
print(f"\nSaved results to {output_path}")
