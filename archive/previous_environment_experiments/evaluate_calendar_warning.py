
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
OUTPUT_PATH = "environmental_data/calendar_warning_results.csv"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

weather_cols = [
    "air_hum", "air_hum_min", "air_hum_max",
    "air_temp", "air_temp_min", "air_temp_max", "pluvm",
]
targets = ["down_mildew", "powdery_mildew"]
horizons = [1, 3, 7]

# Reindex to a complete daily calendar. Missing weather stays missing.
calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
df = df.set_index("date").reindex(calendar)
df.index.name = "date"
df = df.reset_index()

# Construct weather features using only dates strictly before the forecast origin.
for col in weather_cols:
    past = df[col].shift(1)
    df[f"{col}_previous_day"] = past
    df[f"{col}_7day_mean"] = past.rolling(7, min_periods=5).mean()
    df[f"{col}_7day_max"] = past.rolling(7, min_periods=5).max()
    df[f"{col}_7day_min"] = past.rolling(7, min_periods=5).min()

feature_cols = [
    c for c in df.columns
    if c.endswith((
        "_previous_day", "_7day_mean", "_7day_max", "_7day_min"
    ))
]

all_results = []

for horizon in horizons:
    for target in targets:
        data = df.copy()

        # Exact calendar-day target, not row offset.
        data["future_target"] = data[target].shift(-horizon)

        # Persistence predicts the future value using the latest known label.
        data["persistence_prediction"] = data[target].shift(1)

        # Only rows with observed target and a usable weather history.
        data = data.dropna(
            subset=feature_cols + ["future_target", "persistence_prediction"]
        ).reset_index(drop=True)

        split_at = int(len(data) * 0.8)
        train = data.iloc[:split_at].copy()
        test = data.iloc[split_at:].copy()

        y_train = train["future_target"].astype(int)
        y_test = test["future_target"].astype(int)

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print(f"Skipping {target}, horizon {horizon}: one class only")
            continue

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        )
        model.fit(train[feature_cols], y_train)

        probabilities = model.predict_proba(test[feature_cols])[:, 1]
        weather_predictions = (probabilities >= 0.5).astype(int)
        persistence_predictions = test["persistence_prediction"].astype(int)

        for model_name, predictions, scores in [
            ("Weather only", weather_predictions, probabilities),
            ("Persistence", persistence_predictions, persistence_predictions),
        ]:
            all_results.append({
                "target": target,
                "horizon_days": horizon,
                "model": model_name,
                "test_start": test["date"].min().date(),
                "test_end": test["date"].max().date(),
                "test_rows": len(test),
                "prevalence": round(y_test.mean(), 4),
                "average_precision": round(
                    average_precision_score(y_test, scores), 4
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

results = pd.DataFrame(all_results)
print(results.to_string(index=False))
results.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved results to {OUTPUT_PATH}")
