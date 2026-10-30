import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

df = pd.read_csv("environmental_data/environmental_cleaned.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

targets = ["down_mildew", "powdery_mildew"]

weather_cols = [
    "air_hum", "air_hum_min", "air_hum_max",
    "air_temp", "air_temp_min", "air_temp_max", "pluvm"
]

# Features available before the current day's label.
feature_cols = []

for col in weather_cols:
    previous = df[col].shift(1)
    df[f"{col}_prev1"] = previous
    feature_cols.append(f"{col}_prev1")

    for stat in ["mean", "min", "max"]:
        rolling = df[col].shift(1).rolling(7, min_periods=7)

        if stat == "mean":
            values = rolling.mean()
        elif stat == "min":
            values = rolling.min()
        else:
            values = rolling.max()

        name = f"{col}_7day_{stat}"
        df[name] = values
        feature_cols.append(name)

for target in targets:
    df[f"{target}_previous"] = df[target].shift(1)

combined_features = feature_cols + [
    f"{target}_previous" for target in targets
]

required = list(set(
    feature_cols + targets +
    [f"{target}_previous" for target in targets]
))

data = df.dropna(subset=required).reset_index(drop=True)
split_at = int(len(data) * 0.8)
train = data.iloc[:split_at].copy()
test = data.iloc[split_at:].copy()

print("Train:", len(train), train["date"].min().date(),
      "to", train["date"].max().date())
print("Test:", len(test), test["date"].min().date(),
      "to", test["date"].max().date())

def evaluate(name, y, pred, scores):
    print(f"\n{name}")
    print("Average precision:", round(average_precision_score(y, scores), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(y, pred), 4))
    print("Precision:", round(precision_score(y, pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y, pred, zero_division=0), 4))
    print("F1:", round(f1_score(y, pred, zero_division=0), 4))

for target in targets:
    print("\n" + "=" * 60)
    print("TARGET:", target)

    y_train = train[target].astype(int)
    y_test = test[target].astype(int)

    # Persistence baseline: previous record's label.
    persistence_pred = test[f"{target}_previous"].astype(int)
    evaluate(
        "Persistence baseline",
        y_test,
        persistence_pred,
        persistence_pred
    )

    # Weather-only model.
    weather_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000)
    )
    weather_model.fit(train[feature_cols], y_train)
    weather_scores = weather_model.predict_proba(test[feature_cols])[:, 1]
    weather_pred = (weather_scores >= 0.5).astype(int)
    evaluate("Weather-only model", y_test, weather_pred, weather_scores)

    # Combined model: weather + previous labels for BOTH targets.
    combined_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000)
    )
    combined_model.fit(train[combined_features], y_train)
    combined_scores = combined_model.predict_proba(
        test[combined_features]
    )[:, 1]
    combined_pred = (combined_scores >= 0.5).astype(int)
    evaluate("Weather + previous labels", y_test, combined_pred, combined_scores)
