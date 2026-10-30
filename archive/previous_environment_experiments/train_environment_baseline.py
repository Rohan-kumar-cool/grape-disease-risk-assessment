
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    average_precision_score,
    confusion_matrix,
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

target_cols = ["down_mildew", "powdery_mildew"]

# Features from the previous 7 days only.
# Shift first so today's weather cannot predict today's label.
for col in weather_cols:
    previous = df[col].shift(1)
    df[f"{col}_previous_day"] = previous
    df[f"{col}_7day_mean"] = previous.rolling(7).mean()
    df[f"{col}_7day_max"] = previous.rolling(7).max()
    df[f"{col}_7day_min"] = previous.rolling(7).min()

feature_cols = [
    c for c in df.columns
    if c.endswith(("_previous_day", "_7day_mean", "_7day_max", "_7day_min"))
]

# Remove rows without a complete previous 7-day history.
data = df.dropna(subset=feature_cols).reset_index(drop=True)

# First 80% of dates for training, final 20% for testing.
split_at = int(len(data) * 0.8)
train = data.iloc[:split_at]
test = data.iloc[split_at:]

print("Training dates:", train["date"].min().date(), "to",
      train["date"].max().date())
print("Testing dates:", test["date"].min().date(), "to",
      test["date"].max().date())
print("Training rows:", len(train), "| Testing rows:", len(test))

for target in target_cols:
    print("\n" + "=" * 55)
    print("TARGET:", target)
    print("Train positives:", int(train[target].sum()), "/", len(train))
    print("Test positives:", int(test[target].sum()), "/", len(test))

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )

    model.fit(train[feature_cols], train[target])

    predictions = model.predict(test[feature_cols])
    probabilities = model.predict_proba(test[feature_cols])[:, 1]

    print("Accuracy:", round(accuracy_score(test[target], predictions), 4))
    print("Balanced accuracy:",
          round(balanced_accuracy_score(test[target], predictions), 4))
    print("Confusion matrix:\n",
          confusion_matrix(test[target], predictions, labels=[0, 1]))
    print("Classification report:\n",
          classification_report(test[target], predictions, zero_division=0))

    if test[target].nunique() == 2:
        print("PR-AUC (average precision):",
              round(average_precision_score(test[target], probabilities), 4))
    else:
        print("PR-AUC unavailable: test set contains only one class.")
