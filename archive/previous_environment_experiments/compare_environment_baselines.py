import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

df = pd.read_csv("environmental_data/environmental_cleaned.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

targets = ["down_mildew", "powdery_mildew"]

# Previous-record label: persistence baseline
for target in targets:
    df[f"{target}_previous"] = df[target].shift(1)

data = df.dropna(subset=targets + [
    f"{target}_previous" for target in targets
]).reset_index(drop=True)

split_at = int(len(data) * 0.8)
test = data.iloc[split_at:].copy()

print("Test dates:",
      test["date"].min().date(), "to",
      test["date"].max().date())
print("Test rows:", len(test))

for target in targets:
    print("\n" + "=" * 55)
    print("TARGET:", target)

    y = test[target].astype(int)
    pred = test[f"{target}_previous"].astype(int)

    print("Actual positive count:", int(y.sum()))
    print("Previous-label positive count:", int(pred.sum()))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y, pred, labels=[0, 1]))

    print("Balanced accuracy:",
          round(balanced_accuracy_score(y, pred), 4))
    print("Precision:",
          round(precision_score(y, pred, zero_division=0), 4))
    print("Recall:",
          round(recall_score(y, pred, zero_division=0), 4))
    print("F1:",
          round(f1_score(y, pred, zero_division=0), 4))
    print("Positive prevalence:", round(y.mean(), 4))

    print("Average precision of previous-label predictions:",
          round(average_precision_score(y, pred), 4))
