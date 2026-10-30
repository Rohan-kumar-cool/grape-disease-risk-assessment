
from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_FILE = (
    BASE_DIR / "environmental_data" / "anfis_mildew_condition_labels.csv"
)
OUTPUT_DIR = BASE_DIR / "model_outputs_anfis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import the ANFIS class from a normal, importable Python module.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_anfis import CompactSugenoFIS, FEATURES, TARGETS

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"], errors="raise")
df = df.sort_values("date").reset_index(drop=True)

train_end = int(len(df) * 0.70)
train = df.iloc[:train_end].copy()

X_train_raw = train[FEATURES].astype(float)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)

models = {}
for target in TARGETS:
    model = CompactSugenoFIS(n_mfs=2, ridge=1.0)
    model.fit(X_train, train[target].astype(int).to_numpy())
    models[target] = model

# Save to temporary files first, then replace the final artifacts.
scaler_tmp = OUTPUT_DIR / "environment_scaler.tmp.joblib"
models_tmp = OUTPUT_DIR / "environment_anfis_models.tmp.joblib"

joblib.dump(scaler, scaler_tmp)
joblib.dump(models, models_tmp)

scaler_tmp.replace(OUTPUT_DIR / "environment_scaler.joblib")
models_tmp.replace(OUTPUT_DIR / "environment_anfis_models.joblib")

metadata = {
    "features": FEATURES,
    "targets": TARGETS,
    "train_records": len(train),
    "train_date_range": [
        str(train["date"].min().date()),
        str(train["date"].max().date()),
    ],
    "note": (
        "Outputs learn rule-generated weather-favourability labels; "
        "they are not validated field disease probabilities."
    ),
}

metadata_tmp = OUTPUT_DIR / "environment_model_metadata.tmp.json"
metadata_tmp.write_text(json.dumps(metadata, indent=2))
metadata_tmp.replace(OUTPUT_DIR / "environment_model_metadata.json")

print("Saved reusable environmental models successfully.")
print("Scaler:", OUTPUT_DIR / "environment_scaler.joblib")
print("Models:", OUTPUT_DIR / "environment_anfis_models.joblib")
print("Metadata:", OUTPUT_DIR / "environment_model_metadata.json")
print("Features:", FEATURES)
print("Training records:", len(train))
