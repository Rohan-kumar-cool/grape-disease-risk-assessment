from pathlib import Path
import sys

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

# Make the module containing CompactSugenoFIS importable.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

scaler_path = BASE_DIR / "model_outputs_anfis" / "environment_scaler.joblib"
models_path = BASE_DIR / "model_outputs_anfis" / "environment_anfis_models.joblib"

scaler = joblib.load(scaler_path)
models = joblib.load(models_path)

# One example input, in the exact feature order used during training.
sample = np.array([[25, 80, 65, 90, 5, 3, 4]], dtype=float)
scaled_sample = scaler.transform(sample)

print("Loaded scaler successfully.")
print("Loaded model targets:", list(models.keys()))

for target, model in models.items():
    prediction = model.predict_proba(scaled_sample)
    print(f"{target} output:", prediction)

print("ANFIS model loading and prediction test completed.")