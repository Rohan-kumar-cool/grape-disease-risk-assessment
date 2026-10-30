
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "ngld_clean_manifest.csv"
OUTPUT_PATH = PROJECT_DIR / "ngld_dataset_split.csv"

df = pd.read_csv(MANIFEST_PATH)

required = {"sha256", "filename", "path", "label"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if df["sha256"].duplicated().any():
    raise ValueError("Duplicate hashes found in the manifest.")

# Confirm every representative image exists.
missing_files = df.loc[~df["path"].map(lambda p: Path(p).is_file())]
if not missing_files.empty:
    print("Missing image paths:")
    print(missing_files[["filename", "path"]].to_string(index=False))
    raise SystemExit("Fix missing paths before splitting.")

# First: 70% train, 30% temporary.
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"],
)

# Split temporary set equally into 15% validation and 15% test.
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"],
)

train_df = train_df.assign(split="train")
val_df = val_df.assign(split="validation")
test_df = test_df.assign(split="test")

result = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True,
)

# Verify no exact image content crosses splits.
hash_split_counts = result.groupby("sha256")["split"].nunique()
if (hash_split_counts > 1).any():
    raise ValueError("Exact duplicate leakage detected across splits.")

# Save the split manifest.
result.to_csv(OUTPUT_PATH, index=False)

print("Total images:", len(result))
print("\nSplit sizes:")
print(result["split"].value_counts())

print("\nClass counts by split:")
print(pd.crosstab(result["label"], result["split"]))

print("\nExact-hash leakage:", (hash_split_counts > 1).sum())
print("Missing image files:", 0)
print("Saved to:", OUTPUT_PATH)
