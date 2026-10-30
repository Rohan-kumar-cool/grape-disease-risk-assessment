
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST = PROJECT_DIR / "ngld_clean_manifest.csv"
CANDIDATES = PROJECT_DIR / "near_duplicate_candidates.csv"
OUTPUT = PROJECT_DIR / "ngld_grouped_split.csv"

df = pd.read_csv(MANIFEST).reset_index(drop=True)
pairs = pd.read_csv(CANDIDATES)

required = {"sha256", "path", "label"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing manifest columns: {required - set(df.columns)}")

if df["sha256"].duplicated().any():
    raise ValueError("Manifest contains duplicate SHA-256 hashes.")

# Union-find: images connected by a near-duplicate candidate
# will belong to the same group.
parent = list(range(len(df)))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    root_a, root_b = find(a), find(b)
    if root_a != root_b:
        parent[root_b] = root_a

path_to_index = {
    str(row.path): i
    for i, row in enumerate(df.itertuples(index=False))
}

for row in pairs.itertuples(index=False):
    a = path_to_index.get(str(row.path_1))
    b = path_to_index.get(str(row.path_2))

    if a is not None and b is not None:
        union(a, b)

df["group_id"] = [find(i) for i in range(len(df))]

group_sizes = df.groupby("group_id").size()
print("Unique images:", len(df))
print("Near-duplicate groups:", len(group_sizes))
print("Groups with multiple images:", int((group_sizes > 1).sum()))
print("Largest group:", int(group_sizes.max()))

# Stratify at the group level where possible. Each group must have one label.
group_table = (
    df.groupby("group_id")
      .agg(label=("label", "first"), size=("label", "size"))
      .reset_index()
)

label_counts = df.groupby("label").size()
group_counts = group_table.groupby("label").size()

if not (group_table.groupby("group_id")["label"].nunique() == 1).all():
    raise ValueError("A group contains conflicting labels; inspect candidates.")

# Split groups, not individual images.
train_groups, temp_groups = train_test_split(
    group_table,
    test_size=0.30,
    random_state=42,
    stratify=group_table["label"],
)

val_groups, test_groups = train_test_split(
    temp_groups,
    test_size=0.50,
    random_state=42,
    stratify=temp_groups["label"],
)

split_by_group = {}
for groups, split in [
    (train_groups, "train"),
    (val_groups, "validation"),
    (test_groups, "test"),
]:
    for group_id in groups["group_id"]:
        split_by_group[group_id] = split

df["split"] = df["group_id"].map(split_by_group)

# Confirm each group belongs to exactly one split.
assert df.groupby("group_id")["split"].nunique().max() == 1

df.to_csv(OUTPUT, index=False)

print("\nImage counts by split:")
print(df["split"].value_counts())

print("\nClass counts by split:")
print(pd.crosstab(df["label"], df["split"]))

print("\nSaved:", OUTPUT)
