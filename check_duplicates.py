
from pathlib import Path
from collections import defaultdict
import hashlib
import csv

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "NGLD" / "NGLD" / "Grapes Disease Dataset"
REPORT_PATH = PROJECT_DIR / "ngld_duplicate_report.csv"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Group files by their SHA-256 content hash
hash_groups = defaultdict(list)

for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    for path in class_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_groups[digest].append({
                "path": str(path),
                "filename": path.name,
                "label": class_dir.name,
            })

duplicate_groups = {
    digest: files
    for digest, files in hash_groups.items()
    if len(files) > 1
}

conflicting_groups = {
    digest: files
    for digest, files in duplicate_groups.items()
    if len({item["label"] for item in files}) > 1
}

with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sha256",
            "duplicate_group_size",
            "filename",
            "label",
            "path",
            "labels_conflict",
        ],
    )
    writer.writeheader()

    for digest, files in sorted(duplicate_groups.items()):
        labels = {item["label"] for item in files}
        for item in files:
            writer.writerow({
                "sha256": digest,
                "duplicate_group_size": len(files),
                "filename": item["filename"],
                "label": item["label"],
                "path": item["path"],
                "labels_conflict": len(labels) > 1,
            })

print("Total image files:", sum(len(v) for v in hash_groups.values()))
print("Unique image contents:", len(hash_groups))
print("Duplicate groups:", len(duplicate_groups))
print(
    "Redundant copies:",
    sum(len(files) - 1 for files in duplicate_groups.values()),
)
print("Groups with conflicting labels:", len(conflicting_groups))
print("Report saved to:", REPORT_PATH)

if conflicting_groups:
    print("\nConflicting duplicate examples:")
    for digest, files in list(conflicting_groups.items())[:10]:
        print("\nHash:", digest)
        for item in files:
            print(f"  {item['label']}: {item['path']}")
