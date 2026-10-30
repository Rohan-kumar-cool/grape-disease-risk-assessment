
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
import math

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_DIR / "near_duplicate_candidates.csv"
OUTPUT_DIR = PROJECT_DIR / "near_duplicate_review"

# Review the strongest candidates first.
MAX_PAIRS = 40
THUMB_SIZE = (220, 220)
LABEL_HEIGHT = 65

df = pd.read_csv(INPUT_CSV)

if df.empty:
    print("No candidate pairs to review.")
    raise SystemExit(0)

df = df.sort_values("phash_distance").head(MAX_PAIRS)
OUTPUT_DIR.mkdir(exist_ok=True)

columns = 2
rows = math.ceil(len(df) / columns)
cell_width = THUMB_SIZE[0] + 20
cell_height = THUMB_SIZE[1] + LABEL_HEIGHT + 20

sheet = Image.new(
    "RGB",
    (columns * cell_width, rows * cell_height),
    "white",
)
draw = ImageDraw.Draw(sheet)

for index, row in enumerate(df.itertuples(index=False)):
    col = index % columns
    r = index // columns
    x = col * cell_width
    y = r * cell_height

    draw.text(
        (x + 5, y + 2),
        f"Pair {index + 1} | distance={row.phash_distance}",
        fill="black",
    )

    for side, path_text, split, label in [
        ("A", row.path_1, row.split_1, row.label_1),
        ("B", row.path_2, row.split_2, row.label_2),
    ]:
        path = Path(path_text)
        thumb_x = x + 5 if side == "A" else x + 115

        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((105, 105))
                sheet.paste(im, (thumb_x, y + 25))

            short_name = path.name[:23]
            draw.text(
                (thumb_x, y + 135),
                f"{side}: {short_name}",
                fill="black",
            )
            draw.text(
                (thumb_x, y + 150),
                f"{split} | {label[:14]}",
                fill="black",
            )
        except Exception as error:
            draw.text(
                (thumb_x, y + 35),
                f"Could not open: {error}",
                fill="red",
            )

    draw.line(
        (x, y + cell_height - 5, x + cell_width - 5, y + cell_height - 5),
        fill="gray",
    )

output_path = OUTPUT_DIR / "near_duplicate_contact_sheet.jpg"
sheet.save(output_path, quality=90)

review_csv = OUTPUT_DIR / "pairs_to_review.csv"
df.to_csv(review_csv, index=False)

print("Pairs shown:", len(df))
print("Contact sheet:", output_path)
print("Pair details:", review_csv)
