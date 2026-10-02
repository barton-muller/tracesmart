"""The README's pipeline figure: photo, raw SAM 2.1 masks, after filtering, with --care phrases, result.

    uv run --group bench python benchmarks/pipeline_figure.py NAME OUT.jpg
Reads outputs/NAME/{care,auto}/ as made by `tracesmart trace`.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

name, out = sys.argv[1], Path(sys.argv[2])
root = Path("outputs") / name
photo = Image.open(root / "care" / "source.png").convert("RGB")
w, h = photo.size

raw = np.load(root / "care" / "raw_masks.npz")["masks"]
rng = np.random.default_rng(3)
over = np.asarray(photo).astype(float)
for m in sorted(raw, key=lambda m: -m.sum()):
    over[m] = over[m] * 0.4 + rng.uniform(40, 255, 3) * 0.6
raw_img = Image.fromarray(over.astype(np.uint8))


def shape_count(svg: Path) -> int:
    return svg.read_text().count("<path")


panels = [
    (photo, "1. The photo"),
    (raw_img, f"2. SAM 2.1 proposes {len(raw)} masks"),
    (Image.open(root / "auto" / "segments.png").convert("RGB"),
     f"3. Filter by impact keeps {shape_count(root / 'auto' / 'vector.svg')} shapes"),
    (Image.open(root / "care" / "segments.png").convert("RGB"),
     "4. --care adds named shapes (magenta)"),
    (Image.open(root / "care" / "vector.png").convert("RGB"),
     f"5. Result: {shape_count(root / 'care' / 'vector.svg')} flat-colour paths"),
]
pw = 520
ph = round(h * pw / w)
font = ImageFont.load_default(size=20)
sheet = Image.new("RGB", (len(panels) * (pw + 10) + 10, ph + 56), "#1b1b1b")
draw = ImageDraw.Draw(sheet)
for i, (im, label) in enumerate(panels):
    x = 10 + i * (pw + 10)
    sheet.paste(im.resize((pw, ph), Image.LANCZOS), (x, 10))
    draw.text((x, ph + 20), label, fill="#ffffff", font=font)
sheet.save(out, quality=86, optimize=True)
print(out, sheet.size)
