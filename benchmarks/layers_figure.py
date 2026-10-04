"""Figure for docs/LAYERS: the three detail levels built up, and one object group on its own.

    uv run --group bench python benchmarks/layers_figure.py NAME OUT.jpg
Reads outputs/NAME/care/{masks.npz,source.png}.
"""
import io
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import resvg_py
from PIL import Image, ImageDraw, ImageFont

from tracesmart.render import render

name, out = sys.argv[1], Path(sys.argv[2])
root = Path("outputs") / name / "care"
photo = Image.open(root / "source.png").convert("RGB")
data = np.load(root / "masks.npz")
masks, phrases = list(data["masks"]), [p or None for p in data["phrases"].tolist()]
width = 520


def png(svg: str) -> Image.Image:
    data = resvg_py.svg_to_bytes(svg_string=svg, width=width, background="#ffffff")
    return Image.open(io.BytesIO(bytes(data))).convert("RGB")


levels_svg = render(photo, masks, phrases, layers="levels")[0]
n = [int(c) for c in re.findall(r'inkscape:label="\d [A-Za-z]+ \((\d+)\)"', levels_svg)]
panels = [(photo.resize((width, round(photo.height * width / photo.width))), "1. The photo")]
steps = ((1, f"2. Level 1, structure: {n[0]} shapes"), (2, f"3. + level 2, objects: {sum(n[:2])}"),
         (3, f"4. + level 3, details: {sum(n)}"))
for keep, label in steps:
    svg = levels_svg
    for lv in range(keep + 1, 4):
        svg = re.sub(rf'<g id="layer-{lv}".*?</g>', "", svg, flags=re.S)
    panels.append((png(svg), label))

# one object group on its own: the described object with the most parts inside it
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
objects_svg = render(photo, masks, phrases, layers="objects")[0]
tree = ET.fromstring(objects_svg)
groups = [g for g in tree.iter(f"{{{SVG}}}g") if g.get("id", "").endswith("-group")]
people = [g for g in groups if "person" in g.get("id", "")] or groups
best = max(people, key=lambda g: len(list(g.iter(f"{{{SVG}}}path"))))
alone = ET.Element(tree.tag, tree.attrib)  # a fresh <svg> holding only that group
alone.append(best)
parts = len(list(best.iter(f"{{{SVG}}}path")))
caption = f"5. One group alone: {best.get('id')[:-6]}, {parts} parts"
panels.append((png(ET.tostring(alone, encoding="unicode")), caption))

h = max(p.height for p, _ in panels)
font = ImageFont.load_default(size=20)
sheet = Image.new("RGB", (len(panels) * (width + 10) + 10, h + 56), "#1b1b1b")
draw = ImageDraw.Draw(sheet)
for i, (im, label) in enumerate(panels):
    x = 10 + i * (width + 10)
    sheet.paste(im, (x, 10))
    draw.text((x, h + 20), label, fill="#ffffff", font=font)
sheet.save(out, quality=86, optimize=True)
print(out, sheet.size)
