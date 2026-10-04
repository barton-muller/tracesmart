"""Figure for the layers docs: lift a hiker out as one group and move it, with and without background completion.

    uv run --group bench python benchmarks/move_figure.py NAME OUT.jpg [DX]
Reads outputs/NAME/care/{masks.npz,source.png}; DX is how far to move the hiker, in pixels of the 1024 px render.
"""
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import resvg_py
from PIL import Image, ImageDraw, ImageFont

from tracesmart.render import render

name, out = sys.argv[1], Path(sys.argv[2])
dx = float(sys.argv[3]) if len(sys.argv) > 3 else 330.0
root = Path("outputs") / name / "care"
photo = Image.open(root / "source.png").convert("RGB")
data = np.load(root / "masks.npz")
masks, phrases = list(data["masks"]), [p or None for p in data["phrases"].tolist()]

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
width = 500


def png(svg: str) -> Image.Image:
    data = resvg_py.svg_to_bytes(svg_string=svg, width=width)
    return Image.open(io.BytesIO(bytes(data))).convert("RGB")


def hiker_group(tree: ET.Element) -> ET.Element:
    groups = [g for g in tree.iter(f"{{{SVG}}}g") if g.get("id", "").startswith("person-")]
    return max(groups, key=lambda g: len(list(g.iter(f"{{{SVG}}}path"))))


def without(tree: ET.Element, element: ET.Element) -> str:
    parents = {c: p for p in tree.iter() for c in p}
    parents[element].remove(element)
    return ET.tostring(tree, encoding="unicode")


def moved(tree: ET.Element, element: ET.Element, dx: float) -> str:
    parents = {c: p for p in tree.iter() for c in p}
    parent = parents[element]
    parent.remove(element)
    element.set("transform", f"translate({dx},0)")
    parent.append(element)  # on top of everything, like dragging it in an editor
    return ET.tostring(tree, encoding="unicode")


panels = [(png(render(photo, masks, phrases, layers="objects")[0]), "1. Hiker as one group")]
for complete, label in ((False, "without"), (True, "with")):
    svg = render(photo, masks, phrases, layers="objects", complete=complete)[0]
    tree = ET.fromstring(svg)
    number = 3 if complete else 2
    panels.append((png(without(tree, hiker_group(tree))), f"{number}. Lifted out, {label} completion"))
svg = render(photo, masks, phrases, layers="objects", complete=True)[0]
tree = ET.fromstring(svg)
scale = photo.width / 1024  # the move distance is given for a 1024 px wide render
panels.append((png(moved(tree, hiker_group(tree), dx * scale)), "4. Moved aside"))

h = max(p.height for p, _ in panels)
font = ImageFont.load_default(size=19)
sheet = Image.new("RGB", (len(panels) * (width + 10) + 10, h + 50), "#1b1b1b")
draw = ImageDraw.Draw(sheet)
for i, (im, label) in enumerate(panels):
    x = 10 + i * (width + 10)
    sheet.paste(im, (x, 10))
    draw.text((x, h + 18), label, fill="#ffffff", font=font)
sheet.save(out, quality=86, optimize=True)
print(out, sheet.size)
