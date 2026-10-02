"""A close-up of one region with every shape outlined: vtracer's defaults against tracesmart.

    uv run --group bench python benchmarks/closeup.py NAME X0,Y0,X1,Y1 OUT.jpg     # box in the 1280 px source
Reads outputs/NAME/{care,methods}/ as made by `tracesmart trace` and benchmarks/compare.py.
"""
import io
import re
import sys
from pathlib import Path

import resvg_py
from PIL import Image, ImageDraw

name, box, out = sys.argv[1], tuple(int(v) for v in sys.argv[2].split(",")), Path(sys.argv[3])
root = Path("outputs") / name
src = Image.open(root / "care" / "source.png").convert("RGB")
zoom = 4
def render_outlined(svg_path):
    svg = Path(svg_path).read_text()
    svg = re.sub(r"<path ", '<path stroke="#ffffff" stroke-opacity="0.9" stroke-width="0.35" ', svg)
    png = resvg_py.svg_to_bytes(svg_string=svg, width=src.width * zoom, height=src.height * zoom)
    return Image.open(io.BytesIO(bytes(png))).convert("RGB")
def crop(im, scale):
    return im.crop(tuple(int(v * scale) for v in box)).resize((640, round(640 * (box[3] - box[1]) / (box[2] - box[0]))), Image.LANCZOS)
panels = [("photo", crop(src, 1))]
for label, svg in (("vtracer, defaults: every white line is a shape edge", root / "methods" / "vtracer_default.svg"),
                   ("tracesmart --care: every white line is a shape edge", root / "care" / "vector.svg")):
    panels.append((label, crop(render_outlined(svg), zoom)))
w, h = panels[0][1].size
sheet = Image.new("RGB", (3 * w + 40, h + 34), "#1b1b1b")
d = ImageDraw.Draw(sheet)
for i, (label, im) in enumerate(panels):
    sheet.paste(im, (10 + i * (w + 10), 10))
    d.text((10 + i * (w + 10), h + 16), label, fill="#cccccc")
sheet.save(out, quality=88)
print(out, sheet.size, src.size)
