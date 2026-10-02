"""Copy the curated parts of outputs/ into examples/ (downscaled JPEG panels, SVGs, metrics, close-ups).

    uv run --group bench python benchmarks/make_examples.py

Expects outputs/<name>/{care,auto,methods}/ from `tracesmart trace` and benchmarks/compare.py.
"""
import shutil
from pathlib import Path

from PIL import Image

PHOTOS = {  # example -> (photographer, Unsplash photo id)
    "oostpoort": ("Alex vd Slikke", "GYIGt-MQwZ8"), "canal": ("Casper van Battum", "25i3kDguOAE"),
    "hikers": ("Dan Ordze", "4GoNeNKEB1M"), "delft-street": ("Folco Masi", "yvByaC2YqPs"),
    "mountain-lake": ("Kalen Emsley", "mgJSkgIo_JI"), "lone-hiker": ("Robert Bye", "JvUVo08dndQ"),
}
PANELS = {"photo": None, "vtracer-defaults": "vtracer-defaults.png", "vtracer-matched": "vtracer-matched-paths.png",
          "supersvg": "supersvg-cvpr-2024.png", "tracesmart-automatic": "tracesmart-automatic.png",
          "tracesmart-care": "tracesmart-care.png"}


def jpg(src: Path, dst: Path, width: int | None = None, q: int = 84) -> None:
    im = Image.open(src).convert("RGB")
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(dst, quality=q, optimize=True)


out = Path("examples")
for name in PHOTOS:
    d = out / name
    (d / "panels").mkdir(parents=True, exist_ok=True)
    care, auto, meth = Path("outputs") / name / "care", Path("outputs") / name / "auto", Path("outputs") / name / "methods"
    jpg(care / "source.png", d / "source.jpg", q=88)
    shutil.copy(care / "vector.svg", d / "vector.svg")
    shutil.copy(auto / "vector.svg", d / "vector-auto.svg")
    shutil.copy(meth / "supersvg.svg", d / "supersvg.svg")
    shutil.copy(meth / "methods.json", d / "metrics.json")
    jpg(care / "compare.png", d / "compare.jpg", 2400, 85)
    jpg(meth / "methods.png", d / "methods.jpg", 2400, 85)
    for panel, f in PANELS.items():
        jpg(care / "source.png" if f is None else meth / f, d / "panels" / f"{panel}.jpg", 1280)
    jpg(care / "segments.png", d / "panels" / "segments.jpg", 1280)
print(f"examples/: {sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.1f} MB")
