"""Run SuperSVG (CPU) on a photo with a given path budget and restore the photo's aspect ratio.

usage:  SUPERSVG_DIR=/path/to/SuperSVG  /path/to/SuperSVG/.venv/bin/python run_supersvg.py PHOTO PATHS OUT.svg [ITERS]
See benchmarks/SUPERSVG.md for how to build SuperSVG on a Mac. Call the venv's python directly: activating
the venv from a bash script crashed DiffVG here.
SuperSVG works on a 512x512 square, so the photo is squashed to a square, vectorised, and the SVG is then
stretched back to the photo's proportions (a lossless viewBox change).
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

photo, paths, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
iters = sys.argv[4] if len(sys.argv) > 4 else "10"
here = Path(os.environ.get("SUPERSVG_DIR", Path(__file__).resolve().parent / "SuperSVG"))
w, h = Image.open(photo).size
with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    (t / "in").mkdir()
    Image.open(photo).convert("RGB").resize((512, 512), Image.LANCZOS).save(t / "in" / "img.png")
    cmd = [sys.executable, "inference.py", "--input_path", str(t / "in"), "--output_dir", str(t / "out"),
           "--device", "cpu", "--path_num", str(paths), "--optimize_iter", iters]
    for attempt in range(4):  # DiffVG on CPU segfaults now and then; a retry succeeds
        r = subprocess.run(cmd, cwd=here, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            break
        print(f"  attempt {attempt + 1} failed (exit {r.returncode}), retrying", file=sys.stderr)
    else:
        sys.exit("SuperSVG failed 4 times")
    svg = (t / "out" / "img.svg").read_text()
square = 'width="512" height="512"'
svg = svg.replace(square, f'width="{w}" height="{h}" viewBox="0 0 512 512" preserveAspectRatio="none"', 1)
out.write_text(svg)
print(f"{out}: {len(re.findall('<path', svg))} paths")
