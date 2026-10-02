"""Compare segvec with other vectorisers on the same photo.

    uv run --group bench python benchmarks/compare.py NAME --care outputs/NAME/care [--auto outputs/NAME/auto] \
        [--extra "SVGTrace=path/to/result.svg" ...] [--out outputs/NAME/methods]

Every method is rendered at the photo's resolution and scored against it, and a side-by-side sheet is written.
vtracer (a conventional colour-clustering tracer) is run twice: with its defaults, and tuned so its path count is
close to segvec's, which is the fairer comparison. ``--extra`` adds results from tools that cannot be run here
(online tracers, the methods in the papers); give a label and an SVG made from the same photo.

Scores are against the *photo*, so a deliberately simplified illustration scores lower than a trace that copies
every pixel. Read them next to the path count: that is the trade-off being compared.
"""
import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import resvg_py
import vtracer
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def render(svg: Path, size: tuple[int, int]) -> Image.Image:
    png = resvg_py.svg_to_bytes(svg_path=str(svg), width=size[0], height=size[1])
    return Image.open(__import__("io").BytesIO(bytes(png))).convert("RGB").resize(size)


def n_paths(svg: Path) -> int:
    return len(re.findall(r"<path\b", svg.read_text()))


def score(ref: Image.Image, img: Image.Image) -> tuple[float, float]:
    a, b = np.asarray(ref), np.asarray(img)
    return peak_signal_noise_ratio(a, b, data_range=255), structural_similarity(a, b, channel_axis=2, data_range=255)


def run_vtracer(src: Path, dst: Path, **kw) -> int:
    vtracer.convert_image_to_svg_py(str(src), str(dst), colormode="color", hierarchical="stacked", mode="spline", **kw)
    return n_paths(dst)


def tuned_vtracer(src: Path, dst: Path, target: int) -> tuple[int, dict]:
    """Search vtracer's detail knobs for the setting whose path count is closest to ``target``."""
    best = None
    for speckle in (8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160):  # the useful range is narrow
        for precision in (3, 4, 5, 6):
            for diff in (16, 32, 48, 64):
                kw = dict(filter_speckle=speckle, color_precision=precision, layer_difference=diff)
                with tempfile.TemporaryDirectory() as t:
                    n = run_vtracer(src, Path(t) / "x.svg", **kw)
                if best is None or abs(n - target) < abs(best[0] - target):
                    best = (n, kw)
    n = run_vtracer(src, dst, **best[1])
    return n, best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--care", type=Path, required=True, help="segvec run folder made with --care")
    ap.add_argument("--auto", type=Path, help="segvec run folder made without --care")
    ap.add_argument("--extra", action="append", default=[], metavar="LABEL=SVG")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or a.care.parent / "methods"
    out.mkdir(parents=True, exist_ok=True)

    src_png = a.care / "source.png"
    ref = Image.open(src_png).convert("RGB")
    size = ref.size
    care_svg = a.care / "vector.svg"
    target = n_paths(care_svg)

    methods: list[tuple[str, Path, str]] = []  # label, svg, note
    d = out / "vtracer_default.svg"
    if not d.exists():
        run_vtracer(src_png, d)
    methods.append(("vtracer (defaults)", d, "colour clustering"))
    t = out / "vtracer_matched.svg"
    if not t.exists():
        tuned_vtracer(src_png, t, target)
    methods.append(("vtracer (matched paths)", t, "tuned to a similar path count"))
    for label_svg in a.extra:
        label, _, p = label_svg.partition("=")
        methods.append((label, Path(p), "supplied"))
    if a.auto:
        methods.append(("segvec (automatic)", a.auto / "vector.svg", "SAM 2.1 only"))
    methods.append(("segvec (--care)", care_svg, "SAM 2.1 + SAM 3 phrases"))

    panels, rows = [("photo", ref, "")], []
    for label, svg, _ in methods:
        img = render(svg, size)
        psnr, ssim = score(ref, img)
        row = {"method": label, "paths": n_paths(svg), "kb": round(svg.stat().st_size / 1024),
               "psnr": round(psnr, 1), "ssim": round(ssim, 3)}
        rows.append(row)
        panels.append((label, img, f'{row["paths"]} paths · {row["kb"]} KB · PSNR {psnr:.1f} · SSIM {ssim:.2f}'))

    pw = 560
    ph = round(size[1] * pw / size[0])
    sheet = Image.new("RGB", (len(panels) * (pw + 10) + 10, ph + 60), "#1b1b1b")
    draw = ImageDraw.Draw(sheet)
    for i, (label, img, caption) in enumerate(panels):
        x = 10 + i * (pw + 10)
        sheet.paste(img.resize((pw, ph), Image.LANCZOS), (x, 10))
        draw.text((x, ph + 18), label, fill="#ffffff")
        draw.text((x, ph + 34), caption, fill="#aaaaaa")
    sheet.save(out / "methods.png")
    (out / "methods.json").write_text(json.dumps(rows, indent=1))
    print("| method | paths | KB | PSNR | SSIM |\n|---|---|---|---|---|")
    for r in rows:
        print(f'| {r["method"]} | {r["paths"]} | {r["kb"]} | {r["psnr"]} | {r["ssim"]} |')


if __name__ == "__main__":
    main()
