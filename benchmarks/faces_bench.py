"""Run the faces stage over a folder of photos and report how it did.

    uv run --extra faces python benchmarks/faces_bench.py ~/Downloads/benchmark-photos [--out outputs/facebench]

For each photo: downscale to 1280 px, plain trace (slow: SAM, minutes), faces stage in both styles, timings, and a
count of the parts drawn. The face models read the original photo (`--no-from-original` for the downscaled image).
Steps that already have their output are skipped, so a run can be resumed. With `--care "a, b"` and
`--reuse <earlier out dir>` the trace starts from that run's cached SAM masks and only adds SAM 3, and each photo's
`overview.jpg` gets the earlier run as row 1 and this one as row 2. Writes
<out>/summary.json, <out>/summary.md and <out>/contact.jpg (photo | cartoon | detailed for every photo).
Judging quality is by eye on the contact sheet and the per-photo folders; the counts say what was drawn, not whether
it is right.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import typer
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from face_figures import overview  # noqa: E402

from tracesmart import backends, faces  # noqa: E402
from tracesmart.render import write_outputs  # noqa: E402

PARTS = ("hair", "neck", "skin", "eye", "eyebrow", "mouth", "mouth line", "teeth", "glasses", "sunglasses")


def main(photos: Path, out: Path = Path("outputs/facebench"), max_side: int = 1280, grid: int = 32, rounds: int = 2,
         only: str = "", from_original: bool = True, care: str = "", reuse: Path | None = None) -> None:
    files = sorted(p for p in photos.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                   and only in p.name)
    device = backends.best_device()
    rows = []
    for photo in files:
        name = photo.stem
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        src = d / "source.jpg"
        if not src.exists():
            im = Image.open(photo).convert("RGB")
            k = max_side / max(im.size)
            im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS).save(src, quality=92)
        t0 = time.perf_counter()
        if reuse is not None and not (d / "raw_masks.npz").exists():  # start from an earlier run's cached SAM stage
            shutil.copy(reuse / name / "raw_masks.npz", d / "raw_masks.npz")
        if not (d / "masks.npz").exists():
            print(f"[{name}] tracing ...", flush=True)
            cmd = [sys.executable, "-c", "from tracesmart.cli import main; main()", "trace", str(src), "-o", str(d),
                   "--max-side", str(max_side), "--grid", str(grid), "--rounds", str(rounds)]
            subprocess.run(cmd + (["--care", care] if care else []), check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        trace_s = time.perf_counter() - t0
        image = Image.open(src).convert("RGB")
        source = Image.open(photo).convert("RGB") if from_original else None  # the models read the original
        if source is not None and source.width <= image.width:
            source = None
        data = np.load(d / "masks.npz")
        items = list(zip(data["masks"], [p or None for p in data["phrases"].tolist()], strict=True))
        row = {"photo": name, "size": list(image.size), "trace_s": round(trace_s, 1)}
        for style, sub in (("cartoon", "faces"), ("detailed", "faces-detailed")):
            t = time.perf_counter()
            new, counts = faces.add_details(items, image, device, style, log=lambda *_: None, source=source)
            seconds = time.perf_counter() - t
            (d / sub).mkdir(exist_ok=True)
            masks, phrases = [m for m, _ in new], [p for _, p in new]
            np.savez_compressed(d / sub / "masks.npz", masks=np.stack(masks),
                                phrases=np.array([p or "" for p in phrases]))
            write_outputs(image, masks, phrases, d / sub, None, 2.0, None, "none", "person", True)
            if style == "cartoon":
                row.update(faces=counts.get("skin", 0), seconds=round(seconds, 1),
                           parts={k: counts[k] for k in PARTS if k in counts})
            print(f"[{name}] {style}: {sum(counts.values())} parts in {seconds:.1f}s", flush=True)
        rows.append(row)
        auto, with_care = (reuse / name, d) if reuse is not None else (d, None)
        overview(auto, with_care, photo, device).save(d / "overview.jpg", quality=88)  # photo | segments+faces | ...
        (out / "summary.json").write_text(json.dumps(rows, indent=1))

    lines = ["| photo | size | faces | hair | skin | eyes | brows | mouth | teeth | glasses | seconds |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r["parts"]
        lines.append(f"| {r['photo']} | {r['size'][0]}x{r['size'][1]} | {r['faces']} | {p.get('hair', 0)} | "
                     f"{p.get('skin', 0)} | {p.get('eye', 0)} | {p.get('eyebrow', 0)} | "
                     f"{p.get('mouth', 0) + p.get('mouth line', 0)} | {p.get('teeth', 0)} | "
                     f"{p.get('glasses', 0) + p.get('sunglasses', 0)} | {r['seconds']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    h = 420  # contact sheet: photo | cartoon | detailed, one row per photo
    sheet = Image.new("RGB", (3 * 640 + 40, len(rows) * (h + 10)), "#222222")
    for i, r in enumerate(rows):
        for j, f in enumerate(("source.jpg", "faces/vector.png", "faces-detailed/vector.png")):
            im = Image.open(out / r["photo"] / f).convert("RGB")
            im.thumbnail((640, h))
            sheet.paste(im, (j * 660, i * (h + 10)))
    sheet.save(out / "contact.jpg", quality=88)


if __name__ == "__main__":
    typer.run(main)
