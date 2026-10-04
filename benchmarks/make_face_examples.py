"""Build examples/faces/ from outputs/: the faces stage on two Unsplash group photos, with the model steps as figures.

    uv run --extra faces python benchmarks/make_face_examples.py

Expects, for each example name below, outputs/<name>/ from `tracesmart trace` (the plain trace) and its children
`faces/` and `faces-detailed/` from `tracesmart faces ... --style cartoon|detailed`, all run with --max-side 1280.
Optionally outputs/<name>-care/ made the same way with --care words: it becomes row 2 of overview.jpg. The original
photos are read from tests/images/original/<name>.jpg (gitignored), so the models see full resolution.
Writes downscaled JPEG panels, the SVGs, the per-face step strips, the detector overlay and the crop-versus-whole-image
parse comparison.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from face_figures import overview  # noqa: E402

from tracesmart import backends, faces  # noqa: E402

EXAMPLES = {  # example folder -> (outputs/ name, title, photographer, Unsplash photo id)
    "lake-friends": ("tim", "Five friends by the lake", "Tim Mossholder", "hOF1bWoet_Q"),
    "stairs": ("joel", "Group on the stairs", "Joel Muniz", "HvZDCuRnSaY"),
}
T = 300  # tile size of the step strips
PALETTE = np.array([[0, 0, 0], [240, 200, 170], [235, 150, 120], [255, 0, 0], [60, 90, 255], [60, 90, 255],
                    [255, 170, 0], [255, 170, 0], [150, 220, 150], [150, 220, 150], [120, 20, 40], [230, 80, 110],
                    [230, 80, 110], [170, 90, 200], [90, 90, 90], [150, 220, 150], [60, 150, 90], [60, 150, 90],
                    [60, 200, 60]], np.uint8)  # face-parsing classes; glasses red, hair purple


def jpg(im: Image.Image, dst: Path, width: int | None = None, q: int = 85) -> None:
    im = im.convert("RGB")
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(dst, quality=q, optimize=True)


def whole_image_labels(image: Image.Image, device: str) -> np.ndarray:
    """The face parser run on the whole image (long side 1024), to compare with the per-face crops."""
    w, h = image.size
    faces.parse_labels(image.crop((0, 0, 8, 8)), device)  # loads the model
    proc, net = faces._PARSER[device]
    k = 1024 / max(w, h)
    size = (round(w * k / 32) * 32, round(h * k / 32) * 32)
    inputs = proc(images=image.resize(size, Image.LANCZOS), return_tensors="pt", do_resize=False).to(device)
    with torch.no_grad():
        logits = net(**inputs).logits
    return torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear").argmax(1)[0].cpu().numpy()


def overlay(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    a = np.where(labels[..., None] > 0, 0.62, 0.0)
    return (rgb * (1 - a) + PALETTE[labels] * a).astype(np.uint8)


def step_figures(image: Image.Image, root: Path, out: Path, device: str) -> int:
    """detect.jpg, parse.jpg (crops stitched vs whole image) and steps/face-N.jpg for every face."""
    rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    fan = faces.fan()
    boxes = [b for b in fan.face_detector.detect_from_image(rgb) if min(b[2] - b[0], b[3] - b[1]) >= faces.MIN_FACE]
    boxes.sort(key=lambda b: b[0])
    drawn = rgb.copy()
    for b in boxes:
        cv2.rectangle(drawn, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 60, 60), 3)
    jpg(Image.fromarray(drawn), out / "detect.jpg", 1280)

    stitched = np.zeros((h, w), np.uint8)  # what the tight per-face crops give, big faces first
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        x0, y0, x1, y1 = (int(v) for v in b[:4])
        s = int(max(x1 - x0, y1 - y0) * faces.MARGIN)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        region, _ = faces.parse_region(image, cx, cy, s, device)
        stitched[region > 0] = region[region > 0]
    whole = whole_image_labels(image, device).astype(np.uint8)
    jpg(Image.fromarray(np.concatenate([rgb, overlay(rgb, stitched), overlay(rgb, whole)], 1)), out / "parse.jpg", 2400)

    (out / "steps").mkdir(exist_ok=True)
    vectors = {k: Image.open(root / d / "vector.png").convert("RGB")
               for k, d in (("cartoon", "faces"), ("detailed", "faces-detailed"))}
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = (int(v) for v in b[:4])
        s = int(max(x1 - x0, y1 - y0) * faces.MARGIN)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        box = (cx - s, cy - s, cx + s, cy + s)
        crop = image.crop(box)
        labels, _ = faces.parse_labels(crop, device)
        size = faces.SIDE
        corners = ((x0, box[0]), (y0, box[1]), (x1, box[0]), (y1, box[1]))
        found = fan.get_landmarks_from_image(np.asarray(crop.resize((size, size), Image.LANCZOS)),
                                             detected_faces=[[(v - o) * size / (2 * s) for v, o in corners]])
        land = np.asarray(crop.resize((size, size), Image.LANCZOS)).copy()
        if found:
            for p in found[0]:
                cv2.circle(land, (int(p[0]), int(p[1])), 3, (255, 255, 0), -1)
            for a, c in ((36, 42), (42, 48), (17, 22), (22, 27), (48, 60), (60, 68)):
                cv2.polylines(land, [found[0][a:c].astype(np.int32)], a in (36, 42, 48, 60), (0, 255, 255), 2)
        tiles = [crop.resize((T, T), Image.LANCZOS),
                 Image.fromarray(PALETTE[labels]).resize((T, T), Image.NEAREST),
                 Image.fromarray(land).resize((T, T), Image.LANCZOS)]
        for k in ("cartoon", "detailed"):
            z = vectors[k].width / w
            tiles.append(vectors[k].crop(tuple(int(v * z) for v in box)).resize((T, T), Image.LANCZOS))
        sheet = Image.new("RGB", (T * len(tiles), T))
        for j, t in enumerate(tiles):
            sheet.paste(t, (j * T, 0))
        jpg(sheet, out / "steps" / f"face-{i + 1}.jpg", q=88)
    return len(boxes)


def main() -> None:
    device = backends.best_device()
    for example, (name, title, who, photo_id) in EXAMPLES.items():
        root, out = Path("outputs") / name, Path("examples/faces") / example
        (out / "panels").mkdir(parents=True, exist_ok=True)
        image = Image.open(root / "faces" / "source.png").convert("RGB")
        jpg(image, out / "source.jpg", q=88)
        shutil.copy(root / "vector.svg", out / "vector-trace.svg")
        shutil.copy(root / "faces" / "vector.svg", out / "vector-cartoon.svg")
        shutil.copy(root / "faces-detailed" / "vector.svg", out / "vector-detailed.svg")
        panels = {"photo": root / "faces" / "source.png", "trace-only": root / "vector.png",
                  "cartoon": root / "faces" / "vector.png", "detailed": root / "faces-detailed" / "vector.png"}
        ims = []
        for panel, f in panels.items():
            im = Image.open(f).convert("RGB")
            jpg(im, out / "panels" / f"{panel}.jpg", 1280)
            ims.append(im.resize(image.size, Image.LANCZOS))
        row = Image.new("RGB", (image.width * 4 + 60, image.height), "#222222")
        for i, im in enumerate(ims):
            row.paste(im, (i * (image.width + 20), 0))
        jpg(row, out / "compare.jpg", 2400)
        n = step_figures(image, root, out, device)
        care = Path("outputs") / f"{name}-care"
        original = Path("tests/images/original") / f"{name}.jpg"
        jpg(overview(root, care if care.exists() else None, original if original.exists() else None, device),
            out / "overview.jpg", 3000)
        meta = json.loads((root / "run.json").read_text()) if (root / "run.json").exists() else {}
        (out / "run.json").write_text(json.dumps({
            "title": title, "photographer": who, "unsplash": photo_id, "faces": n,
            "trace": {k: meta.get(k) for k in ("grid", "impact", "rounds", "max_side")},
            "commands": ["tracesmart trace photo.jpg -o out --max-side 1280 --grid 48 --impact 3e-5 --rounds 3",
                         "tracesmart faces photo.jpg out/masks.npz --max-side 1280",
                         "tracesmart faces photo.jpg out/masks.npz --max-side 1280 --style detailed "
                         "-o out/faces-detailed"],
        }, indent=1))
        print(example, n, "faces")
    total = sum(f.stat().st_size for f in Path("examples/faces").rglob("*") if f.is_file())
    print(f"examples/faces/: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
