"""Debug and overview figures for the faces stage (used by faces_bench.py and make_face_examples.py).

One row per run: photo | segments + faces | trace without faces | cartoon | detailed.
The "segments + faces" panel is the trace's segment map (every shape in its own colour) with, for every face found by
RetinaFace, the tight face box (white), the crop the models read (blue: room for hair and neck), the face-parsing
labels across that crop, and the 68 landmark points (green if the stage trusts them,
red if it rejects them, so a bad face can be told from a bad drawing).
A second row, from a trace made with --care words, goes underneath when there is one.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tracesmart import faces

PALETTE = np.array([[0, 0, 0], [240, 200, 170], [235, 150, 120], [255, 0, 0], [60, 90, 255], [60, 90, 255],
                    [255, 170, 0], [255, 170, 0], [150, 220, 150], [150, 220, 150], [120, 20, 40], [230, 80, 110],
                    [230, 80, 110], [170, 90, 200], [90, 90, 90], [150, 220, 150], [60, 150, 90], [60, 150, 90],
                    [60, 200, 60]], np.uint8)  # face-parsing classes; glasses red, hair purple
COLUMNS = ("photo", "segments + faces", "trace without faces", "cartoon faces", "detailed faces")


def debug_panel(image: Image.Image, segments: Image.Image, device: str,
                source: Image.Image | None = None) -> Image.Image:
    """The segment map of a trace with the face boxes, the parse inside each box and the landmarks drawn on it."""
    w, h = image.size
    canvas = (np.asarray(segments.convert("RGB").resize((w, h), Image.NEAREST)) * 0.55).astype(np.uint8)
    fan = faces.fan()
    rgb = np.asarray(image.convert("RGB"))
    boxes = [b for b in fan.face_detector.detect_from_image(rgb) if min(b[2] - b[0], b[3] - b[1]) >= faces.MIN_FACE]
    line = max(1, round(max(w, h) / 700))
    for box in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):  # small faces drawn last, on top
        x0, y0, x1, y1 = (int(v) for v in box[:4])
        s = max(int(max(x1 - x0, y1 - y0) * faces.MARGIN), 8)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        near, names = faces.parse_region(image, cx, cy, s, device, source)  # the crop the stage reads: face, hair, neck
        a = (near > 0)[..., None] * 0.7
        canvas = (canvas * (1 - a) + PALETTE[near] * a).astype(np.uint8)
        skin = np.isin(near, [i for i, n in names.items() if n in faces.SKIN or n == "eye_g"])  # as the stage trusts it
        crop = faces.grab(image, source, (cx - s, cy - s, cx + s, cy + s))
        pts = faces.find_landmarks(crop, box, (cx - s, cy - s), s)
        if pts is not None:
            trusted = faces.trust_landmarks(pts, skin, box)
            colour = (60, 255, 90) if trusted else (255, 60, 60)
            for p in pts:
                cv2.circle(canvas, (int(p[0]), int(p[1])), line + 1, colour, -1)
            if trusted:
                for lo, hi in ((36, 42), (42, 48), (17, 22), (22, 27), (48, 60), (60, 68)):
                    cv2.polylines(canvas, [pts[lo:hi].astype(np.int32)], lo in (36, 42, 48, 60), (255, 255, 0), line)
        cv2.rectangle(canvas, (cx - s, cy - s), (cx + s, cy + s), (110, 190, 255), line + 1)  # the crop the models read
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), line)  # RetinaFace's tight face box
    return Image.fromarray(canvas)


def run_files(d: Path) -> dict[str, Path]:
    """The files of one run folder: ``d`` is a trace (``source.png``, ``segments.png``, ``vector.png``) with
    ``faces/`` and ``faces-detailed/`` inside it."""
    return {"photo": d / "source.png", "segments": d / "segments.png", "trace": d / "vector.png",
            "cartoon": d / "faces" / "vector.png", "detailed": d / "faces-detailed" / "vector.png"}


def row(d: Path, device: str, original: Image.Image | None, height: int) -> Image.Image:
    f = run_files(d)
    image = Image.open(f["photo"]).convert("RGB")
    panels = [image, debug_panel(image, Image.open(f["segments"]), device, original)]
    panels += [Image.open(f[k]).convert("RGB") for k in ("trace", "cartoon", "detailed")]
    panels = [p.resize((round(p.width * height / p.height), height), Image.LANCZOS) for p in panels]
    gap = 12
    out = Image.new("RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1), height), "#222222")
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def overview(auto: Path, care: Path | None, original: Path | None, device: str, height: int = 520,
             labels: tuple[str, str] = ("automatic", "with --care")) -> Image.Image:
    """Row 1 from the automatic trace, row 2 (if given) from the --care trace, with column headings."""
    src = Image.open(original).convert("RGB") if original else None
    rows = [(labels[0], row(auto, device, src, height))]
    if care is not None and (care / "faces" / "vector.png").exists():
        rows.append((labels[1], row(care, device, src, height)))
    photo = Image.open(run_files(auto)["photo"])
    panel = round(photo.width * height / photo.height) + 12  # every panel has the photo's shape; 12 px gap
    font = ImageFont.load_default(size=max(14, height // 26))
    line = font.size + 8
    width = rows[0][1].width
    sheet = Image.new("RGB", (width, line + sum(line + r.height for _, r in rows)), "#222222")
    draw = ImageDraw.Draw(sheet)
    for i, col in enumerate(COLUMNS):
        draw.text((i * panel + 6, 4), col, fill="#dddddd", font=font)
    y = line
    for name, im in rows:
        draw.text((6, y + 2), name, fill="#ff9ad5", font=font)
        sheet.paste(im, (0, y + line))
        y += line + im.height
    return sheet
