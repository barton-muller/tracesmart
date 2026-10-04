"""Facial detail as a second stage, run on a finished trace: eyes, brows, lips, teeth, glasses, hair and skin.

SAM cannot resolve an eye a few pixels wide, so this stage draws the parts from where they are. Faces are found with
RetinaFace, each is cropped with a margin and read by two small models:

* a face-parsing SegFormer (``jonathandinu/face-parsing``, CelebAMask-HQ) gives hair, neck, skin and glasses regions;
* ``face-alignment`` (FAN, 68 landmarks; optional extra ``uv sync --extra faces``) gives eyelids, brows, lips and the
  mouth opening.

The shapes are drawn from those positions (an iris is a disc at the darkest spot of the eye, teeth are the pale pixels
of the open mouth, glasses are rings around the eyes). Colours are still sampled from the photo, but ``render``
nudges them towards what the part must look like (teeth towards white, pupils towards black, see
``render.PRIOR_COLOURS``). Shapes are ordinary described masks, so ``--layers`` treats them like ``--care`` shapes.
"""
import cv2
import numpy as np
import torch
from PIL import Image

from .pipeline import tidy

PARSER = "jonathandinu/face-parsing"
SKIN = ("skin", "nose", "l_brow", "r_brow", "l_eye", "r_eye", "mouth", "u_lip", "l_lip", "l_ear", "r_ear")
NECK = ("neck", "neck_l")
PHRASES = ("hair", "neck", "skin", "lips", "mouth", "mouth line", "teeth", "eye", "eye white", "iris", "pupil",
           "eyelid", "eyebrow", "glasses", "sunglasses")
MARGIN = 0.95  # crop half-size as a fraction of the face box's longer side: room for hair
HAIR_COLOUR_GAP = 28  # Lab distance within which a neighbouring trace shape counts as more of the same hair
WIDE = 2.2  # the wide crop for hair and neck is this many times larger, so long hair is not cut off
SIDE = 512  # crop size fed to both models
MIN_FACE = 24  # px: smaller detections are ignored
SMALL_FACE = 40  # px: from here a face also gets lips and an eye line
SEG_FACE = 60  # px: from here the parse gives the brows, the open mouth and (if landmarks fail) the eye dots
FULL_FACE = 80  # px: from here eyes get white, iris and pupil, and an open mouth gets teeth
# (hair, skin and neck at any size; never a nose)
MIN_EYE_SPACING = 0.3  # eye distance as a share of the face box width: less means a profile or bad landmarks
MAX_NOSE_OFFSET = 0.45  # nose tip sideways from the mid-eye point, in eye spacings
BROW_DROP = 0.3 # FAN's brow points trace the brow's top edge: move them this share of the way to the eye
BROW_KEEP = 0.5  # share of the brow band kept, darkest first
SHIFT = 4  # sub-pixel bits for polygon filling: an eye can be 6 px wide
SUNGLASSES = 90  # mean luminance of the glasses region below which the lenses count as dark

_PARSER: dict = {}
_FAN = None


def parse_labels(crop: Image.Image, device: str) -> tuple[np.ndarray, dict[int, str]]:
    """Per-pixel face-parsing labels for ``crop`` at the crop's own size, and the id -> name table."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    if device not in _PARSER:
        _PARSER[device] = (SegformerImageProcessor.from_pretrained(PARSER),
                           SegformerForSemanticSegmentation.from_pretrained(PARSER).to(device).eval())
    proc, net = _PARSER[device]
    inputs = proc(images=crop.resize((SIDE, SIDE), Image.LANCZOS), return_tensors="pt").to(device)
    with torch.no_grad():
        logits = net(**inputs).logits
    logits = torch.nn.functional.interpolate(logits, size=crop.size[::-1], mode="bilinear")
    return logits.argmax(1)[0].cpu().numpy(), net.config.id2label


def fan():
    """The landmark model with its RetinaFace detector (loaded once)."""
    global _FAN
    try:
        import face_alignment
    except ImportError as e:
        raise SystemExit("face detail needs the optional extra: uv sync --extra faces") from e
    if _FAN is None:
        _FAN = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False,
                                            face_detector="retinaface")
    return _FAN


def fill(shape: tuple[int, int], pts: np.ndarray, thickness: int = 0) -> np.ndarray:
    """Rasterise a closed polygon (or, with ``thickness``, an open polyline) with sub-pixel accuracy."""
    canvas = np.zeros(shape, np.uint8)
    p = np.round(pts * (1 << SHIFT)).astype(np.int32)
    if thickness:
        cv2.polylines(canvas, [p], False, 1, thickness, cv2.LINE_8, SHIFT)
    else:
        cv2.fillPoly(canvas, [p], 1, cv2.LINE_8, SHIFT)
    return canvas.astype(bool)


def grow(pts: np.ndarray, k: float) -> np.ndarray:
    """Scale a polygon about its centre."""
    c = pts.mean(0)
    return c + (pts - c) * k


def disc(shape: tuple[int, int], centre, radius: float) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    cv2.circle(canvas, tuple(int(round(v * (1 << SHIFT))) for v in centre), int(round(radius * (1 << SHIFT))), 1, -1,
               cv2.LINE_8, SHIFT)
    return canvas.astype(bool)


def darkest(area: np.ndarray, grey: np.ndarray, keep: float) -> np.ndarray:
    """The darkest ``keep`` share of ``area``: the feature itself rather than the skin around it."""
    if not area.any():
        return area
    dark = area & (grey <= np.quantile(grey[area], keep))
    return cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)


def teeth_mask(inner: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """The pale, unsaturated pixels of an open mouth (``inner``): teeth are not lips, tongue or darkness."""
    if inner.sum() < 12:
        return np.zeros_like(inner)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    val = hsv[..., 2]
    ok = inner & (val >= 0.7 * np.quantile(val[inner], 0.95)) & (hsv[..., 1] < 0.45 * 255)
    return cv2.morphologyEx(ok.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)


def eye_parts(rgb: np.ndarray, grey: np.ndarray, eye: np.ndarray, upper: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """White, iris, pupil and lid line of one eye from its six landmarks (``upper``: the four lid points)."""
    h, w = grey.shape
    width = float(np.linalg.norm(eye[3] - eye[0]))
    if np.ptp(eye[:, 1]) < max(1.0, 0.12 * width):  # closed (laughing, blinking): just the lid line
        return [(fill((h, w), upper, max(1, round(0.12 * width))), "eyelid")]
    area = fill((h, w), grow(eye, 1.1))
    weight = np.where(area, (255.0 - grey) ** 2, 0)
    ys, xs = np.nonzero(area)
    centre = (float((weight[ys, xs] * xs).sum() / weight[ys, xs].sum()),
              float((weight[ys, xs] * ys).sum() / weight[ys, xs].sum()))
    radius = max(1.5, 0.22 * width)
    out = [(area, "eye white"), (disc((h, w), centre, radius) & area, "iris")]
    if radius >= 3:
        out.append((disc((h, w), centre, 0.45 * radius) & area, "pupil"))
    out.append((fill((h, w), upper, max(1, round(0.1 * width))), "eyelid"))
    return out


def cartoon_eye(grey: np.ndarray, eye: np.ndarray, spacing: float, shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    """An eye as a dot where the eye is darkest, or an arc if it is closed (laughing, blinking)."""
    width = float(np.linalg.norm(eye[3] - eye[0]))
    if np.ptp(eye[:, 1]) < max(1.0, 0.12 * width):
        return fill(shape, eye[:4], max(1, round(0.07 * spacing))), "eye"
    area = fill(shape, grow(eye, 1.1))
    weight = np.where(area, (255.0 - grey) ** 2, 0)
    ys, xs = np.nonzero(area)
    centre = ((weight[ys, xs] * xs).sum() / weight[ys, xs].sum(), (weight[ys, xs] * ys).sum() / weight[ys, xs].sum())
    return disc(shape, centre, max(1.2, 0.09 * spacing)), "eye"


def cartoon_mouth(pts: np.ndarray, inner: np.ndarray, size: float, rgb: np.ndarray,
                  shape: tuple[int, int]) -> list[tuple[np.ndarray, str]]:
    """A closed mouth is one line, an open one a dark shape (with a teeth strip on a big face)."""
    width = float(np.linalg.norm(pts[54] - pts[48]))
    if inner.sum() >= 12 and np.ptp(pts[60:68, 1]) >= max(2.0, 0.15 * width):
        parts = [(inner, "mouth")]
        if size >= FULL_FACE:
            parts.append((teeth_mask(inner, rgb), "teeth"))
        return parts
    mid = np.array([pts[60], (pts[61] + pts[67]) / 2, (pts[62] + pts[66]) / 2, (pts[63] + pts[65]) / 2, pts[64]])
    return [(fill(shape, mid, max(2, round(0.04 * width))), "mouth line")]


def stroke(mask: np.ndarray, scale: float = 0.6, min_thick: int = 2, taper: float = 0.0) -> np.ndarray | None:
    """A thin, smooth stroke along the middle of an elongated ``mask`` (a brow, a closed mouth): the mask's centre
    line, averaged along its main axis, drawn about ``scale`` as thick as the mask. None if it is too small."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 6:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    mean = pts.mean(0)
    axis = np.linalg.svd(pts - mean, full_matrices=False)[2][0]
    normal = np.array([-axis[1], axis[0]])
    t, v = (pts - mean) @ axis, (pts - mean) @ normal
    bins = np.round(t).astype(int)
    keep = np.unique(bins)
    if len(keep) < 3:
        return None
    centre = np.array([v[bins == b].mean() for b in keep])
    width = np.array([np.ptp(v[bins == b]) + 1 for b in keep])
    k = max(1, min(5, len(keep) // 3) | 1)  # moving average, odd length
    centre = np.convolve(np.pad(centre, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    line = mean + keep[:, None] * axis + centre[:, None] * normal
    thick = max(min_thick, round(scale * float(np.median(width))))
    return fill(mask.shape, line, thick)


def teeth_from_mouth(mouth: np.ndarray, rgb: np.ndarray) -> np.ndarray | None:
    """Teeth are the parsed inside of the mouth, pulled in from the lips, when that region is mostly pale."""
    if teeth_mask(mouth, rgb).sum() < 0.2 * mouth.sum():
        return None
    ys, xs = np.nonzero(mouth)
    r = max(1, round(0.12 * min(np.ptp(ys) + 1, np.ptp(xs) + 1)))
    return cv2.erode(mouth.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                     ).astype(bool)


def glasses_parts(image_rgb: np.ndarray, region: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """The parsed glasses region as one solid shape (the parse follows the real frames well). Dark lenses count as
    sunglasses and keep their mean colour; clear ones are coloured by the frame, see ``render.shape_colour``."""
    grey = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    solid = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    return [(solid, "sunglasses" if grey[region].mean() < SUNGLASSES else "glasses")]


def parse_region(image: Image.Image, cx: int, cy: int, s: int, device: str) -> tuple[np.ndarray, dict[int, str]]:
    """Face-parsing labels for the square crop of half-size ``s`` round (cx, cy), pasted into a full-size map."""
    w, h = image.size
    labels, names = parse_labels(image.crop((cx - s, cy - s, cx + s, cy + s)), device)
    full = np.zeros((h, w), labels.dtype)
    fx0, fy0, fx1, fy1 = max(cx - s, 0), max(cy - s, 0), min(cx + s, w), min(cy + s, h)
    full[fy0:fy1, fx0:fx1] = labels[fy0 - (cy - s):fy1 - (cy - s), fx0 - (cx - s):fx1 - (cx - s)]
    return full, names


def parse_full(image: Image.Image, device: str) -> np.ndarray:
    """Hair on the whole image (long side 1024). Poor on small faces, but it has hair that runs past a face crop."""
    w, h = image.size
    parse_labels(image.crop((0, 0, 8, 8)), device)  # loads the model
    proc, net = _PARSER[device]
    k = 1024 / max(w, h)
    size = (max(32, round(w * k / 32) * 32), max(32, round(h * k / 32) * 32))
    inputs = proc(images=image.resize(size, Image.LANCZOS), return_tensors="pt", do_resize=False).to(device)
    with torch.no_grad():
        logits = net(**inputs).logits
    labels = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear").argmax(1)[0].cpu().numpy()
    return labels == next(i for i, n in net.config.id2label.items() if n == "hair")


def face_shapes(image: Image.Image, box, device: str, style: str = "cartoon", min_px: int = 4,
                full_hair: np.ndarray | None = None) -> list[tuple[np.ndarray, str]]:
    """Detail shapes for one face (``box`` x0, y0, x1, y1), bottom first, as full-image ``(mask, phrase)`` pairs.

    ``cartoon``: eyes are dots, the mouth a line or a dark open shape, brows strokes. ``detailed``: eye whites,
    irises, pupils, lids, lips and teeth. On big faces the parse gives the brows and the open mouth; landmarks give
    the rest, and everything on small faces. The parsed nose is never a shape: it is part of the skin."""
    rgb = np.asarray(image.convert("RGB"))
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = grey.shape
    x0, y0, x1, y1 = (int(v) for v in box[:4])
    size = min(x1 - x0, y1 - y0)
    face_area = (x1 - x0) * (y1 - y0)
    s = max(int(max(x1 - x0, y1 - y0) * MARGIN), 8)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    ox, oy = cx - s, cy - s
    crop = image.crop((ox, oy, cx + s, cy + s))
    near, names = parse_region(image, cx, cy, s, device)  # tight crop: the face features
    wide, _ = parse_region(image, cx, cy, round(s * WIDE), device)  # wide crop: hair and neck are not cut at the edge

    def ids(*group: str) -> list[int]:
        return [i for i, n in names.items() if n in group]

    def parsed(*group: str) -> np.ndarray:
        return np.isin(near, ids(*group))

    def seg(*group: str, px: int = 3) -> np.ndarray | None:
        return tidy(parsed(*group), px)

    def whole(*group: str) -> np.ndarray:
        """The parsed region from the tight crop plus whatever of it the wide crop shows beyond the crop edge."""
        tight = parsed(*group)
        both = tight | np.isin(wide, ids(*group))
        if full_hair is not None and group == ("hair",):  # long hair runs past the crops: allow it near the face
            reach = np.zeros_like(both)
            fw, fh = x1 - x0, y1 - y0
            reach[max(0, y0 - fh // 2):max(0, y1 + 3 * fh), max(0, x0 - fw):max(0, x1 + fw)] = True
            both = both | (full_hair & reach)
        n, comp = cv2.connectedComponents(both.astype(np.uint8), connectivity=8)
        touching = np.unique(comp[tight & (comp > 0)])
        return np.isin(comp, touching[touching > 0]) if len(touching) else tight

    out: list[tuple[np.ndarray, str]] = []
    skin = parsed(*SKIN)  # includes the nose: it is never drawn as its own shape
    for phrase, m in (("hair", whole("hair")), ("neck", whole(*NECK)), ("skin", skin)):
        m = tidy(m, min_px)
        if m is not None:
            out.append((m, phrase))

    glasses = parsed("eye_g")
    has_glasses = glasses.sum() > 0.01 * face_area
    sun = has_glasses and grey[glasses].mean() < SUNGLASSES
    parts = glasses_parts(rgb, glasses) if has_glasses else []  # from the parse alone; eyes and brows go over it

    # landmarks: only trusted if they sit on the face (a turned or occluded face gives garbage)
    found = fan().get_landmarks_from_image(np.asarray(crop.resize((SIDE, SIDE), Image.LANCZOS)),
                                           detected_faces=[[(v - o) * SIDE / (2 * s) for v, o in
                                                            ((x0, ox), (y0, oy), (x1, ox), (y1, oy))]])
    pts = found[0] * (2 * s) / SIDE + [ox, oy] if found else None  # crop -> image coordinates
    ok = False
    if pts is not None:
        key = [pts[36:42].mean(0), pts[42:48].mean(0), pts[48:68].mean(0), pts[30]]
        on_skin = sum(bool(skin[min(max(int(y), 0), h - 1), min(max(int(x), 0), w - 1)]) for x, y in key)
        spacing = np.linalg.norm(pts[36:42].mean(0) - pts[42:48].mean(0))
        centred = abs(pts[30][0] - (pts[36:42].mean(0)[0] + pts[42:48].mean(0)[0]) / 2) <= MAX_NOSE_OFFSET * spacing
        ok = on_skin >= 3 and spacing >= MIN_EYE_SPACING * (x1 - x0) and centred
    big = size >= FULL_FACE
    parse_ok = size >= SEG_FACE
    shape = (h, w)

    # mouth: the parse first (inside of the mouth, lips), landmarks for what it cannot give
    mouth = seg("mouth", px=12) if parse_ok else None
    lips = seg("u_lip", "l_lip", "mouth", px=12) if parse_ok else None
    if mouth is not None and mouth.sum() >= 0.004 * face_area:  # open mouth
        if style == "detailed" and lips is not None:
            parts.append((lips, "lips"))
        parts.append((mouth, "mouth"))
        teeth = teeth_from_mouth(mouth, rgb) if big else None
        if teeth is not None:
            parts.append((teeth, "teeth"))
    elif lips is not None and style == "cartoon":  # closed mouth: a line along the lips
        line = stroke(lips, 0.35, 2)
        if line is not None:
            parts.append((line, "mouth line"))
    elif ok and size >= SMALL_FACE:
        if style == "detailed":
            parts.append((fill(shape, pts[48:60]), "lips"))
        inner = fill(shape, pts[60:68])
        if style == "cartoon":
            parts += cartoon_mouth(pts, inner, size, rgb, shape)
        elif big and inner.sum() >= 12:
            parts += [(inner, "mouth"), (teeth_mask(inner, rgb), "teeth")]

    # eyebrows: thin strokes along the parsed brows, or along the landmarks on small faces
    thick = max(2, round(0.04 * (y1 - y0)))
    brows = [seg(n, px=8) for n in ("l_brow", "r_brow")] if parse_ok else [None, None]
    for i, mask in enumerate(brows):
        line = stroke(mask, 0.6, 2) if mask is not None else None
        if line is not None:
            parts.append((line, "eyebrow"))
        elif ok:
            b, e = ((pts[17:22], pts[36:42]), (pts[22:27], pts[42:48]))[i]
            b = b + [0, BROW_DROP * (e[:, 1].mean() - b[:, 1].mean())]  # FAN traces the brow's top edge
            parts.append((darkest(fill(shape, b, 2 * thick), grey, BROW_KEEP), "eyebrow"))

    # eyes: a dot at the parsed eye, an arc where the landmarks say it is closed
    if not sun and size >= SMALL_FACE:
        seg_eyes = [seg(n) for n in ("l_eye", "r_eye")] if parse_ok else []
        centres = [np.array(np.nonzero(e)[::-1]).mean(1) for e in seg_eyes if e is not None]
        if ok:
            gap = np.linalg.norm(pts[36:42].mean(0) - pts[42:48].mean(0))
            for eye in (pts[36:42], pts[42:48]):
                width = np.linalg.norm(eye[3] - eye[0])
                near = [c for c in centres if np.linalg.norm(c - eye.mean(0)) < 0.6 * width]
                closed = np.ptp(eye[:, 1]) < max(1.0, 0.12 * width)
                if style == "cartoon" and near and not closed:
                    parts.append((disc(shape, near[0], max(1.2, 0.09 * gap)), "eye"))
                elif style == "cartoon":
                    parts.append(cartoon_eye(grey, eye, gap, shape))
                elif big:
                    parts += eye_parts(rgb, grey, eye, eye[:4])
                else:
                    parts.append((fill(shape, eye[:4], max(1, round(0.14 * width))), "eyelid"))
        elif len(centres) == 2:  # landmarks rejected: dots at the parsed eyes
            gap = np.linalg.norm(centres[0] - centres[1])
            parts += [(disc(shape, c, max(1.2, 0.09 * gap)), "eye") for c in centres]

    # tidy fills holes: fine for solid parts, and glasses are stacked shapes without holes
    tidied = [(m if p in ("glasses", "sunglasses") and m.sum() >= min_px else tidy(m, min_px), p)
              for m, p in parts]
    return out + [(m, p) for m, p in tidied if m is not None]


def same_thing(old: np.ndarray, new: np.ndarray) -> bool:
    """An automatic shape that mostly is the new face region: heavy overlap, or lying mostly inside it. Left in place,
    such shapes show as patches of another colour on a hair shape, or as slivers round its edge."""
    inter = float((old & new).sum())
    return inter / float((old | new).sum()) > 0.5 or inter >= 0.7 * old.sum()


def insert_pos(items: list[tuple[np.ndarray, str | None]], m: np.ndarray) -> int:
    """Where ``m`` goes in painter's order: just above the last shape at least as big that covers a good part of
    it, so smaller shapes (details, and anything meant to be in front) stay on top. The order of ``items`` is not
    sorted by area, so searching for the first smaller shape is not enough."""
    area = m.sum()
    last = -1
    for i, (k, _) in enumerate(items):
        if k.sum() >= area and (k & m).sum() >= 0.25 * area:
            last = i
    return last + 1


def grow_hair(hair: np.ndarray, items: list[tuple[np.ndarray, str | None]], rgb: np.ndarray,
              avoid: np.ndarray) -> np.ndarray:
    """Extend a parsed hair mask along the trace's own shapes that touch it and have the same colour. Hair that runs
    past what the parser labels (long hair over a shoulder, read as clothing) then ends where the picture does,
    not at a crop edge. Shapes that overlap ``avoid`` (the face and neck) are never taken."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lum = lab[..., 0][hair]
    colour = lab[hair][lum <= np.quantile(lum, 0.7)].mean(0)
    grown = hair.copy()
    limit = 0.8 * hair.sum()
    for _ in range(2):  # chains of strands
        reach = cv2.dilate(grown.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        for k, phrase in items:
            if phrase is not None or k.sum() > limit or (k & grown).sum() >= k.sum() or not (k & reach).any():
                continue
            if (k & avoid).sum() > 0.05 * k.sum() or np.linalg.norm(lab[k].mean(0) - colour) > HAIR_COLOUR_GAP:
                continue
            grown |= k
    return grown


def add_details(items: list[tuple[np.ndarray, str | None]], image: Image.Image, device: str, style: str = "cartoon",
                log=print):
    """Add face detail to ``items`` (``(mask, phrase)`` pairs in painter's order, bottom first).

    Earlier face shapes are dropped first, so the stage can be rerun. Hair, neck and skin go in by area, like
    ``--care`` shapes (so smaller automatic shapes stay on top), skin directly above the hair; everything else goes
    on top. Returns the new items and a count per phrase.
    """
    items = [(m, p) for m, p in items if p not in PHRASES]
    full_hair = parse_full(image, device)
    boxes = [b for b in fan().face_detector.detect_from_image(np.asarray(image.convert("RGB")))
             if min(b[2] - b[0], b[3] - b[1]) >= MIN_FACE]
    log(f"  {len(boxes)} faces")
    counts: dict[str, int] = {}
    tops: list[tuple[np.ndarray, str | None]] = []
    for box in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        hair = None
        shapes = face_shapes(image, box, device, style, full_hair=full_hair)
        face_parts = [m for m, q in shapes if q in ("skin", "neck")]
        if face_parts and any(q == "hair" for _, q in shapes):
            avoid = np.logical_or.reduce(face_parts)
            shapes = [(grow_hair(m, items, np.asarray(image.convert("RGB")), avoid) if q == "hair" else m, q)
                      for m, q in shapes]
        for m, phrase in shapes:
            counts[phrase] = counts.get(phrase, 0) + 1
            if phrase in ("hair", "neck", "skin"):
                # the trace's own shape for the same thing would peek out as slivers: replace it
                items = [(k, q) for k, q in items if q is not None or not same_thing(k, m)]
                pos = insert_pos(items, m)
                if phrase == "hair":
                    hair = m
                elif phrase == "skin" and hair is not None:  # never below its own hair
                    pos = max(pos, 1 + next(i for i, (k, _) in enumerate(items) if k is hair))
                items.insert(pos, (m, phrase))
            else:
                tops.append((m, phrase))
    log("  " + ", ".join(f"{n} {p}" for p, n in counts.items()))
    return items + tops, counts
