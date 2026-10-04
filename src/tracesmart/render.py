"""Render stacked masks to SVG (and PNG), plus the segment map and the HTML index of runs."""
import colorsys
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import layers as lay
from .trace import hex_colour, mask_path, smooth_mask

SVG_NS = 'xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"'


def shape_name(index: int, phrase: str | None) -> tuple[str, str]:
    """``(id, label)`` for a shape. Described shapes are named after their phrase, e.g. ``window-12``."""
    if not phrase:
        return f"shape-{index}", f"shape {index}"
    slug = re.sub(r"[^a-z0-9]+", "-", phrase.lower().replace("'", "")).strip("-") or "described"
    return f"{slug}-{index}", f"{phrase} {index}"


PRIOR_COLOURS = {  # described face parts: (colour the part must look like, how far to pull the sampled mean towards it)
    "teeth": ((240, 236, 228), 0.55),
    "eye white": ((236, 232, 226), 0.5),
    "iris": ((40, 28, 20), 0.3),
    "pupil": ((10, 8, 8), 0.75),
    "eyelid": ((25, 18, 16), 0.6),
    "eyebrow": ((30, 22, 18), 0.25),
    "mouth": ((120, 30, 35), 0.7),
    "mouth line": ((60, 25, 25), 0.6),
    "eye": ((15, 12, 12), 0.85),
}


def shape_colour(phrase: str | None, pixels: np.ndarray) -> np.ndarray:
    """A shape's fill from its visible ``pixels`` (N x 3): their mean, nudged towards the known colour of a face part
    (teeth are white, pupils dark) so a flat mean over a few mixed pixels does not turn them grey. Clear glasses
    take the colour of the pixels furthest from the usual colour inside the shape, which is the frame."""
    mean = pixels.mean(0)
    if phrase == "glasses" and len(pixels) >= 20:
        far = np.linalg.norm(pixels - np.median(pixels, 0), axis=1)
        return pixels[far >= np.quantile(far, 0.65)].mean(0)
    if phrase in PRIOR_COLOURS:
        target, k = PRIOR_COLOURS[phrase]
        return (1 - k) * mean + k * np.asarray(target, float)
    return mean


def uncovered_regions(masks: list[np.ndarray], min_px: int) -> list[np.ndarray]:
    """Connected areas that no mask covers, big enough to matter. They become the bottom layer, so the
    picture is never left showing a flat backdrop colour where, say, a gravel path should be."""
    covered = np.zeros(masks[0].shape, bool)
    for m in masks:
        covered |= m
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~covered).astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]


def close_seams(masks: list[np.ndarray], max_px: float) -> list[np.ndarray]:
    """Grow every shape a little so neighbours overlap instead of leaving a thin sliver of the layer below.

    Masks from different SAM prompts do not tile perfectly, and smoothing shrinks each one a bit more, so two
    neighbours (sky and mountain) can end up 1 to 3 px apart with a darker shape showing through. A neighbour that
    is painted later covers the sliver. Big shapes grow by up to ``max_px``; small ones barely at all, so a window
    keeps its size.
    """
    out = []
    for m in masks:
        r = int(round(min(max_px, 0.04 * np.sqrt(m.sum()))))
        if r >= 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            m = cv2.dilate(m.astype(np.uint8), kernel).astype(bool)
        out.append(m)
    return out


LEVEL_NAMES = ("1 Structure", "2 Objects", "3 Details")


def compose(mode: str, rgb: np.ndarray, masks: list[np.ndarray], names: list[tuple[str, str]],
            elements: dict[int, str], attach: dict[int, int] | None = None) -> str:
    """Join the shape elements in stack order, optionally organised into groups or layers.

    ``objects`` nests parts inside the shape that contains them; ``levels`` puts shapes on Inkscape layers from
    coarse to fine, ``depth`` from back to front with no overlaps inside a layer. All keep every noticeably
    overlapping pair in its original order, so the picture does not change.
    """
    if mode == "none":
        return "".join(elements[i] for i in sorted(elements))
    areas = [int(m.sum()) for m in masks]
    inter = lay.intersections(masks)
    pairs = lay.significant(inter, areas)
    if mode == "levels":
        level = lay.levels(rgb, masks, pairs)
        out = []
        for lv, name in enumerate(LEVEL_NAMES):
            inner = "".join(elements[i] for i in sorted(elements) if level[i] == lv)
            count = sum(1 for i in elements if level[i] == lv)
            out.append(f'<g id="layer-{lv + 1}" inkscape:groupmode="layer" '
                       f'inkscape:label="{name} ({count})">{inner}</g>')
        return "".join(out)
    if mode == "depth":
        depth = lay.depths(len(masks), pairs)
        out = []
        for d in range(max(depth, default=-1) + 1):
            members = [i for i in sorted(elements) if depth[i] == d]
            where = " (back)" if d == 0 else " (front)" if d == max(depth) else ""
            out.append(f'<g id="depth-{d + 1}" inkscape:groupmode="layer" '
                       f'inkscape:label="Depth {d + 1}{where}, {len(members)} shapes">'
                       f'{"".join(elements[i] for i in members)}</g>')
        return "".join(out)
    if mode != "objects":
        raise ValueError(f"layers must be none, objects, levels or depth, not {mode!r}")

    def emit(node: lay.Node) -> str:
        own = elements.get(node.index, "")
        if not node.children:
            return own
        sid, label = names[node.index]
        return (f'<g id="{sid}-group" inkscape:label="{label} (group)">{own}'
                f'{"".join(emit(c) for c in node.children)}</g>')

    return "".join(emit(n) for n in lay.tree(lay.hierarchy(masks, inter, pairs, attach=attach)))


def render(image: Image.Image, masks: list[np.ndarray], phrases: list[str | None] | None = None,
           round_px: float | None = None, tol: float | None = None, seam_px: float | None = None,
           layers: str = "none", group: str = "person", complete: bool = True):
    """Stack ``masks`` (bottom first) into an SVG.

    Returns ``(svg, preview image, segment-map svg)``. Each shape is filled with the mean colour of its
    visible pixels. The segment map paints every shape in its own arbitrary colour so segmentation problems
    can be told apart from colour problems; described shapes are outlined in magenta.
    """
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    phrases = phrases or [None] * len(masks)
    gaps = uncovered_regions(masks, max(20, int(0.0004 * h * w))) if masks else []
    masks, phrases = gaps + list(masks), [None] * len(gaps) + list(phrases)  # uncovered areas sit at the bottom
    attach: dict[int, int] = {}
    if layers == "objects":
        # shapes that belong to an object (a hiker's shirt, boots and backpack) are grouped with it, and the shapes
        # below it are extended under its silhouette so that lifting the group leaves no hole
        attach = lay.object_groups(masks, phrases, {g.strip() for g in group.split(",") if g.strip()},
                                   first=len(gaps))
        if complete and attach:
            masks = lay.complete_under(masks, attach, len(gaps))
    round_px = 0.0018 * max(h, w) if round_px is None else round_px
    tol = tol or max(1.0, 0.002 * max(h, w))
    if round_px > 0:
        masks = [smooth_mask(m, min(round_px, 0.05 * np.sqrt(m.sum()))) if m.sum() > 100 else m for m in masks]
    seam_px = 0.0016 * max(h, w) if seam_px is None else seam_px
    if seam_px > 0:
        masks = close_seams(masks, seam_px)

    above = np.zeros((h, w), bool)  # visible part of a shape = itself minus everything painted after it
    colours = [None] * len(masks)
    for i in range(len(masks) - 1, -1, -1):
        visible = masks[i] & ~above
        colours[i] = shape_colour(phrases[i], rgb[visible if visible.sum() >= 5 else masks[i]])
        above |= masks[i]
    backdrop = rgb.reshape(-1, 3).mean(0)
    preview = np.zeros_like(rgb)
    preview[:] = backdrop.astype(np.uint8)

    paths, segments = {}, {}
    names = [shape_name(i, p) for i, p in enumerate(phrases)]
    for i, (m, colour, phrase) in enumerate(zip(masks, colours, phrases, strict=True)):
        preview[m] = colour.astype(np.uint8)
        d = mask_path(m, tol)
        if not d:
            continue
        sid, label = names[i]
        paths[i] = f'<path id="{sid}" inkscape:label="{label}" fill="{hex_colour(colour)}" d="{d}"/>'
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.55 + 0.35 * ((i * 7) % 3) / 2,
                                      0.95 - 0.25 * ((i * 5) % 3) / 2)  # golden-angle hues: neighbours differ
        edge = 'stroke="#ff00ff" stroke-width="2.5"' if phrase else 'stroke="#ffffff" stroke-width="0.8"'
        segments[i] = (f'<path id="{sid}" inkscape:label="{label}" fill="{hex_colour((r * 255, g * 255, b * 255))}" '
                       f'{edge} stroke-linejoin="round" d="{d}"/>')
    head = f'<svg {SVG_NS} width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    body = compose(layers, rgb, masks, names, paths, attach)
    seg_body = compose(layers, rgb, masks, names, segments, attach)
    svg = f'{head}<rect id="backdrop" width="{w}" height="{h}" fill="{hex_colour(backdrop)}"/>{body}</svg>\n'
    seg_svg = f'{head}<rect width="{w}" height="{h}" fill="#222222"/>{seg_body}</svg>\n'
    return svg, Image.fromarray(preview), seg_svg


def write_outputs(image: Image.Image, masks: list[np.ndarray], phrases: list[str | None], out: Path,
                  round_px: float | None = None, zoom: float = 3.0, seam_px: float | None = None,
                  layers: str = "none", group: str = "person", complete: bool = True) -> None:
    """Write the SVG, a hi-res PNG, the segment map (SVG + PNG) and a source | segments | vector comparison."""
    import resvg_py

    svg, preview, seg_svg = render(image, masks, phrases, round_px=round_px, seam_px=seam_px, layers=layers,
                                   group=group, complete=complete)
    (out / "vector.svg").write_text(svg)
    (out / "segments.svg").write_text(seg_svg)
    preview.save(out / "preview.png")
    pngs = {}
    for name, text in (("vector", svg), ("segments", seg_svg)):
        (out / f"{name}.png").write_bytes(bytes(resvg_py.svg_to_bytes(svg_string=text, zoom=zoom)))
        pngs[name] = Image.open(out / f"{name}.png").convert("RGB")
    source = image.resize(pngs["vector"].size, Image.LANCZOS)
    w, h = source.size
    sheet = Image.new("RGB", (w * 3 + 40, h), "#222222")
    for i, im in enumerate((source, pngs["segments"], pngs["vector"])):
        sheet.paste(im, (i * (w + 20), 0))
    sheet.save(out / "compare.png")


def build_index(folder: Path) -> int:
    """Write ``folder/index.html`` showing every run (``folder/<image>/<variant>/``) with its care words."""
    runs = sorted(p.parent for p in folder.glob("*/*/compare.png"))
    cards = []
    for run in runs:
        rel = run.relative_to(folder)
        files = ("vector.svg", "vector.png", "segments.svg", "segments.png", "compare.png")
        links = " · ".join(f'<a href="{rel}/{f}">{f}</a>' for f in files)
        meta = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
        chips = "".join(
            f'<span class="chip{" zero" if c["found"] == 0 else ""}">{c["phrase"]} '
            f'<b>{c["kept"]}</b>{"" if c["kept"] == c["found"] else "/" + str(c["found"])}</span>'
            for c in meta.get("care") or [])
        note = (f'<p class="l">described with SAM 3 (kept/found): {chips}</p>' if chips
                else '<p class="l">automatic only: no description</p>')
        cards.append(f'<section><h2>{rel.parent} / {rel.name}</h2>{note}<p class="l">{links}</p>'
                     f'<a href="{rel}/compare.png"><img src="{rel}/compare.png" loading="lazy"></a>'
                     '<p class="l">left: source · middle: segments (magenta outline = described) · '
                     'right: vector</p></section>')
    style = ("body{font:15px system-ui;margin:24px;background:#1b1b1b;color:#ddd}h2{margin:28px 0 4px}"
             "img{width:100%;border-radius:6px}.chip{display:inline-block;background:#333;border-radius:12px;"
             "padding:1px 9px;margin:2px 3px 2px 0;color:#eee}.chip b{color:#ff7bff}"
             ".zero{background:#4a2a2a;color:#e99}"
             ".l{color:#999;margin:2px 0 8px}a{color:#8cb4ff}")
    (folder / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>tracesmart outputs</title><style>{style}</style>"
        f"<h1>tracesmart outputs</h1>{''.join(cards)}")
    return len(runs)
