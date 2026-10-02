"""The segmentation pipeline: SAM masks -> filter by impact -> fill the gaps -> optional text descriptions.

Follows SAMVG (Zhu et al., ICASSP 2024, arXiv:2311.05276) without its differentiable-rendering optimisation:

1. SAM 2.1 automatic masks (point grid + zoomed crops).
2. "Filter by impact": paint masks large to small in their mean colour and keep a mask only if it lowers the
   image error by at least ``impact``.
3. Find big badly covered regions (distance transform), prompt SAM at their centres, filter again.
4. Optionally describe what matters in words: SAM 3 finds every instance of each phrase and those masks are
   always kept, replacing near-duplicate automatic shapes.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from . import backends

MAX_ERR = 3 * 255.0**2


def tidy(mask: np.ndarray, min_px: int) -> np.ndarray | None:
    """Fill holes (upper layers cover them anyway) and keep the largest connected piece, if big enough."""
    mask = ndimage.binary_fill_holes(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest) if stats[biggest, cv2.CC_STAT_AREA] >= min_px else None


class Canvas:
    """A painter's-algorithm canvas that tracks the per-pixel error against the photo."""

    def __init__(self, rgb: np.ndarray):
        self.image = rgb.astype(np.float32)
        self.h, self.w = rgb.shape[:2]
        self.error = np.full((self.h, self.w), MAX_ERR, np.float32)  # uncovered pixels have the maximum error
        self.kept: list[np.ndarray] = []

    def try_add(self, mask: np.ndarray, impact: float) -> bool:
        """Paint ``mask`` in its mean colour if that lowers the normalised image error by >= ``impact``."""
        px = self.image[mask]
        new = ((px - px.mean(0)) ** 2).sum(1)
        gain = (self.error[mask].sum() - new.sum()) / (self.h * self.w * MAX_ERR)
        if gain < impact:
            return False
        self.error[mask] = new
        self.kept.append(mask)
        return True


def prompt_points(canvas: Canvas, radius: float, err_thresh: float, limit: int = 60) -> list[tuple[int, int]]:
    """Centres of big badly covered regions: pixels at least ``radius`` away from any well covered pixel."""
    dist = cv2.distanceTransform((canvas.error > err_thresh).astype(np.uint8), cv2.DIST_L2, 5)
    points = []
    while len(points) < limit:
        y, x = np.unravel_index(int(dist.argmax()), dist.shape)
        d = dist[y, x]
        if d < radius:
            break
        points.append((int(x), int(y)))
        cv2.circle(dist, (int(x), int(y)), int(d * 1.5) + 1, 0, -1)
    return points


def describe(image: Image.Image, items: list[tuple[np.ndarray, str | None]], phrases: list[str], threshold: float,
             min_px: int, device: str, log=print):
    """Add SAM 3 text-prompted masks to ``items`` (``(mask, phrase)`` pairs in painter's order, bottom first).

    Described masks are always kept. One that duplicates another phrase's mask (IoU > 0.85) is skipped.
    It replaces automatic shapes that are essentially the same object and is inserted in area order, so
    smaller automatic details inside it stay on top. Returns the new items and a per-phrase report.
    """
    found, report = [], []
    for phrase in phrases:
        try:
            masks = backends.sam3_text(image, phrase, device=device, threshold=threshold)
        except Exception as e:  # not every op is implemented on every device: retry on the CPU
            log(f"  sam3 on {device} failed ({type(e).__name__}); retrying on cpu")
            masks = backends.sam3_text(image, phrase, device="cpu", threshold=threshold)
        masks = [m for m in (tidy(m, min_px) for m in masks) if m is not None]
        log(f"  '{phrase}': {len(masks)} instances")
        report.append({"phrase": phrase, "found": len(masks), "kept": 0})
        found += [(len(report) - 1, m) for m in masks]
    accepted: list[np.ndarray] = []
    for ri, m in sorted(found, key=lambda t: -t[1].sum()):
        area = m.sum()
        if any((m & a).sum() / (m | a).sum() > 0.85 for a in accepted):
            continue
        accepted.append(m)
        report[ri]["kept"] += 1
        items = [(k, p) for k, p in items
                 if p is not None or not (k.sum() >= 0.4 * area and (k & m).sum() >= 0.8 * k.sum())]
        pos = next((i for i, (k, _) in enumerate(items) if k.sum() < area), len(items))
        items.insert(pos, (m, phrases[ri]))
    return items, report


def vectorise(image: Image.Image, *, grid: int = 32, crops: bool = True, impact: float = 2e-4, rounds: int = 2,
              min_area: float = 0.0004, err_thresh: float = 3 * 30.0**2, radius: float = 0.02,
              care: list[str] | None = None, care_threshold: float = 0.5, care_min_area: float = 0.00002,
              device: str | None = None, cache: Path | None = None, log=print):
    """Segment ``image`` into stacked masks.

    Returns ``(masks, phrases, report)``: masks in painter's order (bottom first), the phrase each one was
    described with (``None`` for automatic shapes) and a per-phrase ``{phrase, found, kept}`` report.
    ``cache`` is an .npz path for the slow automatic stage, reused on later runs.
    """
    device = device or backends.best_device()
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    min_px = max(20, int(min_area * h * w))
    canvas = Canvas(rgb)

    if cache is not None and cache.exists():
        log("stage 1: loaded cached automatic masks")
        masks = list(np.load(cache)["masks"])
    else:
        log("stage 1: SAM 2.1 automatic masks")
        masks = [m for m in (tidy(m, min_px) for m in backends.auto_masks(image, device=device, grid=grid, crops=crops))
                 if m is not None]
        if cache is not None:
            np.savez_compressed(cache, masks=np.stack(masks))
    log(f"  {len(masks)} raw masks")

    log("stage 2: filter by impact")
    for m in sorted(masks, key=lambda m: -m.sum()):
        canvas.try_add(m, impact)
    log(f"  kept {len(canvas.kept)}")

    prompter = None
    for r in range(rounds):
        points = prompt_points(canvas, radius * max(h, w), err_thresh)
        log(f"stage 3, round {r + 1}: {len(points)} badly covered regions")
        if not points:
            break
        prompter = prompter or backends.Prompter(image, device=device)
        new = [m for m in (tidy(m, min_px) for m in prompter.masks(points)) if m is not None]
        added = sum(canvas.try_add(m, impact) for m in sorted(new, key=lambda m: -m.sum()))
        log(f"  added {added}")

    items: list[tuple[np.ndarray, str | None]] = [(m, None) for m in canvas.kept]
    report: list[dict] = []
    if care:
        log(f"stage 4: SAM 3 descriptions ({len(care)})")
        items, report = describe(image, items, care, care_threshold, max(6, int(care_min_area * h * w)), device, log)
    return [m for m, _ in items], [p for _, p in items], report
