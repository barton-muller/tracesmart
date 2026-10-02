"""Optional colour splitting: break a shape with strong internal colour contrast into flat key-colour patches."""
import cv2
import numpy as np
from PIL import Image

from .pipeline import tidy


def _lab(rgb: np.ndarray) -> np.ndarray:
    """Float Lab: L in 0-100, a/b centred on zero."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    return lab


def split_tones(lab: np.ndarray, mask: np.ndarray, tol: float, kmax: int, sigma: float):
    """Cluster the pixels of ``mask`` in Lab space until the rms colour error is <= ``tol`` (or ``kmax`` tones).

    Returns an int label map (-1 outside the mask) and the number of tones. The tone map is smoothed
    spatially by ``sigma`` pixels so patches are blobs, not speckle.
    """
    ys, xs = np.nonzero(mask)
    data = lab[ys, xs].astype(np.float32)
    k, labels, centers = 1, np.zeros(len(data), np.int32), data.mean(0, keepdims=True)
    for k in range(1, kmax + 1):
        if k > 1:
            if len(data) < 50 * k:
                k -= 1
                break
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
            _, lb, centers = cv2.kmeans(data, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
            labels = lb.ravel().astype(np.int32)
        if np.sqrt(((data - centers[labels]) ** 2).sum(1).mean()) <= tol:
            break
    out = np.full(mask.shape, -1, np.int32)
    out[ys, xs] = labels
    if k > 1:
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        sub, inside = out[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        score = np.stack([cv2.GaussianBlur(((sub == j) & inside).astype(np.float32), (0, 0), sigma) for j in range(k)])
        smoothed = score.argmax(0).astype(np.int32)
        smoothed[~inside] = -1
        out[y0:y1, x0:x1] = smoothed
    return out, k


def tone_patches(image: Image.Image, masks: list[np.ndarray], tone_rms: float, min_area: float = 0.002,
                 kmax: int = 3, tol: float = 14.0, sigma: float = 2.0, log=print) -> list[np.ndarray]:
    """Extra masks to paint on top of ``masks``: shapes whose visible pixels deviate from their mean colour
    by more than ``tone_rms`` (rms RGB distance) are split into up to ``kmax`` tones; the biggest tone stays
    as the base shape and the rest are returned as new patches."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    flat = cv2.pyrMeanShiftFiltering(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 6, 14)
    lab = _lab(cv2.cvtColor(flat, cv2.COLOR_BGR2RGB))
    min_px = int(min_area * h * w)
    above = np.zeros((h, w), bool)
    extra: list[np.ndarray] = []
    for i in range(len(masks) - 1, -1, -1):  # topmost shape first; its visible part is what the viewer sees
        visible = masks[i] & ~above
        above |= masks[i]
        if visible.sum() < 4 * min_px:
            continue
        px = rgb[visible].astype(np.float32)
        rms = float(np.sqrt(((px - px.mean(0)) ** 2).sum(1).mean()))
        if rms < tone_rms:
            continue
        labels, k = split_tones(lab, visible, tol, kmax, sigma)
        order = sorted(range(k), key=lambda j: -int((labels == j).sum()))
        for j in order[1:]:
            patch = tidy(labels == j, min_px)
            if patch is not None:
                extra.append(patch)
        log(f"  shape {i}: rms {rms:.0f} -> {k} tones")
    return extra
