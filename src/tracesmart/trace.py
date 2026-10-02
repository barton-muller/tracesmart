"""Masks to paths: corner-preserving Bezier tracing and mask smoothing."""
import cv2
import numpy as np


def mask_path(mask: np.ndarray, tol: float = 1.5, corner_deg: float = 35.0) -> str | None:
    """Trace the outer contour of ``mask`` as one closed SVG path.

    The contour is simplified to within ``tol`` pixels. Vertices where the outline turns by more than
    ``corner_deg`` stay sharp corners (straight lines between two corners, so windows stay rectangular);
    everywhere else the outline is smoothed with cubic Beziers.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, tol, True)[:, 0, :].astype(float)
    n = len(c)
    if n < 3:
        return None
    prev = c - np.roll(c, 1, 0)
    nxt = np.roll(c, -1, 0) - c
    turn = np.degrees(np.abs(np.arctan2(prev[:, 0] * nxt[:, 1] - prev[:, 1] * nxt[:, 0], (prev * nxt).sum(1))))
    corner = turn > corner_deg
    tangent = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    d = f"M{c[0][0]:.1f},{c[0][1]:.1f}"
    for i in range(n):
        j = (i + 1) % n
        p1, p2 = c[i], c[j]
        if corner[i] and corner[j]:
            if j != 0:  # the closing straight edge is drawn by Z
                d += f"L{p2[0]:.1f},{p2[1]:.1f}"
            continue
        length = np.linalg.norm(p2 - p1)
        c1 = p1 + ((p2 - p1) / 3 if corner[i] else tangent[i] * length / 3)
        c2 = p2 - ((p2 - p1) / 3 if corner[j] else tangent[j] * length / 3)
        d += f"C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"
    return d + "Z"


def smooth_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Round off a mask by blurring it and re-thresholding (on its bounding box only)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    pad = int(sigma * 3) + 2
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = cv2.GaussianBlur(mask[y0:y1, x0:x1].astype(np.float32), (0, 0), sigma) > 0.5
    return out


def hex_colour(rgb) -> str:
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"
