"""Group the stacked shapes into layers without changing how the picture looks.

The flat output is a painter's stack: shapes are drawn bottom first. Two ways of organising it for editing:

* ``objects``: nest parts inside the thing they belong to. A hat that lies inside a person becomes a child of that
  person's group, so you can move or recolour the whole person.
* ``depth``: Inkscape layers from back to front. Shapes in one layer never overlap each other, so each layer is a
  clean cut-out; this is the layering used by "Layered Image Vectorization via Semantic Simplification" (Wang et al.).
* ``levels``: Inkscape layers from coarse structure to fine detail, so detail can be hidden. A shape's level comes
  from its *impact*, how much it reduces the error against the photo when painted (the measure SAMVG uses to decide
  which masks matter).

Reordering shapes is only safe where it cannot change a pixel. Every plan is checked against the pairs of shapes that
overlap noticeably: for each such pair the one painted later in the flat stack must still be drawn later. Shapes
that would break this are moved back to where the flat stack had them.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

MAX_ERR = 3 * 255.0**2


@dataclass
class Node:
    """A shape (``index``) or a group, with children drawn after the shape itself."""

    index: int
    children: list["Node"] = field(default_factory=list)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """``(y0, y1, x0, x1)`` of the mask, end-exclusive; all zeros for an empty mask."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if len(rows) == 0:
        return 0, 0, 0, 0
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def intersections(masks: list[np.ndarray]) -> dict[tuple[int, int], int]:
    """Pixel overlap of every pair of masks whose bounding boxes touch, as ``{(i, j): pixels}`` with ``i < j``."""
    boxes = [bbox(m) for m in masks]
    out: dict[tuple[int, int], int] = {}
    for i in range(len(masks)):
        y0, y1, x0, x1 = boxes[i]
        for j in range(i + 1, len(masks)):
            a, b, c, d = boxes[j]
            iy0, iy1, ix0, ix1 = max(y0, a), min(y1, b), max(x0, c), min(x1, d)
            if iy0 >= iy1 or ix0 >= ix1:
                continue
            n = int(np.count_nonzero(masks[i][iy0:iy1, ix0:ix1] & masks[j][iy0:iy1, ix0:ix1]))
            if n:
                out[(i, j)] = n
    return out


def significant(inter: dict[tuple[int, int], int], areas: list[int], min_px: int = 60,
                min_frac: float = 0.12) -> list[tuple[int, int]]:
    """Pairs whose overlap is big enough that swapping their order would visibly change the picture."""
    return [(i, j) for (i, j), n in inter.items() if n >= max(min_px, min_frac * min(areas[i], areas[j]))]


def impacts(rgb: np.ndarray, masks: list[np.ndarray]) -> list[float]:
    """How much each shape reduces the error against the photo when painted in stack order (SAMVG's measure)."""
    h, w = rgb.shape[:2]
    img = rgb.astype(np.float32)
    err = np.full((h, w), MAX_ERR, np.float32)
    out = []
    for m in masks:
        px = img[m]
        if len(px) == 0:
            out.append(0.0)
            continue
        new = ((px - px.mean(0)) ** 2).sum(1)
        out.append(float((err[m].sum() - new.sum()) / (h * w * MAX_ERR)))
        err[m] = new
    return out


def levels(rgb: np.ndarray, masks: list[np.ndarray], pairs: list[tuple[int, int]],
           bounds: tuple[float, float] = (3e-3, 3e-4)) -> list[int]:
    """Level 0 (structure), 1 (objects) or 2 (details) for every shape, by impact.

    A shape painted earlier than an overlapping shape cannot sit in a later layer than it, so levels are pulled
    down where needed to keep the picture identical.
    """
    gain = impacts(rgb, masks)
    level = [0 if g >= bounds[0] else 1 if g >= bounds[1] else 2 for g in gain]
    later: dict[int, list[int]] = {}
    for i, j in pairs:
        later.setdefault(i, []).append(j)
    for i in range(len(masks) - 1, -1, -1):  # an earlier shape may not be in a deeper level than any later overlap
        for j in later.get(i, ()):
            level[i] = min(level[i], level[j])
    return level


def depths(n: int, pairs: list[tuple[int, int]]) -> list[int]:
    """Depth layer of every shape: 0 if nothing noticeable lies under it, otherwise one more than the deepest
    shape it overlaps that is painted earlier. Shapes in one layer never overlap each other, and every shape sits
    in a layer in front of the shapes it covers (back to front, like the layering in Wang et al., 2025).
    """
    earlier: dict[int, list[int]] = {}
    for i, j in pairs:
        earlier.setdefault(j, []).append(i)
    depth = [0] * n
    for j in range(n):
        depth[j] = 1 + max((depth[i] for i in earlier.get(j, ())), default=-1)
    return depth


def convex_hull(mask: np.ndarray) -> np.ndarray:
    """Filled convex hull of a mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    if contours:
        cv2.fillConvexPoly(out, cv2.convexHull(np.vstack(contours)), 1)
    return out.astype(bool)


def object_groups(masks: list[np.ndarray], phrases: list[str | None], anchors: set[str], inside: float = 0.5,
                  contact: float = 0.3, max_ratio: float = 0.6, first: int = 0, hull: float = 0.7,
                  hull_ratio: float = 0.1) -> dict[int, int]:
    """Which shapes belong to which object, as ``{shape: anchor}``.

    Anchors are the described shapes whose phrase is in ``anchors`` (for example ``person``). Another shape joins an
    anchor if at least ``inside`` of it lies in the anchor's mask (a shirt on a person), or if at least ``contact`` of
    its outline borders the anchor (a backpack, which SAM does not count as part of the person), or, for small thin
    things such as a hiking pole, at least ``hull`` of it lies in the anchor's convex hull and it is at most
    ``hull_ratio`` of the anchor's area; and it is clearly
    smaller than the anchor (at most ``max_ratio`` of its area), so big background shapes are never taken. Shapes
    before index ``first`` (the generated fill for uncovered areas) are ignored.
    """
    names = {a.lower() for a in anchors}
    heads = [i for i, p in enumerate(phrases) if i >= first and p and p.lower() in names]
    if not heads:
        return {}
    areas = [int(m.sum()) for m in masks]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hulls = {a: convex_hull(masks[a]) for a in heads} if hull > 0 else {}
    attach: dict[int, int] = {}
    for i, m in enumerate(masks):
        if i < first or i in heads or areas[i] < 100:
            continue
        ring = cv2.dilate(m.astype(np.uint8), kernel).astype(bool) & ~m
        ring_px = int(ring.sum())
        best, best_score = -1, 0.0
        for a in heads:
            if areas[i] > max_ratio * areas[a]:
                continue
            share = np.count_nonzero(m & masks[a]) / areas[i]
            touch = np.count_nonzero(ring & masks[a]) / ring_px if ring_px else 0.0
            score = max(share if share >= inside else 0.0, touch if touch >= contact else 0.0)
            if score == 0.0 and a in hulls and areas[i] <= hull_ratio * areas[a]:
                in_hull = np.count_nonzero(m & hulls[a]) / areas[i]
                score = in_hull if in_hull >= hull else 0.0
            if score > best_score:
                best, best_score = a, score
        if best >= 0:
            attach[i] = best
    return attach


def complete_under(masks: list[np.ndarray], attach: dict[int, int], n_gaps: int, grow: int = 2) -> list[np.ndarray]:
    """Extend the shapes below each object so that lifting the object leaves no hole.

    Where an object stands, the shapes under it (the ground, say) are missing: SAM masks stop at the object's
    outline. Every pixel under an object's silhouette that no lower shape covers is given to the nearest lower shape.
    That area is hidden by the object, so the picture does not change. The first ``n_gaps`` shapes are the generated
    fill for uncovered areas and are not used as owners.
    """
    members: dict[int, list[int]] = {}
    for j, a in attach.items():
        members.setdefault(a, []).append(j)
    grouped = set(attach) | set(members)
    out = list(masks)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    for anchor, js in members.items():
        group = [anchor, *js]
        silhouette = np.zeros_like(masks[0])
        for g in group:
            silhouette |= masks[g]
        silhouette = cv2.dilate(silhouette.astype(np.uint8), kernel).astype(bool)
        owner = np.full(masks[0].shape, -1, np.int32)
        for i in range(n_gaps, min(group)):
            if i not in grouped:
                owner[out[i]] = i  # the topmost lower shape at each pixel
        holes = silhouette & (owner < 0)
        if not holes.any() or (owner >= 0).sum() == 0:
            continue
        nearest = ndimage.distance_transform_edt(owner < 0, return_distances=False, return_indices=True)
        given = owner[nearest[0], nearest[1]]
        for o in np.unique(given[holes]):
            out[o] = out[o] | (holes & (given == o))
    return out


def hierarchy(masks: list[np.ndarray], inter: dict[tuple[int, int], int], pairs: list[tuple[int, int]],
              contained: float = 0.85, bigger: float = 1.5, attach: dict[int, int] | None = None) -> list[int]:
    """Parent of every shape (-1 for none): the smallest earlier shape that contains it.

    ``contained`` is the share of the child inside the parent, ``bigger`` how much larger the parent must be.
    Nesting puts a child right after its parent, so any overlapping neighbour that the flat stack painted in
    between would change order; such children are moved back out to the top level until the order is safe.
    """
    areas = [int(m.sum()) for m in masks]
    parent = [-1] * len(masks)
    for (i, j), n in inter.items():  # i is painted before j
        inside = areas[j] and n / areas[j] >= contained and areas[i] >= bigger * areas[j]
        if inside and (parent[j] == -1 or areas[i] < areas[parent[j]]):
            parent[j] = i
    for j, a in (attach or {}).items():  # object membership (a backpack on its hiker) beats plain containment
        parent[j] = a
    for _ in range(len(masks)):
        order = flatten(tree(parent))
        pos = {k: p for p, k in enumerate(order)}
        bad = [(i, j) for i, j in pairs if pos[i] > pos[j]]
        if not bad:
            return parent
        i, j = bad[0]  # i was painted first but now comes after j: free whichever of the two nesting moved
        victim = j if parent[j] != -1 else i
        if parent[victim] == -1:
            break
        parent[victim] = -1
    return [-1] * len(masks)


def tree(parent: list[int]) -> list[Node]:
    """Top-level nodes in stack order; children follow their parent in stack order."""
    nodes = [Node(i) for i in range(len(parent))]
    roots = []
    for i, p in enumerate(parent):
        (roots if p == -1 else nodes[p].children).append(nodes[i])
    return roots


def flatten(roots: list[Node]) -> list[int]:
    out: list[int] = []
    for n in roots:
        out.append(n.index)
        out += flatten(n.children)
    return out
