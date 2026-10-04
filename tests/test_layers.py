import io

import numpy as np
import resvg_py
from PIL import Image

from tracesmart import layers as lay
from tracesmart.render import render


def scene():
    """A photo-like image and stacked masks: sky, person, hat inside the person, a window crossing the person."""
    h, w = 120, 160
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (110, 160, 220)
    img[40:110, 50:100] = (150, 60, 50)  # person
    img[40:55, 60:90] = (30, 30, 30)  # hat
    img[20:70, 90:140] = (230, 220, 150)  # window, partly over the person
    def box(y0, y1, x0, x1):
        m = np.zeros((h, w), bool)
        m[y0:y1, x0:x1] = True
        return m
    masks = [box(0, h, 0, w), box(40, 110, 50, 100), box(40, 55, 60, 90), box(20, 70, 90, 140)]
    return Image.fromarray(img), masks, [None, "person", "hat", "window"]


def pixels(svg: str) -> np.ndarray:
    png = resvg_py.svg_to_bytes(svg_string=svg)
    return np.asarray(Image.open(io.BytesIO(bytes(png))).convert("RGB")).astype(int)


def test_objects_nests_the_hat_in_the_person_without_changing_the_picture():
    img, masks, phrases = scene()
    flat, _, _ = render(img, masks, phrases, round_px=0, seam_px=0)
    nested, _, _ = render(img, masks, phrases, round_px=0, seam_px=0, layers="objects")
    assert '<g id="person-1-group"' in nested
    assert nested.index('id="hat-2"') > nested.index('id="person-1-group"')
    assert np.abs(pixels(flat) - pixels(nested)).max() == 0


def test_levels_makes_inkscape_layers_without_changing_the_picture():
    img, masks, phrases = scene()
    flat, _, _ = render(img, masks, phrases, round_px=0, seam_px=0)
    layered, _, _ = render(img, masks, phrases, round_px=0, seam_px=0, layers="levels")
    assert layered.count('inkscape:groupmode="layer"') == 3
    assert np.abs(pixels(flat) - pixels(layered)).max() == 0


def test_hierarchy_frees_a_child_when_nesting_would_change_the_order():
    h, w = 40, 40
    def box(y0, y1, x0, x1):
        m = np.zeros((h, w), bool)
        m[y0:y1, x0:x1] = True
        return m
    big = box(0, 40, 0, 40)  # 0: painted first
    cover = box(0, 40, 10, 40)  # 1: painted over most of it
    small = box(5, 15, 12, 20)  # 2: lies inside both
    masks = [big, cover, small]
    inter = lay.intersections(masks)
    pairs = lay.significant(inter, [int(m.sum()) for m in masks])
    parent = lay.hierarchy(masks, inter, pairs)
    order = lay.flatten(lay.tree(parent))
    pos = {k: p for p, k in enumerate(order)}
    assert all(pos[i] < pos[j] for i, j in pairs)  # every overlapping pair keeps its stack order


def test_depth_layers_never_overlap_inside_a_layer_and_keep_the_picture():
    img, masks, phrases = scene()
    flat, _, _ = render(img, masks, phrases, round_px=0, seam_px=0)
    layered, _, _ = render(img, masks, phrases, round_px=0, seam_px=0, layers="depth")
    assert layered.count('inkscape:groupmode="layer"') >= 3
    assert np.abs(pixels(flat) - pixels(layered)).max() == 0
    inter = lay.intersections(masks)
    pairs = lay.significant(inter, [int(m.sum()) for m in masks])
    depth = lay.depths(len(masks), pairs)
    assert all(depth[i] < depth[j] for i, j in pairs)  # a shape is always in front of what it covers
