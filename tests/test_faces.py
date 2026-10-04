"""The model-free parts of faces: polygon filling and the dark-feature rule need no weights."""
import numpy as np

from tracesmart import faces


def test_fill_polygon_subpixel():
    square = np.array([[2.5, 2.5], [7.5, 2.5], [7.5, 7.5], [2.5, 7.5]])
    m = faces.fill((10, 10), square)
    assert m.dtype == bool and 20 <= m.sum() <= 36 and m[5, 5] and not m[0, 0]


def test_fill_polyline_has_thickness():
    line = np.array([[1.0, 5.0], [9.0, 5.0]])
    assert faces.fill((10, 10), line, 3).sum() > faces.fill((10, 10), line, 1).sum()


def test_grow_keeps_centre():
    poly = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
    big = faces.grow(poly, 2.0)
    assert np.allclose(big.mean(0), poly.mean(0)) and np.ptp(big[:, 0]) == 8


def test_stroke_is_a_thin_line_inside_the_mask():
    brow = np.zeros((40, 80), bool)
    brow[18:26, 10:70] = True  # a thick, flat brow
    line = faces.stroke(brow, 0.5, 2)
    assert line is not None and 0 < line.sum() < brow.sum() * 0.75
    assert (line & ~brow).sum() <= 0.1 * line.sum()  # stays on the brow
    assert faces.stroke(np.zeros((10, 10), bool)) is None


def test_insert_pos_goes_above_the_last_big_overlapping_shape_whatever_the_order():
    def box(x0, x1):
        m = np.zeros((20, 40), bool)
        m[:, x0:x1] = True
        return m

    new = box(5, 15)  # 200 px
    items = [(box(0, 40), None), (box(30, 40), None), (box(0, 20), None), (box(6, 8), None)]
    # the first and third are bigger and cover it; the second does not touch it; the last is a small detail
    assert faces.insert_pos(items, new) == 3
    assert faces.insert_pos([(box(30, 40), None)], new) == 0


def test_same_thing_replaces_duplicates_and_shapes_inside_but_not_neighbours():
    new = np.zeros((20, 40), bool)
    new[:, 5:25] = True
    dup = np.zeros_like(new)
    dup[:, 6:26] = True
    detail = np.zeros_like(new)
    detail[8:12, 10:14] = True
    beside = np.zeros_like(new)
    beside[:, 24:40] = True
    assert faces.same_thing(dup, new) and faces.same_thing(detail, new) and not faces.same_thing(beside, new)


def test_hair_colour_ignores_bright_gaps_between_curls():
    from tracesmart.render import shape_colour

    px = np.array([[40, 30, 20]] * 70 + [[200, 220, 255]] * 20, float)  # brown curls with sky showing through
    assert shape_colour("hair", px).max() < 50 and shape_colour(None, px).max() > 60


def test_squinting_eyes_count_as_closed_in_both_styles():
    open_eye = np.array([[0, 5], [6, 0], [12, 0], [18, 5], [12, 10], [6, 10]], float)  # height about half the width
    shut = np.array([[0, 5], [6, 4], [12, 4], [18, 5], [12, 6], [6, 6]], float)
    assert not faces.eye_closed(open_eye) and faces.eye_closed(shut)
    arc, phrase = faces.eye_arc(shut, 40.0, (12, 20))
    assert phrase == "eye" and arc.sum() >= 18
