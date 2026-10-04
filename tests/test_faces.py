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
