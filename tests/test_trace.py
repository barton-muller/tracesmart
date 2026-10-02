import numpy as np

from segvec.trace import hex_colour, mask_path, smooth_mask


def disc(n=100, r=30):
    yy, xx = np.mgrid[:n, :n]
    return (yy - n / 2) ** 2 + (xx - n / 2) ** 2 < r**2


def test_rectangle_keeps_sharp_corners():
    mask = np.zeros((100, 100), bool)
    mask[20:60, 30:80] = True
    d = mask_path(mask)
    assert d.startswith("M") and d.endswith("Z")
    assert "C" not in d  # four corners joined by straight lines
    assert d.count("L") == 3


def test_circle_is_smooth():
    d = mask_path(disc())
    assert "C" in d


def test_empty_mask_has_no_path():
    assert mask_path(np.zeros((10, 10), bool)) is None


def test_smooth_mask_rounds_corners_and_keeps_shape():
    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:80] = True
    out = smooth_mask(mask, 4.0)
    assert out.sum() < mask.sum()
    assert out.sum() > 0.9 * mask.sum()
    assert not out[20, 20]  # the corner pixel is rounded off
    assert out[50, 50]


def test_hex_colour():
    assert hex_colour((255, 0, 128.4)) == "#ff0080"
