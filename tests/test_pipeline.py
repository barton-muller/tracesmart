import numpy as np

from segvec.pipeline import Canvas, prompt_points, tidy


def test_tidy_fills_holes_and_keeps_largest_piece():
    mask = np.zeros((60, 60), bool)
    mask[5:40, 5:40] = True
    mask[15:25, 15:25] = False  # hole
    mask[50:53, 50:53] = True  # speck
    out = tidy(mask, min_px=20)
    assert out[20, 20]  # hole filled
    assert not out[51, 51]  # speck dropped


def test_tidy_rejects_small_masks():
    mask = np.zeros((60, 60), bool)
    mask[5:8, 5:8] = True
    assert tidy(mask, min_px=50) is None


def test_canvas_keeps_masks_that_matter():
    rgb = np.zeros((80, 80, 3), np.uint8)
    rgb[:, :40] = (200, 30, 30)
    rgb[:, 40:] = (30, 30, 200)
    canvas = Canvas(rgb)
    everything = np.ones((80, 80), bool)
    left = np.zeros((80, 80), bool)
    left[:, :40] = True
    assert canvas.try_add(everything, impact=1e-4)
    assert canvas.try_add(left, impact=1e-4)  # fixes half the image, so it matters
    assert not canvas.try_add(left, impact=1e-4)  # a repeat gains nothing
    assert len(canvas.kept) == 2


def test_prompt_points_finds_the_uncovered_region():
    rgb = np.full((100, 100, 3), 120, np.uint8)
    canvas = Canvas(rgb)
    covered = np.ones((100, 100), bool)
    covered[30:90, 30:90] = False  # a big hole
    canvas.try_add(covered, impact=0.0)
    points = prompt_points(canvas, radius=10, err_thresh=3 * 30.0**2)
    assert len(points) == 1
    x, y = points[0]
    assert 30 <= x < 90 and 30 <= y < 90
