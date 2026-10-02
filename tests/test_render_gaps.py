import numpy as np

from tracesmart.render import uncovered_regions


def test_uncovered_regions_are_found_and_small_ones_ignored():
    covered = np.zeros((100, 100), bool)
    covered[:50, :] = True  # top half covered; bottom half is one big gap
    covered[60, 60] = False
    gaps = uncovered_regions([covered], min_px=100)
    assert len(gaps) == 1
    assert gaps[0][80, 10] and not gaps[0][10, 10]


def test_close_seams_removes_the_sliver_between_neighbours():
    from tracesmart.render import close_seams

    below = np.ones((60, 120), bool)  # a big shape underneath everything
    left = np.zeros_like(below)
    left[:, :58] = True
    right = np.zeros_like(below)
    right[:, 61:] = True  # a 3 px gap where `below` would show through
    grown = close_seams([below, left, right], max_px=2)
    covered = grown[1] | grown[2]
    assert covered[:, 55:65].all()
    assert grown[0].all()  # the shape below is unchanged
