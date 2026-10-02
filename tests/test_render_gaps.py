import numpy as np

from tracesmart.render import uncovered_regions


def test_uncovered_regions_are_found_and_small_ones_ignored():
    covered = np.zeros((100, 100), bool)
    covered[:50, :] = True  # top half covered; bottom half is one big gap
    covered[60, 60] = False
    gaps = uncovered_regions([covered], min_px=100)
    assert len(gaps) == 1
    assert gaps[0][80, 10] and not gaps[0][10, 10]
