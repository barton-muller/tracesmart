import numpy as np

from segvec.tones import split_tones


def test_split_tones_separates_two_colours():
    lab = np.zeros((60, 60, 3), np.float32)
    lab[:, :30] = (20, 0, 0)  # dark half
    lab[:, 30:] = (90, 0, 0)  # light half
    labels, k = split_tones(lab, np.ones((60, 60), bool), tol=5, kmax=4, sigma=1.5)
    assert k == 2
    assert len(np.unique(labels[:, :30])) == 1 and len(np.unique(labels[:, 30:])) == 1
    assert labels[0, 0] != labels[0, 59]


def test_split_tones_leaves_flat_regions_alone():
    lab = np.full((40, 40, 3), 50, np.float32)
    _, k = split_tones(lab, np.ones((40, 40), bool), tol=5, kmax=4, sigma=1.5)
    assert k == 1
