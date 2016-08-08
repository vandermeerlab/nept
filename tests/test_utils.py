import numpy as np

import vdmlab as vdm


toy_array = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

def test_find_nearest_idx():
    assert vdm.find_nearest_idx(toy_array, 13) == 3
    assert vdm.find_nearest_idx(toy_array, 11.49) == 1
    assert vdm.find_nearest_idx(toy_array, 11.51) == 2
    assert vdm.find_nearest_idx(toy_array, 25) == 10
    assert vdm.find_nearest_idx(toy_array, 1) == 0

def test_find_nearest_indices():
    assert np.allclose(vdm.find_nearest_indices(toy_array, np.array([13.2])), np.array([3]))
    assert np.allclose(vdm.find_nearest_indices(toy_array, np.array([10, 20])), np.array([0, 10]))
