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


def test_sort_idx():
    linear = vdm.Position(np.linspace(0, 10, 4), np.linspace(0, 3, 4))

    spikes = [vdm.SpikeTrain(np.array([1.5]), 'test'),
              vdm.SpikeTrain(np.array([0.5]), 'test'),
              vdm.SpikeTrain(np.array([2.5]), 'test')]

    tuning = vdm.tuning_curve(linear, spikes, binsize=3, gaussian_std=None)
    print(tuning)
    sort_idx = vdm.get_sort_idx(tuning)

    assert np.allclose(sort_idx, [1, 0, 2])


def test_sort_idx1():
    linear = vdm.Position(np.linspace(0, 9, 4), np.linspace(0, 3, 4))

    spikes = [vdm.SpikeTrain(np.array([2.5]), 'test'),
              vdm.SpikeTrain(np.array([0.0]), 'test'),
              vdm.SpikeTrain(np.array([2.0]), 'test'),
              vdm.SpikeTrain(np.array([1.0]), 'test')]

    tuning = vdm.tuning_curve(linear, spikes, binsize=3, gaussian_std=None)
    print(tuning)
    sort_idx = vdm.get_sort_idx(tuning)

    assert np.allclose(sort_idx, [1, 3, 0, 2])


def test_get_counts():
    spikes = np.hstack((np.arange(0, 10, 1.4), np.arange(0.2, 5, 0.3)))
    spikes = [vdm.SpikeTrain(np.sort(spikes), 'test')]

    edges = [0, 2, 4, 6, 8, 10]
    counts = vdm.get_counts(spikes, edges)

    assert np.allclose(counts, [9., 7., 5., 1., 2.])


def test_get_counts_unequal_edges():
    spikes = [vdm.SpikeTrain([1., 1.1, 1.2, 5., 5.2, 7.6])]

    edges = [0, 2.5, 4, 5, 6, 10]
    counts = vdm.get_counts(spikes, edges)

    assert np.allclose(counts, [3., 0., 0., 2., 1.])
