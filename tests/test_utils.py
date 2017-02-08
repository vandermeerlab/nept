import numpy as np
import pytest
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
    sort_idx = vdm.get_sort_idx(tuning)

    assert np.allclose(sort_idx, [1, 0, 2])


def test_sort_idx1():
    linear = vdm.Position(np.linspace(0, 9, 4), np.linspace(0, 3, 4))

    spikes = [vdm.SpikeTrain(np.array([2.5]), 'test'),
              vdm.SpikeTrain(np.array([0.0]), 'test'),
              vdm.SpikeTrain(np.array([2.0]), 'test'),
              vdm.SpikeTrain(np.array([1.0]), 'test')]

    tuning = vdm.tuning_curve(linear, spikes, binsize=3, gaussian_std=None)
    sort_idx = vdm.get_sort_idx(tuning)

    assert np.allclose(sort_idx, [1, 3, 0, 2])


def test_get_counts():
    spikes = np.hstack((np.arange(0, 10, 1.4), np.arange(0.2, 5, 0.3)))
    spikes = [vdm.SpikeTrain(np.sort(spikes), 'test')]

    edges = [0, 2, 4, 6, 8, 10]
    counts = vdm.get_counts(spikes, edges)

    assert np.allclose(counts.data, np.array([[9.], [7.], [5.], [1.], [2.]]))


def test_get_counts_unequal_edges():
    spikes = [vdm.SpikeTrain([1., 1.1, 1.2, 5., 5.2, 7.6])]

    edges = [0, 2.5, 4, 5, 6, 10]
    counts = vdm.get_counts(spikes, edges)

    assert np.allclose(counts.data, np.array([[3.], [0.], [0.], [2.], [1.]]))


def test_get_counts_large_gaussian():
    spikes = [vdm.SpikeTrain([1., 1.1, 1.2, 5., 5.2, 7.6])]

    edges = [0, 2.5, 4, 5, 6, 10]

    with pytest.raises(ValueError) as excinfo:
        counts = vdm.get_counts(spikes, edges, gaussian_std=3)

    assert str(excinfo.value) == 'gaussian_std is too large for these times'


def test_multi_in_epochs_one():
    epochs = vdm.Epoch(np.array([[1.0, 4.0, 6.0], [2.0, 5.0, 7.0]]))

    spikes = [vdm.SpikeTrain(np.array([6.7])),
              vdm.SpikeTrain(np.array([1.1, 6.5])),
              vdm.SpikeTrain(np.array([1.3, 4.1])),
              vdm.SpikeTrain(np.array([1.7, 4.3]))]

    min_involved = 3
    multi_epochs = vdm.find_multi_in_epochs(spikes, epochs, min_involved)

    assert np.allclose(multi_epochs.starts, np.array([1.]))
    assert np.allclose(multi_epochs.stops, np.array([2.]))


def test_multi_in_epochs_mult():
    epochs = vdm.Epoch(np.array([[1.0, 4.0, 6.0], [2.0, 5.0, 7.0]]))

    spikes = [vdm.SpikeTrain(np.array([1.1, 6.5])),
              vdm.SpikeTrain(np.array([1.3, 4.1])),
              vdm.SpikeTrain(np.array([1.7, 4.3]))]

    min_involved = 2
    multi_epochs = vdm.find_multi_in_epochs(spikes, epochs, min_involved)

    assert np.allclose(multi_epochs.starts, np.array([1., 4.]))
    assert np.allclose(multi_epochs.stops, np.array([2., 5.]))


def test_multi_in_epoch_none():
    epochs = vdm.Epoch(np.array([[0.0], [1.0]]))

    spikes = [vdm.SpikeTrain(np.array([1.1, 6.5])),
              vdm.SpikeTrain(np.array([1.3, 4.1])),
              vdm.SpikeTrain(np.array([1.7, 4.3]))]

    min_involved = 2
    multi_epochs = vdm.find_multi_in_epochs(spikes, epochs, min_involved)

    assert np.allclose(multi_epochs.starts, np.array([]))
    assert np.allclose(multi_epochs.stops, np.array([]))


def test_get_xyedges_mult():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([[1.0, 1.1],
                     [5.0, 5.1],
                     [10.0, 10.1]])

    position = vdm.Position(data, times)

    xedges, yedges = vdm.get_xyedges(position, binsize=3)

    assert np.allclose(xedges, np.array([1., 4., 7., 10.]))
    assert np.allclose(yedges, np.array([1.1, 4.1, 7.1, 10.1]))


def test_get_xyedges_one_full():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([[1.0, 1.1],
                     [5.0, 5.1],
                     [10.0, 10.1]])

    position = vdm.Position(data, times)

    xedges, yedges = vdm.get_xyedges(position, binsize=10)

    assert np.allclose(xedges, np.array([1., 11.]))
    assert np.allclose(yedges, np.array([1.1, 11.1]))
