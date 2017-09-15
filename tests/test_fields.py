import numpy as np
import pytest

import nept


tuning = [np.array([0., 1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0.]),
          np.array([0., 1., 2., 3., 3., 1., 0., 0., 0., 0., 0., 0.]),
          np.array([0., 3., 3., 3., 0., 2., 0., 0., 3., 2., 0., 0.]),
          np.array([0., 1., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0.])]


def test_find_fields():
    with_fields = nept.find_fields(tuning, hz_thresh=2, min_length=1, max_length=5, max_mean_firing=8)

    assert np.allclose(len(with_fields), 3)
    assert np.allclose(with_fields[1], [np.array([3, 4])])
    assert np.allclose(with_fields[2][0], np.array([1, 2, 3]))
    assert np.allclose(with_fields[2][1], np.array([8]))
    assert np.allclose(with_fields[3], [np.array([7])])


def test_find_fields_firing_thresh():
    with_fields = nept.find_fields(tuning, hz_thresh=2, min_length=1, max_length=5, max_mean_firing=1)

    assert np.allclose(len(with_fields), 2)
    assert np.allclose(with_fields[1], [np.array([3, 4])])
    assert np.allclose(with_fields[3], [np.array([7])])


def test_find_fields_length_thresh():
    with_fields = nept.find_fields(tuning, hz_thresh=2, min_length=1, max_length=1, max_mean_firing=1)

    assert np.allclose(len(with_fields), 1)
    assert np.allclose(with_fields[3], [np.array([7])])


def test_get_single_fields():
    with_fields = nept.find_fields(tuning, hz_thresh=2, min_length=1, max_length=5, max_mean_firing=8)
    single_fields = nept.get_single_field(with_fields)

    assert np.allclose(len(single_fields), 2)
    assert np.allclose(single_fields[1], [np.array([3, 4])])
    assert np.allclose(single_fields[3], [np.array([7])])


def test_get_heatmaps():
    position = np.vstack([np.arange(0, 10, 1), np.hstack((np.arange(0, 10, 2), np.arange(10, 0, -2)))]),
    position = nept.Position(position, np.arange(0, 30, 3))

    neuron_list = [0, 2, 3]

    spikes = [nept.SpikeTrain(np.array([19.9, 20., 20.1]), 'test'),
              nept.SpikeTrain(np.array([8.]), 'test'),
              nept.SpikeTrain(np.array([0., 15., 27.]), 'test'),
              nept.SpikeTrain(np.array([9., 10., 11., 15., 16.]), 'test')]

    heatmaps = nept.get_heatmaps(neuron_list, spikes, position, num_bins=5)

    assert np.allclose(len(heatmaps), 3)
    assert np.allclose(np.max(heatmaps[0]), 3.)
    assert np.allclose(np.mean(heatmaps[0]), 0.12)
    assert np.allclose(np.max(heatmaps[2]), 1.)
    assert np.allclose(np.mean(heatmaps[2]), 0.12)
    assert np.allclose(np.max(heatmaps[3]), 2.)
    assert np.allclose(np.mean(heatmaps[3]), 0.2)


def test_get_heatmaps_1d_position():
    position = nept.Position(np.arange(0, 10, 1), np.arange(0, 30, 3))

    neuron_list = [0, 2, 3]

    spikes = [nept.SpikeTrain(np.array([19.9, 20., 20.1]), 'test'),
              nept.SpikeTrain(np.array([8.]), 'test'),
              nept.SpikeTrain(np.array([0., 15., 27.]), 'test'),
              nept.SpikeTrain(np.array([9., 10., 11., 15., 16.]), 'test')]

    with pytest.raises(ValueError) as excinfo:
        heatmaps = nept.get_heatmaps(neuron_list, spikes, position, num_bins=5)

    assert str(excinfo.value) == 'pos must be two-dimensional'
