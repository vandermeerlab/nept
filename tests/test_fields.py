import numpy as np

import vdmlab as vdm


tuning = [np.array([0., 1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0.]),
              np.array([0., 1., 2., 3., 3., 1., 0., 0., 0., 0., 0., 0.]),
              np.array([0., 3., 3., 3., 0., 2., 0., 0., 3., 2., 0., 0.]),
              np.array([0., 1., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0.])]


def test_consecutive():
    array = np.array([0, 3, 4, 5, 9, 12, 13, 14])

    groups = vdm.consecutive(array, stepsize=1)

    assert np.allclose(len(groups), 4)
    assert np.allclose(groups[0], [0])
    assert np.allclose(groups[1], [3, 4, 5])
    assert np.allclose(groups[2], [9])
    assert np.allclose(groups[3], [12, 13, 14])


def test_find_fields():
    with_fields = vdm.find_fields(tuning, hz_thresh=2, min_length=1, max_length=5, max_mean_firing=8)

    assert np.allclose(len(with_fields), 3)
    assert np.allclose(with_fields[1], [np.array([3, 4])])
    assert np.allclose(with_fields[2][0], np.array([1, 2, 3]))
    assert np.allclose(with_fields[2][1], np.array([8]))
    assert np.allclose(with_fields[3], [np.array([7])])


def test_get_single_fields():
    with_fields = vdm.find_fields(tuning, hz_thresh=2, min_length=1, max_length=5, max_mean_firing=8)
    single_fields = vdm.get_single_field(with_fields)

    assert np.allclose(len(single_fields), 2)
    assert np.allclose(single_fields[1], [np.array([3, 4])])
    assert np.allclose(single_fields[3], [np.array([7])])


def test_get_heatmaps():
    pos = np.vstack([np.arange(0, 10, 1), np.hstack((np.arange(0, 10, 2), np.arange(10, 0, -2)))]),
    pos = vdm.Position(pos, np.arange(0, 30, 3))

    neuron_list = [0, 2, 3]

    spikes = [vdm.SpikeTrain(np.array([19.9, 20., 20.1]), 'test'),
              vdm.SpikeTrain(np.array([8.]), 'test'),
              vdm.SpikeTrain(np.array([0., 15., 27.]), 'test'),
              vdm.SpikeTrain(np.array([9., 10., 11., 15., 16.]), 'test')]

    heatmaps = vdm.get_heatmaps(neuron_list, spikes, pos, num_bins=5)

    assert np.allclose(len(heatmaps), 3)
    assert np.allclose(np.max(heatmaps[0]), 3.)
    assert np.allclose(np.mean(heatmaps[0]), 0.12)
    assert np.allclose(np.max(heatmaps[2]), 1.)
    assert np.allclose(np.mean(heatmaps[2]), 0.12)
    assert np.allclose(np.max(heatmaps[3]), 2.)
    assert np.allclose(np.mean(heatmaps[3]), 0.2)
