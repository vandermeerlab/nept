import numpy as np
import pytest
import nept


def test_spiketrain_labels():
    spikes = [nept.SpikeTrain(np.array([1., 3., 5., 7., 9.])),
              nept.SpikeTrain(np.array([1.3, 3.3, 5.3]), 'check_label')]

    assert spikes[0].label == None
    assert spikes[1].label == 'check_label'


def test_spiketrain_sort_times():
    spikes = [nept.SpikeTrain(np.array([9., 7., 5., 3., 1.])),
              nept.SpikeTrain(np.array([1.3, 5.3, 3.3]))]

    assert np.allclose(spikes[0].time, np.array([1., 3., 5., 7., 9.]))
    assert np.allclose(spikes[1].time, np.array([1.3, 3.3, 5.3]))


def test_spiketrain_time_slice():
    spikes_a = nept.SpikeTrain(np.arange(1, 100, 5), 'test')
    spikes_b = nept.SpikeTrain(np.arange(24, 62, 1), 'test')
    spikes_c = nept.SpikeTrain(np.hstack((np.arange(0, 24, 3), np.arange(61, 100, 3))), 'test')

    t_start = 25.
    t_stop = 60.

    sliced_spikes_a = spikes_a.time_slice(t_start, t_stop)
    sliced_spikes_b = spikes_b.time_slice(t_start, t_stop)
    sliced_spikes_c = spikes_c.time_slice(t_start, t_stop)

    assert np.allclose(sliced_spikes_a.time, np.array([26, 31, 36, 41, 46, 51, 56]))
    assert np.allclose(sliced_spikes_b.time, np.array([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                                       37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                                       49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]))
    assert np.allclose(sliced_spikes_c.time, np.array([]))


def test_spiketrain_time_slices():
    spikes = [nept.SpikeTrain(np.array([1., 3., 5., 7., 9.])),
              nept.SpikeTrain(np.array([1.3, 3.3, 5.3]))]

    starts = np.array([1., 7.])
    stops = np.array([4., 10.])

    sliced_spikes = [spike.time_slices(starts, stops) for spike in spikes]

    assert np.allclose(sliced_spikes[0].time, np.array([1., 3., 7., 9.]))
    assert np.allclose(sliced_spikes[1].time, np.array([1.3, 3.3]))
