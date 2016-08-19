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


def test_time_slice():
    spikes_a = np.arange(1, 100, 5)
    spikes_b = np.arange(24, 62, 1)
    spikes_c = np.hstack((np.arange(0, 24, 3), np.arange(61, 100, 3)))

    t_start = 25
    t_stop = 60

    sliced_spikes_a = vdm.time_slice([spikes_a], t_start, t_stop)
    sliced_spikes_b = vdm.time_slice([spikes_b], t_start, t_stop)
    sliced_spikes_c = vdm.time_slice([spikes_c], t_start, t_stop)

    assert np.allclose(sliced_spikes_a, [26, 31, 36, 41, 46, 51, 56])
    assert np.allclose(sliced_spikes_b, [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                         37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                         49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60])
    assert np.allclose(sliced_spikes_c, [])


def test_idx_in_pos():
    position = dict()
    position['x'] = [0, 1, 2]
    position['y'] = [9, 7, 5]
    position['time'] = [10, 11, 12]

    pos = vdm.idx_in_pos(position, 1)

    assert np.allclose(pos['x'], 1)
    assert np.allclose(pos['y'], 7)
    assert np.allclose(pos['time'], 11)


def test_sort_idx():
    linear = dict()
    linear['position'] = np.linspace(0, 10, 4)
    linear['time'] = np.linspace(0, 3, 4)

    times = [1.5, 0.5, 2.5]
    spikes = dict(time=[])
    for time in times:
        spikes['time'].append([time])

    tuning = vdm.tuning_curve(linear, spikes['time'], sampling_rate=1, binsize=3, gaussian_std=None)
    sort_idx = vdm.get_sort_idx(tuning)

    assert np.allclose(sort_idx, [1, 0, 2])


def test_sort_idx1():
    linear = dict()
    linear['position'] = np.linspace(0, 9, 4)
    linear['time'] = np.linspace(0, 3, 4)

    times = [2.5, 0.0, 2.0, 1.0]
    spikes = dict(time=[])
    for time in times:
        spikes['time'].append([time])

    tuning = vdm.tuning_curve(linear, spikes['time'], sampling_rate=1, binsize=3, gaussian_std=None)
    sort_idx = vdm.get_sort_idx(tuning)

    assert np.allclose(sort_idx, [1, 3, 0, 2])


def test_get_counts():
    spikes = np.hstack((np.arange(0, 10, 1.4), np.arange(0.2, 5, 0.3)))
    spikes = [np.sort(spikes)]

    edges = [0, 2, 4, 6, 8, 10]
    counts = vdm.get_counts(spikes, edges, apply_filter=False)

    assert np.allclose(counts, [9., 7., 5., 1., 2.])
