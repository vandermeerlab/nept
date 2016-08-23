import numpy as np
import pytest
import vdmlab as vdm


def test_idx_in_pos():
    position = vdm.Position([[0, 1, 2], [9, 7, 5]], [10, 11, 12])
    pos = position[1]

    assert np.allclose(pos.x, 1)
    assert np.allclose(pos.y, 7)
    assert np.allclose(pos.time, 11)


def test_time_slice():
    spikes_a = vdm.SpikeTrain(np.arange(1, 100, 5), 'test')
    spikes_b = vdm.SpikeTrain(np.arange(24, 62, 1), 'test')
    spikes_c = vdm.SpikeTrain(np.hstack((np.arange(0, 24, 3), np.arange(61, 100, 3))), 'test')

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


def test_epoch_duration():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = vdm.Epoch(times)
    assert np.allclose(epoch.durations, np.array([1., 0.6, 0.4]))


def test_epoch_stops():
    start_times = np.array([[0.0],
                            [0.9],
                            [1.6]])
    durations = np.array([1., 0.6, 0.4])
    epoch = vdm.Epoch(start_times, durations)
    assert np.allclose(epoch.stops, np.array([1., 1.5, 2.]))


def test_epoch_sort():
    times = np.array([[1.6, 2.0],
                      [0.0, 1.0],
                      [0.9, 1.5]])
    epoch = vdm.Epoch(times)
    assert np.allclose(epoch.starts, np.array([0., 0.9, 1.6]))
    assert np.allclose(epoch.stops, np.array([1., 1.5, 2.]))


def test_epoch_sortlist():
    start_times = [0.9, 0.0, 1.6]
    durations = [0.6, 1.0, 0.4]
    epoch = vdm.Epoch(start_times, durations)
    assert np.allclose(epoch.starts, np.array([0.0, 0.9, 1.6]))
    assert np.allclose(epoch.stops, np.array([1., 1.5, 2.]))


def test_epoch_reshape():
    times = np.array([[0.0, 0.9, 1.6], [1.0, 1.5, 2.0]])
    epoch = vdm.Epoch(times)
    assert np.allclose(epoch.time.shape, (3, 2))


def test_epoch_centers():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.4],
                      [1.6, 2.0]])
    epoch = vdm.Epoch(times)
    assert np.allclose(epoch.centers, np.array([0.5, 1.15, 1.8]))


def test_epoch_too_many_parameters():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    durations = np.array([1., 0.6, 0.4])
    with pytest.raises(ValueError) as excinfo:
        epoch = vdm.Epoch(times, durations)
    assert str(excinfo.value) == 'duration not allowed when using start and stop times'


def test_epoch_simple_intersect():
    times_1 = np.array([[0.0, 1.0],
                        [0.9, 1.5],
                        [1.6, 2.0]])
    epoch_1 = vdm.Epoch(times_1)

    times_2 = np.array([[1.55, 1.8]])
    epoch_2 = vdm.Epoch(times_2)

    intersected_epochs = epoch_1.intersect(epoch_2)
    assert np.allclose(intersected_epochs.starts, np.array([1.6]))
    assert np.allclose(intersected_epochs.stops, np.array([1.8]))


def test_epoch_intersect():
    times_1 = np.array([[0.0, 1.0],
                        [0.9, 1.5],
                        [1.6, 2.0]])
    epoch_1 = vdm.Epoch(times_1)

    times_2 = np.array([[1.2, 1.8]])
    epoch_2 = vdm.Epoch(times_2)

    intersected_epochs = epoch_1.intersect(epoch_2)

    assert np.allclose(intersected_epochs.starts, np.array([1.2, 1.6]))
    assert np.allclose(intersected_epochs.stops, np.array([1.5, 1.8]))


def test_epoch_no_intersect():
    times_1 = np.array([[0.0, 1.0],
                        [0.9, 1.5],
                        [1.6, 2.0]])
    epoch_1 = vdm.Epoch(times_1)

    times_2 = np.array([[1.5, 1.6]])
    epoch_2 = vdm.Epoch(times_2)

    intersected_epochs = epoch_1.intersect(epoch_2)

    assert np.allclose(intersected_epochs.starts, np.array([]))
    assert np.allclose(intersected_epochs.stops, np.array([]))


def test_epoch_merge():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])

    epoch = vdm.Epoch(times)

    merged = epoch.merge()
    assert np.allclose(merged.starts, np.array([0., 1.5]))
    assert np.allclose(merged.stops, np.array([1.6, 2.0]))


def test_epoch_merge_with_gap():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])

    epoch = vdm.Epoch(times)

    merged = epoch.merge(gap=0.1)
    assert np.allclose(merged.starts, np.array([0.]))
    assert np.allclose(merged.stops, np.array([2.0]))


def test_epoch_resize_both():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = vdm.Epoch(times)

    resized = epoch.resize(0.5)

    assert np.allclose(resized.starts, np.array([-0.5, 0.4, 1.1]))
    assert np.allclose(resized.stops, np.array([1.5, 2.0, 2.5]))


def test_epoch_resize_start():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = vdm.Epoch(times)

    resized = epoch.resize(0.5, direction='start')

    assert np.allclose(resized.starts, np.array([-0.5, 0.4, 1.1]))
    assert np.allclose(resized.stops, np.array([1.0, 1.5, 2.0]))


def test_epoch_resize_stop():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = vdm.Epoch(times)

    resized = epoch.resize(0.5, direction='stop')

    assert np.allclose(resized.starts, np.array([0.0, 0.9, 1.6]))
    assert np.allclose(resized.stops, np.array([1.5, 2.0, 2.5]))


def test_epoch_resize_shrink():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = vdm.Epoch(times)

    shrinked = epoch.resize(-0.1)

    assert np.allclose(shrinked.starts, np.array([0.1, 1.0, 1.7]))
    assert np.allclose(shrinked.stops, np.array([0.9, 1.4, 1.9]))


def test_epoch_join():
    times_1 = np.array([[0.0, 1.0],
                        [0.9, 1.5],
                        [1.6, 2.0]])
    epoch_1 = vdm.Epoch(times_1)

    times_2 = np.array([[1.8, 2.5]])
    epoch_2 = vdm.Epoch(times_2)

    union = epoch_1.join(epoch_2)

    assert np.allclose(union.starts, np.array([0.0, 0.9, 1.6, 1.8]))
    assert np.allclose(union.stops, np.array([1.0, 1.5, 2.0, 2.5]))
