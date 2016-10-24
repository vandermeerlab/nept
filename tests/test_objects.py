import numpy as np
import pytest
import vdmlab as vdm
from shapely.geometry import Point, LineString


def test_position_xy():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([[1.1, 3.1],
                     [0.9, 2.0],
                     [2.3, 1.4]])

    pos = vdm.Position(data, times)

    assert np.allclose(pos.x, np.array([1.1, 0.9, 2.3]))
    assert np.allclose(pos.y, np.array([3.1, 2.0, 1.4]))
    assert np.allclose(pos.time, np.array([1.0, 2.0, 3.0]))


def test_position_xy_reshaped():
    times = np.array([1.0, 2.0, 3.0])
    x = np.array([1.1, 0.9, 2.3])
    y = np.array([3.1, 2.0, 1.4])

    pos = vdm.Position(np.array([x, y]), times)

    assert np.allclose(pos.x, np.array([1.1, 0.9, 2.3]))
    assert np.allclose(pos.y, np.array([3.1, 2.0, 1.4]))
    assert np.allclose(pos.time, np.array([1.0, 2.0, 3.0]))


def test_position_idx_in_pos():
    position = vdm.Position([[0, 1, 2], [9, 7, 5]], [10, 11, 12])
    pos = position[1]

    assert np.allclose(pos.x, 1)
    assert np.allclose(pos.y, 7)
    assert np.allclose(pos.time, 11)


def test_position_1d_y():
    times = np.array([1.0, 2.0, 3.0])
    x = np.array([1.1, 0.9, 2.3])

    pos = vdm.Position(x, times)

    with pytest.raises(ValueError) as excinfo:
        y_val = pos.y
    assert str(excinfo.value) == "can't get 'y' of one-dimensional position"


def test_position_distance_1d():
    times = np.array([1.0, 2.0, 3.0])
    x = np.array([1.1, 0.9, 2.3])
    y = np.array([3.1, 2.0, 1.4])

    pos = vdm.Position(x, times)
    other = vdm.Position(y, times)

    distance_pos = pos.distance(other)
    distance_other = other.distance(pos)

    assert np.allclose(distance_pos, np.array([2.0, 1.1, 0.9]))
    assert np.all(distance_other == distance_pos)


def test_position_distance_2d():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([[1.0, 3.1],
                     [2.0, 2.1],
                     [3.0, 1.1]])
    times2 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([[1.2, 3.3],
                      [2.0, 2.6],
                      [3.0, 1.1]])

    pos = vdm.Position(data, times)
    other = vdm.Position(data2, times2)

    dist = pos.distance(other)

    assert np.allclose(dist, np.array([0.28284271, 0.5, 0.0]))


def test_position_distance_dimensions():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([[1.0, 3.1],
                     [2.0, 2.1],
                     [3.0, 1.1]])
    x = np.array([1.1, 0.9, 2.3])

    pos = vdm.Position(data, times)
    other = vdm.Position(x, times)

    with pytest.raises(ValueError) as excinfo:
        dist = pos.distance(other)

    assert str(excinfo.value) == "'pos' must be 2 dimensions"


def test_position_distance_diff_size():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    data = np.array([[1.0, 3.1],
                     [2.0, 2.1],
                     [3.0, 1.1],
                     [4.0, 0.1]])
    times2 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([[1.2, 3.3],
                      [2.0, 2.6],
                      [3.0, 1.1]])

    pos = vdm.Position(data, times)
    other = vdm.Position(data2, times2)

    with pytest.raises(ValueError) as excinfo:
        dist = pos.distance(other)
    assert str(excinfo.value) == "'pos' must have 4 samples"


def test_position_linearize():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([[0.0, 0.5],
                     [0.5, 0.1],
                     [1.0, 1.2]])

    pos = vdm.Position(data, times)
    line = LineString([(0.0, 0.0), (1.0, 1.0)])

    zone_start = Point([1., 1.])
    zone_stop = Point([9., 9.])
    expand_by = 1
    zone = vdm.expand_line(zone_start, zone_stop, line, expand_by)

    linear = pos.linearize(line, zone)

    assert np.allclose(linear.x, np.array([0.35355339, 0.42426407, 1.41421356]))
    assert np.allclose(linear.time, np.array([1., 2., 3.]))


def test_positon_speed_simple():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.array([0.0, 0.5, 1.0, 0.7, 1.7])

    pos = vdm.Position(data, times)
    speed = pos.speed()

    assert np.allclose(speed.data, np.array([[0.0], [0.5], [0.5], [0.3], [1.0]]))


def test_position_speed_complex():
    time = np.linspace(0, np.pi * 2, 201)
    data = np.hstack((np.sin(time)))

    position = vdm.Position(data, time)
    speed = position.speed()
    run_idx = np.squeeze(speed.data) >= 0.015
    run_position = position[run_idx]

    assert np.allclose(len(run_position.x), 136)


def test_position_speed_complex2():
    time = np.linspace(0, np.pi * 2, 201)
    data = np.hstack((np.sin(time)))

    position = vdm.Position(data, time)
    speed = position.speed()
    run_idx = np.squeeze(speed.data) >= 0.01
    run_position = position[run_idx]

    assert np.allclose(len(run_position.x), 160)


def test_position_speed_unequal_time():
    time = np.hstack((np.linspace(0, 10, 10), np.linspace(11, 101, 10)))
    data = np.arange(0, 20)

    position = vdm.Position(data, time)
    speed = position.speed()
    run_idx = np.squeeze(speed.data) >= 0.7
    run_position = position[run_idx]

    assert np.allclose(len(run_position.x), 10)


def test_localfieldpotential_nsamples():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([1.1, 0.9, 2.3])
    lfp = vdm.LocalFieldPotential(data, times)

    assert np.allclose(lfp.n_samples, 3)


def test_localfieldpotential_too_many():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([1.1, 0.9, 2.3])
    other = np.array([3.1, 2.0, 1.4])

    with pytest.raises(ValueError) as excinfo:
        lfp = vdm.LocalFieldPotential([data, other], times)

    assert str(excinfo.value) == 'can only contain one LFP'


def test_localfieldpotential_slice():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([1.1, 0.9, 2.3])

    lfp = vdm.LocalFieldPotential(data, times)
    sliced = lfp[:2]

    assert np.allclose(sliced.time, np.array([1.0, 2.0]))
    assert np.allclose(sliced.data, np.array([[1.1], [0.9]]))


def test_spiketrain_time_slice():
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
                        [1.1, 1.5],
                        [1.6, 2.0]])
    epoch_1 = vdm.Epoch(times_1)

    times_2 = np.array([[1.55, 1.8]])
    epoch_2 = vdm.Epoch(times_2)

    intersected_epochs = epoch_1.intersect(epoch_2)
    assert np.allclose(intersected_epochs.starts, np.array([1.6]))
    assert np.allclose(intersected_epochs.stops, np.array([1.8]))


def test_epoch_intersect():
    times_1 = np.array([[0.0, 1.0],
                        [1.1, 1.5],
                        [1.6, 2.0]])
    epoch_1 = vdm.Epoch(times_1)

    times_2 = np.array([[1.2, 1.8]])
    epoch_2 = vdm.Epoch(times_2)

    intersects = epoch_1.intersect(epoch_2)

    assert np.allclose(intersects.starts, np.array([1.2, 1.6]))
    assert np.allclose(intersects.stops, np.array([1.5, 1.8]))


def test_epoch_contains():
    times_1 = np.array([[0.0, 1.0],
                        [1.1, 1.5],
                        [1.6, 2.0]])
    epoch_1 = vdm.Epoch(times_1)

    times_2 = np.array([[1.2, 1.8]])
    epoch_2 = vdm.Epoch(times_2)

    contains = epoch_1.intersect(epoch_2, boundaries=False)

    assert np.allclose(contains.starts, np.array([1.2]))
    assert np.allclose(contains.stops, np.array([1.8]))


def test_epoch_intersect_a_short():
    times_a = np.array([[1.0, 2.0]])
    epoch_a = vdm.Epoch(times_a)

    times_b = np.array([[0.0, 3.0]])
    epoch_b = vdm.Epoch(times_b)

    intersects = epoch_a.intersect(epoch_b)

    assert np.allclose(intersects.starts, np.array([1.0]))
    assert np.allclose(intersects.stops, np.array([2.0]))


def test_epoch_intersect_a_long():
    times_a = np.array([[1.0, 2.0]])
    epoch_a = vdm.Epoch(times_a)

    times_b = np.array([[1.1, 1.9]])
    epoch_b = vdm.Epoch(times_b)

    intersects = epoch_a.intersect(epoch_b)

    assert np.allclose(intersects.starts, np.array([1.1]))
    assert np.allclose(intersects.stops, np.array([1.9]))


def test_epoch_intersect_a_left():
    times_a = np.array([[1.0, 2.0]])
    epoch_a = vdm.Epoch(times_a)

    times_b = np.array([[1.5, 2.5]])
    epoch_b = vdm.Epoch(times_b)

    intersects = epoch_a.intersect(epoch_b)

    assert np.allclose(intersects.starts, np.array([1.5]))
    assert np.allclose(intersects.stops, np.array([2.0]))


def test_epoch_intersect_a_right():
    times_a = np.array([[1.0, 2.0]])
    epoch_a = vdm.Epoch(times_a)

    times_b = np.array([[0.5, 1.7]])
    epoch_b = vdm.Epoch(times_b)

    intersects = epoch_a.intersect(epoch_b)

    assert np.allclose(intersects.starts, np.array([1.0]))
    assert np.allclose(intersects.stops, np.array([1.7]))


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


def test_epoch_merge_far_stop():
    times = np.array([[0.0, 10.0],
                      [1.0, 3.0],
                      [2.0, 5.0],
                      [11.0, 12.0]])

    epoch = vdm.Epoch(times)
    merged = epoch.merge()
    assert np.allclose(merged.starts, np.array([0.0, 10.0]))
    assert np.allclose(merged.stops, np.array([11.0, 12.0]))


def test_epoch_expand_both():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = vdm.Epoch(times)

    resized = epoch.expand(0.5)

    assert np.allclose(resized.starts, np.array([-0.5, 0.4, 1.1]))
    assert np.allclose(resized.stops, np.array([1.5, 2.0, 2.5]))


def test_epoch_expand_start():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = vdm.Epoch(times)

    resized = epoch.expand(0.5, direction='start')

    assert np.allclose(resized.starts, np.array([-0.5, 0.4, 1.1]))
    assert np.allclose(resized.stops, np.array([1.0, 1.5, 2.0]))


def test_epoch_expand_stop():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = vdm.Epoch(times)

    resized = epoch.expand(0.5, direction='stop')

    assert np.allclose(resized.starts, np.array([0.0, 0.9, 1.6]))
    assert np.allclose(resized.stops, np.array([1.5, 2.0, 2.5]))


def test_epoch_shrink():
    times = np.array([[0.0, 1.0],
                      [0.9, 1.5],
                      [1.6, 2.0]])
    epoch = vdm.Epoch(times)

    shrinked = epoch.shrink(0.1)

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


def test_epoch_start_stop():
    epoch = vdm.Epoch(np.array([[721.9412, 900.0],
                                [1000.0, 1027.1]]))

    assert np.allclose(epoch.start, 721.9412)
    assert np.allclose(epoch.stop, 1027.1)
