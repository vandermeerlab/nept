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
