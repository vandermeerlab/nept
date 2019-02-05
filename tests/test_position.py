import numpy as np
import pytest
from shapely.geometry import Point, LineString
import nept


def test_position_xy():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([[1.1, 3.1],
                     [0.9, 2.0],
                     [2.3, 1.4]])

    pos = nept.Position(data, times)

    assert np.allclose(pos.x, np.array([1.1, 0.9, 2.3]))
    assert np.allclose(pos.y, np.array([3.1, 2.0, 1.4]))
    assert np.allclose(pos.time, np.array([1.0, 2.0, 3.0]))


def test_position_xy_reshaped():
    times = np.array([1.0, 2.0, 3.0])
    x = np.array([1.1, 0.9, 2.3])
    y = np.array([3.1, 2.0, 1.4])

    pos = nept.Position(np.array([x, y]), times)

    assert np.allclose(pos.x, np.array([1.1, 0.9, 2.3]))
    assert np.allclose(pos.y, np.array([3.1, 2.0, 1.4]))
    assert np.allclose(pos.time, np.array([1.0, 2.0, 3.0]))


def test_position_idx_in_pos():
    position = nept.Position([[0, 1, 2], [9, 7, 5]], [10, 11, 12])
    pos = position[1]

    assert np.allclose(pos.x, 1)
    assert np.allclose(pos.y, 7)
    assert np.allclose(pos.time, 11)


def test_position_1d_y():
    times = np.array([1.0, 2.0, 3.0])
    x = np.array([1.1, 0.9, 2.3])

    pos = nept.Position(x, times)

    with pytest.raises(ValueError) as excinfo:
        y_val = pos.y
    assert str(excinfo.value) == "can't get 'y' of one-dimensional position"


def test_position_distance_1d():
    times = np.array([1.0, 2.0, 3.0])
    x = np.array([1.1, 0.9, 2.3])
    y = np.array([3.1, 2.0, 1.4])

    pos = nept.Position(x, times)
    other = nept.Position(y, times)

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

    pos = nept.Position(data, times)
    other = nept.Position(data2, times2)

    dist = pos.distance(other)

    assert np.allclose(dist, np.array([0.28284271, 0.5, 0.0]))


def test_position_distance_dimensions():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([[1.0, 3.1],
                     [2.0, 2.1],
                     [3.0, 1.1]])
    x = np.array([1.1, 0.9, 2.3])

    pos = nept.Position(data, times)
    other = nept.Position(x, times)

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

    pos = nept.Position(data, times)
    other = nept.Position(data2, times2)

    with pytest.raises(ValueError) as excinfo:
        dist = pos.distance(other)
    assert str(excinfo.value) == "'pos' must have 4 samples"


def test_position_linearize():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([[0.0, 0.5],
                     [0.5, 0.1],
                     [1.0, 1.2]])

    pos = nept.Position(data, times)
    line = LineString([(0.0, 0.0), (1.0, 1.0)])

    zone_start = Point([1., 1.])
    zone_stop = Point([9., 9.])
    expand_by = 1
    zone = nept.expand_line(zone_start, zone_stop, line, expand_by)

    linear = pos.linearize(line, zone)

    assert np.allclose(linear.x, np.array([0.35355339, 0.42426407, 1.41421356]))
    assert np.allclose(linear.time, np.array([1., 2., 3.]))


def test_positon_speed_simple():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.array([0.0, 0.5, 1.0, 0.7, 1.7])

    pos = nept.Position(data, times)
    speed = pos.speed()

    assert np.allclose(speed.data, np.array([[0.0], [0.5], [0.5], [0.3], [1.0]]))


def test_position_speed_simple_rest():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    data = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    pos = nept.Position(data, times)
    speed = pos.speed(t_smooth=None)

    assert np.allclose(speed.data, np.array([[0.0], [1.0], [0.0], [0.0], [1.0], [0.0]]))


def test_positon_speed_simple_false_smooth():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.array([0.0, 0.5, 1.0, 0.7, 1.7])

    pos = nept.Position(data, times)
    # No smoothing occurs when t_smooth > dt
    speed = pos.speed(t_smooth=0.1)

    assert np.allclose(speed.data, np.array([[0.0], [0.5], [0.5], [0.3], [1.0]]))


def test_position_speed_simple_smooth():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.array([0.0, 0.5, 1.0, 0.7, 1.7])

    pos = nept.Position(data, times)
    speed = pos.speed(t_smooth=0.3)

    assert np.allclose(speed.data, np.array([[0.00191813],
                                             [0.49808187],
                                             [0.49923275],
                                             [0.30345263],
                                             [0.99347836]]))


def test_position_speed_unequal_time():
    time = np.hstack((np.linspace(0, 10, 10), np.linspace(11, 101, 10)))
    data = np.arange(0, 20)

    position = nept.Position(data, time)
    speed = position.speed()
    run_idx = np.squeeze(speed.data) >= 0.7
    run_position = position[run_idx]

    assert np.allclose(len(run_position.x), 10)


def test_position_empty_epoch_slice():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    data = np.array([[1.0, 3.1],
                     [2.0, 2.1],
                     [3.0, 1.1],
                     [4.0, 0.1]])
    position = nept.Position(data, times)

    epochs = nept.Epoch([], [])

    sliced_position = position[epochs]

    assert sliced_position.time.size == 0


def test_position_epoch_slice():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    data = np.array([[1.0, 3.1],
                     [2.0, 2.1],
                     [3.0, 1.1],
                     [4.0, 0.1]])
    position = nept.Position(data, times)

    epochs = nept.Epoch([1.8], [3.2])

    sliced_position = position[epochs]

    assert np.allclose(sliced_position.time, np.array([2., 3.]))
    assert np.allclose(sliced_position.data, np.array([[2., 2.1], [3., 1.1]]))


def test_position_x_setter_array():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    data = np.array([[1.0, 3.1],
                     [2.0, 2.1],
                     [3.0, 1.1],
                     [4.0, 0.1]])
    position = nept.Position(data, times)
    position.x = np.array([0.0, 1.0, 2.0, 3.0])

    assert np.allclose(position.x, np.array([0.0, 1.0, 2.0, 3.0]))


def test_position_x_setter_value():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    data = np.array([[1.0, 3.1],
                     [2.0, 2.1],
                     [3.0, 1.1],
                     [4.0, 0.1]])
    position = nept.Position(data, times)
    position.x = 3.3

    assert np.allclose(position.x, np.array([3.3, 3.3, 3.3, 3.3]))


def test_position_y_setter_array():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    data = np.array([[1.0, 3.1],
                     [2.0, 2.1],
                     [3.0, 1.1],
                     [4.0, 0.1]])
    position = nept.Position(data, times)
    position.y = np.array([0.0, 1.0, 2.0, 3.0])

    assert np.allclose(position.y, np.array([0.0, 1.0, 2.0, 3.0]))


def test_position_noy_setter():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    data = np.array([[1.0],
                     [2.0],
                     [3.0],
                     [4.0]])
    position = nept.Position(data, times)

    with pytest.raises(ValueError) as excinfo:
        position.y = np.array([0.0, 1.0, 2.0, 3.0])

    assert str(excinfo.value) == "can't set 'y' of one-dimensional position"


def test_position_combine():
    position = nept.Position([[1, 1, 1], [2, 2, 2]], [0, 1, 2])
    pos = nept.Position([[8, 3, 4], [6, 8, 4]], [0.5, 1, 2.5])

    combined = position.combine(pos)

    assert np.allclose(combined.time, np.array([0.0, 0.5, 1.0, 1.0, 2.0, 2.5]))
    assert np.allclose(combined.x, np.array([1., 8., 1., 3., 1., 4.]))
    assert np.allclose(combined.y, np.array([2., 6., 2., 8., 2., 4.]))


def test_position_combine_wrong_dimension():
    position = nept.Position([[1, 1, 1], [2, 2, 2]], [0, 1, 2])
    pos = nept.Position([[8, 3, 4]], [0.5, 1, 2.5])

    with pytest.raises(ValueError) as excinfo:
        combined = position.combine(pos)

    assert str(excinfo.value) == "'pos' must be 2 dimensions"


def test_position_combine_wrong_dimension2():
    position = nept.Position([[1, 1, 1]], [0, 1, 2])
    pos = nept.Position([[8, 3, 4], [6, 8, 4]], [0.5, 1, 2.5])

    with pytest.raises(ValueError) as excinfo:
        combined = position.combine(pos)

    assert str(excinfo.value) == "'pos' must be 1 dimensions"


def test_position_combine_same_times():
    position = nept.Position([[1, 1, 1], [2, 2, 2]], [0, 1, 2])
    pos = nept.Position([[8, 3, 4], [6, 8, 4]], [0, 1, 2])

    combined = position.combine(pos)

    assert np.allclose(combined.time, np.array([0., 0., 1., 1., 2., 2.]))
    assert np.allclose(combined.x, np.array([1., 8., 1., 3., 1., 4.]))
    assert np.allclose(combined.y, np.array([2., 6., 2., 8., 2., 4.]))
