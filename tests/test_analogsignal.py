import numpy as np
import pytest
import nept


def test_analogsignal_time_slice_1d():
    data = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    analogsignal = nept.AnalogSignal(data, time)

    start = 1.0
    stop = 3.0
    sliced_analogsignal = analogsignal.time_slice(start, stop)

    assert np.allclose(sliced_analogsignal.data, np.array([[7.0], [5.0], [3.0]]))
    assert np.allclose(sliced_analogsignal.time, np.array([1.0, 2.0, 3.0]))


def test_analogsignal_time_slices_1d():
    data = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    analogsignal = nept.AnalogSignal(data, time)

    starts = np.array([1.0, 3.0])
    stops = np.array([1.5, 4.0])
    sliced_analogsignal = analogsignal.time_slice(starts, stops)

    assert np.allclose(sliced_analogsignal.data, np.array([[7.0], [3.0], [1.0]]))
    assert np.allclose(sliced_analogsignal.time, np.array([1.0, 3.0, 4.0]))


def test_analogsignal_time_slice_2d():
    x = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    y = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    data = np.hstack([np.array(x)[..., np.newaxis], np.array(y)[..., np.newaxis]])

    analogsignal = nept.AnalogSignal(data, time)

    start = 1.0
    stop = 3.0
    sliced_analogsignal = analogsignal.time_slice(start, stop)

    assert np.allclose(
        sliced_analogsignal.data, np.array([[7.0, 7.0], [5.0, 5.0], [3.0, 3.0]])
    )
    assert np.allclose(sliced_analogsignal.time, np.array([1.0, 2.0, 3.0]))


def test_analogsignal_time_slices_2d():
    x = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    y = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    data = np.hstack([np.array(x)[..., np.newaxis], np.array(y)[..., np.newaxis]])

    analogsignal = nept.AnalogSignal(data, time)

    starts = np.array([1.0, 3.0])
    stops = np.array([1.5, 4.0])
    sliced_analogsignal = analogsignal.time_slice(starts, stops)

    assert np.allclose(
        sliced_analogsignal.data, np.array([[7.0, 7.0], [3.0, 3.0], [1.0, 1.0]])
    )
    assert np.allclose(sliced_analogsignal.time, np.array([1.0, 3.0, 4.0]))


def test_analogsignal_time_slices_2d_none_start():
    x = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    y = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    data = np.hstack([np.array(x)[..., np.newaxis], np.array(y)[..., np.newaxis]])

    analogsignal = nept.AnalogSignal(data, time)

    starts = np.array([None, 3.0])
    stops = np.array([1.5, 3.5])
    sliced_analogsignal = analogsignal.time_slice(starts, stops)

    assert np.allclose(
        sliced_analogsignal.data, np.array([[9.0, 9.0], [7.0, 7.0], [3.0, 3.0]])
    )
    assert np.allclose(sliced_analogsignal.time, np.array([0.0, 1.0, 3.0]))


def test_analogsignal_time_slices_2d_none_stop():
    x = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    y = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    data = np.hstack([np.array(x)[..., np.newaxis], np.array(y)[..., np.newaxis]])

    analogsignal = nept.AnalogSignal(data, time)

    starts = np.array([0.1, 3.0])
    stops = np.array([1.5, None])
    sliced_analogsignal = analogsignal.time_slice(starts, stops)

    assert np.allclose(
        sliced_analogsignal.data, np.array([[7.0, 7.0], [3.0, 3.0], [1.0, 1.0]])
    )
    assert np.allclose(sliced_analogsignal.time, np.array([1.0, 3.0, 4.0]))


def test_analogsignal_time_slices_2d_lone_none():
    x = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    y = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    data = np.hstack([np.array(x)[..., np.newaxis], np.array(y)[..., np.newaxis]])

    analogsignal = nept.AnalogSignal(data, time)

    starts = None
    stops = 1.5
    sliced_analogsignal = analogsignal.time_slice(starts, stops)

    assert np.allclose(sliced_analogsignal.data, np.array([[9.0, 9.0], [7.0, 7.0]]))
    assert np.allclose(sliced_analogsignal.time, np.array([0.0, 1.0]))


def test_analogsignal_time_slices_2d_length_error():
    x = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    y = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    data = np.hstack([np.array(x)[..., np.newaxis], np.array(y)[..., np.newaxis]])

    analogsignal = nept.AnalogSignal(data, time)

    starts = [0.1]
    stops = [1.5, 6.0]
    with pytest.raises(ValueError) as excinfo:
        sliced_analogsignal = analogsignal.time_slice(starts, stops)

    assert str(excinfo.value) == "must have same number of start and stop times"


def test_analogsignal_time_slices_1d_length_error():
    x = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    analogsignal = nept.AnalogSignal(x, time)

    starts = [0.1]
    stops = [1.5, 6.0]
    with pytest.raises(ValueError) as excinfo:
        sliced_analogsignal = analogsignal.time_slice(starts, stops)

    assert str(excinfo.value) == "must have same number of start and stop times"


def test_analogsignal_time_slices_1d_none_start():
    x = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    analogsignal = nept.AnalogSignal(x, time)

    starts = [None, 3]
    stops = [1.5, 4.0]

    sliced_analogsignal = analogsignal.time_slice(starts, stops)

    assert np.allclose(sliced_analogsignal.data, np.array([[9.0], [7.0], [3.0], [1.0]]))
    assert np.allclose(sliced_analogsignal.time, np.array([0.0, 1.0, 3.0, 4.0]))


def test_analogsignal_time_not_vector():
    x = np.array([1.0, 2.0])
    time = np.array([[1.0, 2.0], [2.0, 3.0]])

    with pytest.raises(ValueError) as excinfo:
        analogsignal = nept.AnalogSignal(x, time)

    assert str(excinfo.value) == "time must be a vector"


def test_analogsignal_data_ndim_big():
    x = np.zeros((2, 3, 4))
    time = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError) as excinfo:
        analogsignal = nept.AnalogSignal(x, time)

    assert str(excinfo.value) == "data must be vector or 2D array"


def test_analogsignal_notempty():
    x = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    y = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    data = np.hstack([np.array(x)[..., np.newaxis], np.array(y)[..., np.newaxis]])

    analogsignal = nept.AnalogSignal(data, time)

    assert not analogsignal.isempty


def test_analogsignal_isempty():
    x = np.array([])
    time = np.array([])

    analogsignal = nept.AnalogSignal(x, time)

    assert analogsignal.isempty


def test_analogsignal_n_data_time():
    x = np.array([1.0, 2.0, 3.0])
    time = np.array([0.0, 1.0])

    with pytest.raises(ValueError) as excinfo:
        analogsignal = nept.AnalogSignal(x, time)

    assert str(excinfo.value) == "data and time should be the same length"


def test_analogsignal_mismatch():
    data = np.array([[9.0, 7.0, 5.0, 3.0], [1.0, 2.0, 2.0, 3.0]])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError) as excinfo:
        analogsignal = nept.AnalogSignal(data, time)

    assert str(excinfo.value) == "must have same number of time and data samples"
