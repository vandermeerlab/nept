import numpy as np
import pytest
import nept


def test_analogsignal_time_slice_1d():
    data = np.array([9., 7., 5., 3., 1.])
    time = np.array([0., 1., 2., 3., 4.])

    analogsignal = nept.AnalogSignal(data, time)

    start = 1.
    stop = 3.
    sliced_analogsignal = analogsignal.time_slice(start, stop)

    assert np.allclose(sliced_analogsignal.data, np.array([[7.], [5.], [3.]]))
    assert np.allclose(sliced_analogsignal.time, np.array([1., 2., 3.]))


def test_analogsignal_time_slices_1d():
    data = np.array([9., 7., 5., 3., 1.])
    time = np.array([0., 1., 2., 3., 4.])

    analogsignal = nept.AnalogSignal(data, time)

    starts = np.array([1., 3.])
    stops = np.array([1.5, 4.])
    sliced_analogsignal = analogsignal.time_slice(starts, stops)

    assert np.allclose(sliced_analogsignal.data, np.array([[7.], [3.], [1.]]))
    assert np.allclose(sliced_analogsignal.time, np.array([1., 3., 4.]))


def test_analogsignal_time_slice_2d():
    x = np.array([9., 7., 5., 3., 1.])
    y = np.array([9., 7., 5., 3., 1.])
    time = np.array([0., 1., 2., 3., 4.])

    data = np.hstack([np.array(x)[..., np.newaxis], np.array(y)[..., np.newaxis]])

    analogsignal = nept.AnalogSignal(data, time)

    start = 1.
    stop = 3.
    sliced_analogsignal = analogsignal.time_slice(start, stop)

    assert np.allclose(sliced_analogsignal.data, np.array([[7., 7.], [5., 5.], [3., 3.]]))
    assert np.allclose(sliced_analogsignal.time, np.array([1., 2., 3.]))


def test_analogsignal_time_slices_2d():
    x = np.array([9., 7., 5., 3., 1.])
    y = np.array([9., 7., 5., 3., 1.])
    time = np.array([0., 1., 2., 3., 4.])

    data = np.hstack([np.array(x)[..., np.newaxis], np.array(y)[..., np.newaxis]])

    analogsignal = nept.AnalogSignal(data, time)

    starts = np.array([1., 3.])
    stops = np.array([1.5, 4.])
    sliced_analogsignal = analogsignal.time_slice(starts, stops)

    assert np.allclose(sliced_analogsignal.data, np.array([[7., 7.], [3., 3.], [1., 1.]]))
    assert np.allclose(sliced_analogsignal.time, np.array([1., 3., 4.]))
