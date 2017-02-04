import numpy as np
import pytest

import vdmlab as vdm


def test_decode_location():
    likelihood = np.array([[0.1, 0.8, 0.1],
                           [0.4, 0.3, 0.3],
                           [0.15, 0.15, 0.7]])

    pos_centers = np.array([[1.], [2.], [3.]])
    time_centers = np.array([0., 1., 2.])
    decoded = vdm.decode_location(likelihood, pos_centers, time_centers)

    assert np.allclose(decoded.x, np.array([2., 1., 3.]))
    assert np.allclose(decoded.time, np.array([0., 1., 2.]))


def test_decode_location_equal():
    likelihood = np.array([[0.5, 0.5, 0.],
                           [0., 0.5, 0.5],
                           [0.5, 0., 0.5]])
    pos_centers = np.array([[1.], [2.], [3.]])
    time_centers = np.array([0., 1., 2.])
    decoded = vdm.decode_location(likelihood, pos_centers, time_centers)

    assert np.allclose(decoded.x, np.array([1., 2., 1.]))
    assert np.allclose(decoded.time, np.array([0., 1., 2.]))


def test_remove_teleports():
    decoded = vdm.Position(np.array([1., 1.5, 2., 3., 15.5, 17., 21., 22., 23.]),
                           np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.]))

    decoded_sequences = vdm.remove_teleports(decoded, speed_thresh=4, min_length=3)

    assert np.allclose(decoded_sequences.starts, np.array([0., 6.]))
    assert np.allclose(decoded_sequences.stops, np.array([3., 8.]))


def test_filter_jumps_empty():
    decoded = vdm.Position(np.array([10., 20., 30., 40.]), np.array([0., 1., 2., 3.]))
    with pytest.raises(ValueError) as excinfo:
        decoded_sequences = vdm.remove_teleports(decoded, speed_thresh=9, min_length=3)
    assert str(excinfo.value) == "resulted in all position samples removed. Adjust min_length or speed_thresh."
