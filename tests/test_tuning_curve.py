import numpy as np
import pytest
from shapely.geometry import Point, LineString
import nept


def test_simple_tc():
    linear = nept.Position(np.linspace(0, 10, 4), np.linspace(0, 3, 4))

    spikes = [
        nept.SpikeTrain(np.array([0.5]), "test"),
        nept.SpikeTrain(np.array([1.5]), "test"),
        nept.SpikeTrain(np.array([2.5]), "test"),
    ]

    edges = nept.get_bin_edges(linear, binsize=3)
    tuning, _ = nept.tuning_curve_1d(linear, spikes, edges, gaussian_std=None)

    assert np.allclose(
        tuning, ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])
    )


def test_simple_tc1():
    """Time spent in each bin not the same."""
    linear = nept.Position(np.linspace(0, 9, 4), np.linspace(0, 3, 4))

    spikes = [
        nept.SpikeTrain(np.array([0.0]), "test"),
        nept.SpikeTrain(np.array([1.0]), "test"),
        nept.SpikeTrain(np.array([2.0]), "test"),
        nept.SpikeTrain(np.array([2.5]), "test"),
    ]

    edges = nept.get_bin_edges(linear, binsize=3)
    tuning, _ = nept.tuning_curve_1d(linear, spikes, edges, gaussian_std=None)

    assert np.allclose(
        tuning, ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5])
    )


def test_tuning_curve_1d_gaussian():
    linear = nept.Position(np.linspace(0, 9, 4), np.linspace(0, 3, 4))

    spikes = [
        nept.SpikeTrain(np.array([0.0]), "test"),
        nept.SpikeTrain(np.array([1.0]), "test"),
        nept.SpikeTrain(np.array([2.0]), "test"),
        nept.SpikeTrain(np.array([2.5]), "test"),
    ]

    edges = nept.get_bin_edges(linear, binsize=3)
    tuning, _ = nept.tuning_curve_1d(linear, spikes, edges, gaussian_std=1.5)

    assert np.allclose(
        tuning,
        (
            [0.78698604, 0.10650698, 0.0],
            [0.10650698, 0.78698604, 0.10650698],
            [0.0, 0.05325349, 0.39349302],
            [0.0, 0.05325349, 0.39349302],
        ),
    )


def test_tuning_curve_1d_with_2d_position():
    position = nept.Position(
        np.hstack(
            [
                np.array([2, 4, 6, 8])[..., np.newaxis],
                np.array([7, 5, 3, 1])[..., np.newaxis],
            ]
        ),
        np.array([0.0, 1.0, 2.0, 3.0]),
    )
    spikes = [nept.SpikeTrain(np.array([0.0, 3.0, 3.0, 3.0]), "test")]

    with pytest.raises(ValueError) as excinfo:
        nept.tuning_curve_1d(position, spikes, edges=None)

    assert str(excinfo.value) == "position must be linear"


def test_get_bin_edges_2d_position():
    position = nept.Position(
        np.hstack(
            [
                np.array([2, 4, 6, 8])[..., np.newaxis],
                np.array([7, 5, 3, 1])[..., np.newaxis],
            ]
        ),
        np.array([0.0, 1.0, 2.0, 3.0]),
    )
    binsize = 2

    with pytest.raises(ValueError) as excinfo:
        nept.get_bin_edges(position, binsize=binsize)

    assert str(excinfo.value) == "position must be linear"


def test_linearize():
    t_start = 1
    t_stop = 6

    xy = np.array(([1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]))
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    pos = nept.Position(xy, time)

    trajectory = [[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]]
    line = LineString(trajectory)

    sliced_pos = pos[t_start:t_stop]

    linear = sliced_pos.linearize(line)

    assert np.allclose(
        linear.x, np.array([2.82842712, 4.24264069, 5.65685425, 7.07106781])
    )
    assert np.allclose(linear.time, [1, 2, 3, 4])


def test_tuning_curve_2d():
    position = nept.Position(
        np.hstack(
            [
                np.array([2, 4, 6, 8])[..., np.newaxis],
                np.array([7, 5, 3, 1])[..., np.newaxis],
            ]
        ),
        np.array([0.0, 1.0, 2.0, 3.0]),
    )

    binsize = 2
    xedges = np.arange(position.x.min(), position.x.max() + binsize, binsize)
    yedges = np.arange(position.y.min(), position.y.max() + binsize, binsize)

    spikes = [nept.SpikeTrain(np.array([0.0, 3.0, 3.0, 3.0]), "test")]

    tuning_curves = nept.tuning_curve_2d(position, spikes, xedges, yedges)

    assert np.allclose(
        tuning_curves[~np.isnan(tuning_curves)], np.array([3.0, 0.0, 1.0, 0.0])
    )


def test_tuning_curve_2d_gaussian():
    position = nept.Position(
        np.hstack(
            [
                np.array([2, 4, 6, 8])[..., np.newaxis],
                np.array([9, 5, 3, 1])[..., np.newaxis],
            ]
        ),
        np.array([0.0, 1.0, 2.0, 3.0]),
    )

    binsize = 2
    xedges = np.arange(position.x.min(), position.x.max() + binsize, binsize)
    yedges = np.arange(position.y.min(), position.y.max() + binsize, binsize)

    spikes = [nept.SpikeTrain(np.array([0.0, 3.0, 3.0, 3.0]), "test")]

    tuning_curves = nept.tuning_curve_2d(
        position, spikes, xedges, yedges, gaussian_std=0.4
    )

    assert np.allclose(
        tuning_curves[~np.isnan(tuning_curves)], np.array([3.0, 0.0, 0.0, 1.0])
    )


def test_get_occupancy():
    position = nept.Position(
        np.hstack(
            [
                np.array([2, 4, 6, 8])[..., np.newaxis],
                np.array([7, 5, 3, 1])[..., np.newaxis],
            ]
        ),
        np.array([0.0, 1.0, 2.0, 3.0]),
    )

    binsize = 2
    xedges = np.arange(position.x.min(), position.x.max() + binsize, binsize)
    yedges = np.arange(position.y.min(), position.y.max() + binsize, binsize)

    occupancy = nept.get_occupancy(position, yedges, xedges)

    assert np.allclose(
        occupancy, np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    )
