import numpy as np
from shapely.geometry import Point, LineString

import vdmlab as vdm


def test_simple_tc():
    linear = dict()
    linear['position'] = np.linspace(0, 10, 4)
    linear['time'] = np.linspace(0, 3, 4)

    spikes = dict()
    spikes['time'] = np.array([[0.5], [1.5], [2.5]])

    tuning = vdm.tuning_curve(linear, spikes['time'], binsize=3, sampling_rate=1., gaussian_std=None)

    assert np.allclose(tuning, ([1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]))


def test_simple_tc1():
    """Time spent in each bin not the same."""
    linear = dict()
    linear['position'] = np.linspace(0, 9, 4)
    linear['time'] = np.linspace(0, 3, 4)

    spikes = dict()
    spikes['time'] = np.array([[0.0], [1.0], [2.0], [2.5]])

    tuning = vdm.tuning_curve(linear, spikes['time'], binsize=3, sampling_rate=1., gaussian_std=None)

    assert np.allclose(tuning, ([1., 0., 0.], [0., 1., 0.], [0., 0., 0.5], [0., 0., 0.5]))


def test_linearize():
    trial_start = 1.0
    trial_stop = 6.0

    pos = dict()
    pos['time'] = np.arange(0, 10, 1)
    pos['x'] = np.arange(1, 11, 1)
    pos['y'] = np.arange(1, 11, 1)

    trajectory = [[0., 0.], [5., 5.], [10., 10.]]
    line = LineString(trajectory)

    linear = vdm.linear_trajectory(pos, line, trial_start, trial_stop)

    assert np.allclose(linear['position'], [2.82842712, 4.24264069, 5.65685425, 7.07106781, 8.48528137])
    assert np.allclose(linear['time'], [1, 2, 3, 4, 5])


def test_tuning_curve_2d():
    pos = dict()
    pos['x'] = np.array([2, 4, 6, 8])
    pos['y'] = np.array([7, 5, 3, 1])
    pos['time'] = np.array([0., 1., 2., 3.])

    binsize = 2
    xedges = np.arange(pos['x'].min(), pos['x'].max()+binsize, binsize)
    yedges = np.arange(pos['y'].min(), pos['y'].max()+binsize, binsize)

    spikes = dict()
    spikes['time'] = np.array([[0., 3., 3., 3.]])

    tuning_curves = vdm.tuning_curve_2d(spikes, pos, xedges, yedges, sampling_rate=1.)

    assert np.allclose(tuning_curves, [np.array([[0., 0., 3.], [0., 0., 0.], [1., 0., 0.]])])
