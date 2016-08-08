import numpy as np

import vdmlab as vdm


def test_simple_tc():
    linear = dict()
    linear['position'] = np.linspace(0, 10, 4)
    linear['time'] = np.linspace(0, 3, 4)

    times = [0.5, 1.5, 2.5]
    spikes = dict(time=[])
    for time in times:
        spikes['time'].append([time])

    tuning = vdm.tuning_curve(linear, spikes, sampling_rate=1, binsize=3, filter_type=None)

    assert np.allclose(tuning, ([1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]))


def test_simple_tc1():
    """Time spent in each bin not the same."""
    linear = dict()
    linear['position'] = np.linspace(0, 9, 4)
    linear['time'] = np.linspace(0, 3, 4)

    times = [0.0, 1.0, 2.0, 2.5]
    spikes = dict(time=[])
    for time in times:
        spikes['time'].append([time])

    tuning = vdm.tuning_curve(linear, spikes, sampling_rate=1, binsize=3, filter_type=None)

    assert np.allclose(tuning, ([1., 0., 0.], [0., 1., 0.], [0., 0., 0.5], [0., 0., 0.5]))
