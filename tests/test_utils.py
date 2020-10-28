import nept
import numpy as np
import pytest

toy_array = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


def test_find_nearest_idx():
    assert nept.find_nearest_idx(toy_array, 13) == 3
    assert nept.find_nearest_idx(toy_array, 11.49) == 1
    assert nept.find_nearest_idx(toy_array, 11.51) == 2
    assert nept.find_nearest_idx(toy_array, 25) == 10
    assert nept.find_nearest_idx(toy_array, 1) == 0


def test_find_nearest_indices():
    assert np.allclose(
        nept.find_nearest_indices(toy_array, np.array([13.2])), np.array([3])
    )
    assert np.allclose(
        nept.find_nearest_indices(toy_array, np.array([10, 20])), np.array([0, 10])
    )


def test_sort_idx():
    linear = nept.Position(np.linspace(0, 10, 4), np.linspace(0, 3, 4))

    spikes = [
        nept.SpikeTrain(np.array([1.5]), "test"),
        nept.SpikeTrain(np.array([0.5]), "test"),
        nept.SpikeTrain(np.array([2.5]), "test"),
    ]

    edges = nept.get_bin_edges(linear, binsize=3)
    tuning, _ = nept.tuning_curve_1d(linear, spikes, edges, gaussian_std=None)
    sort_idx = nept.get_sort_idx(tuning)

    assert np.allclose(sort_idx, [1, 0, 2])


def test_sort_idx1():
    linear = nept.Position(np.linspace(0, 9, 4), np.linspace(0, 3, 4))

    spikes = [
        nept.SpikeTrain(np.array([2.5]), "test"),
        nept.SpikeTrain(np.array([0.0]), "test"),
        nept.SpikeTrain(np.array([2.0]), "test"),
        nept.SpikeTrain(np.array([1.0]), "test"),
    ]

    edges = nept.get_bin_edges(linear, binsize=3)
    tuning, _ = nept.tuning_curve_1d(linear, spikes, edges, gaussian_std=None)
    sort_idx = nept.get_sort_idx(tuning)

    assert np.allclose(sort_idx, [1, 3, 0, 2])


def test_multi_in_epochs_one():
    epochs = nept.Epoch([1.0, 4.0, 6.0], [2.0, 5.0, 7.0])

    spikes = [
        nept.SpikeTrain(np.array([6.7])),
        nept.SpikeTrain(np.array([1.1, 6.5])),
        nept.SpikeTrain(np.array([1.3, 4.1])),
        nept.SpikeTrain(np.array([1.7, 4.3])),
    ]

    min_involved = 3
    multi_epochs = nept.find_multi_in_epochs(spikes, epochs, min_involved)

    assert np.allclose(multi_epochs.starts, np.array([1.0]))
    assert np.allclose(multi_epochs.stops, np.array([2.0]))


def test_multi_in_epochs_edge():
    epochs = nept.Epoch([1.0, 4.0, 6.0], [2.0, 5.0, 7.0])

    spikes = [
        nept.SpikeTrain(np.array([6.7])),
        nept.SpikeTrain(np.array([2.0, 6.5])),
        nept.SpikeTrain(np.array([2.0, 4.1])),
        nept.SpikeTrain(np.array([2.0, 4.3])),
    ]

    min_involved = 3
    multi_epochs = nept.find_multi_in_epochs(spikes, epochs, min_involved)

    assert np.allclose(multi_epochs.starts, np.array([1.0]))
    assert np.allclose(multi_epochs.stops, np.array([2.0]))


def test_multi_in_epochs_mult():
    epochs = nept.Epoch([1.0, 4.0, 6.0], [2.0, 5.0, 7.0])

    spikes = [
        nept.SpikeTrain(np.array([1.1, 6.5])),
        nept.SpikeTrain(np.array([1.3, 4.1])),
        nept.SpikeTrain(np.array([1.7, 4.3])),
    ]

    min_involved = 2
    multi_epochs = nept.find_multi_in_epochs(spikes, epochs, min_involved)

    assert np.allclose(multi_epochs.starts, np.array([1.0, 4.0]))
    assert np.allclose(multi_epochs.stops, np.array([2.0, 5.0]))


def test_multi_in_epoch_none():
    epochs = nept.Epoch([0.0], [1.0])

    spikes = [
        nept.SpikeTrain(np.array([1.1, 6.5])),
        nept.SpikeTrain(np.array([1.3, 4.1])),
        nept.SpikeTrain(np.array([1.7, 4.3])),
    ]

    min_involved = 2
    multi_epochs = nept.find_multi_in_epochs(spikes, epochs, min_involved)

    assert np.allclose(multi_epochs.starts, np.array([]))
    assert np.allclose(multi_epochs.stops, np.array([]))


def test_get_xyedges_mult():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([[1.0, 1.1], [5.0, 5.1], [10.0, 10.1]])

    position = nept.Position(data, times)

    xedges, yedges = nept.get_xyedges(position, binsize=3)

    assert np.allclose(xedges, np.array([1.0, 4.0, 7.0, 10.0]))
    assert np.allclose(yedges, np.array([1.1, 4.1, 7.1, 10.1]))


def test_get_xyedges_one_full():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([[1.0, 1.1], [5.0, 5.1], [10.0, 10.1]])

    position = nept.Position(data, times)
    position = nept.Position(data, times)

    xedges, yedges = nept.get_xyedges(position, binsize=10)

    assert np.allclose(xedges, np.array([1.0, 11.0]))
    assert np.allclose(yedges, np.array([1.1, 11.1]))


def test_get_xyedges_1d_position():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([1.0, 5.0, 10.0])

    position = nept.Position(data, times)

    with pytest.raises(ValueError) as excinfo:
        xedges, yedges = nept.get_xyedges(position, binsize=3)

    assert str(excinfo.value) == "position must be 2-dimensional"


def test_bin_spikes():
    spikes = [nept.SpikeTrain([0.8, 1.1, 1.2, 1.2, 2.1, 3.1])]

    counts = nept.bin_spikes(
        spikes,
        0.0,
        4.0,
        dt=0.5,
        window=2.0,
        gaussian_std=None,
    )

    assert np.allclose(
        counts.data,
        [[0.625], [1.0], [1.125], [1.125], [0.75], [0.5], [0.375]],
    )


def test_bin_spikes_gaussian():
    spikes = [nept.SpikeTrain([0.8, 1.1, 1.2, 1.2, 2.1, 3.1])]

    counts = nept.bin_spikes(
        spikes,
        0.0,
        10.0,
        dt=0.5,
        window=2.0,
        gaussian_std=0.51,
    )

    assert np.allclose(
        counts.data,
        [
            [5.56643201e-01],
            [8.82737318e-01],
            [1.03531768e00],
            [9.84563945e-01],
            [7.78806879e-01],
            [5.52981161e-01],
            [3.84158508e-01],
            [2.51405526e-01],
            [1.33534061e-01],
            [4.65665762e-02],
            [8.51168517e-03],
            [6.91574971e-04],
            [2.23758525e-05],
            [0.00000000e00],
            [0.00000000e00],
            [0.00000000e00],
            [0.00000000e00],
            [0.00000000e00],
            [0.00000000e00],
        ],
    )


def test_bin_spikes_gaussian_even():
    spikes = [nept.SpikeTrain([0.8, 1.1, 1.2, 1.2, 2.1, 3.1])]

    counts = nept.bin_spikes(
        spikes,
        0.0,
        10.0,
        dt=0.5,
        window=2.0,
        gaussian_std=0.5,
    )

    assert np.allclose(
        counts.data,
        [
            [5.57137350e-01],
            [8.86524329e-01],
            [1.03950464e00],
            [9.87959874e-01],
            [7.79038092e-01],
            [5.51601069e-01],
            [3.83461026e-01],
            [2.51191610e-01],
            [1.32923772e-01],
            [4.54730760e-02],
            [7.90704282e-03],
            [5.87440359e-04],
            [1.67288281e-05],
            [0.00000000e00],
            [0.00000000e00],
            [0.00000000e00],
            [0.00000000e00],
            [0.00000000e00],
            [0.00000000e00],
        ],
    )


def test_bin_spikes_mult_neurons():
    spikes = [
        nept.SpikeTrain([0.8, 1.1, 1.2, 1.2, 2.1, 3.1]),
        nept.SpikeTrain([0.8, 1.1, 1.2, 1.2, 2.1, 3.1]),
    ]

    counts = nept.bin_spikes(spikes, 0.0, 4.0, dt=0.5, window=2, gaussian_std=None)

    assert np.allclose(
        counts.data,
        [
            [0.625, 0.625],
            [1.0, 1.0],
            [1.125, 1.125],
            [1.125, 1.125],
            [0.75, 0.75],
            [0.5, 0.5],
            [0.375, 0.375],
        ],
    )


def test_bin_spikes_mult_neurons_adjust_window():
    spikes = [
        nept.SpikeTrain([0.8, 1.1, 1.2, 1.2, 2.1, 3.1]),
        nept.SpikeTrain([0.8, 1.1, 1.2, 1.2, 2.1, 3.1]),
    ]

    counts = nept.bin_spikes(spikes, 0.0, 4.0, dt=0.5, window=2.5, gaussian_std=None)

    assert np.allclose(
        counts.data,
        np.array(
            [
                [0.8, 0.8],
                [0.8, 0.8],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [0.4, 0.4],
                [0.4, 0.4],
            ]
        ),
    )


def test_cartesian():
    xcenters = np.array([0.0, 4.0, 8.0])
    ycenters = np.array([0.0, 2.0, 4.0])
    xy_centers = nept.cartesian(xcenters, ycenters)

    assert np.allclose(
        xy_centers,
        np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [8.0, 0.0],
                [0.0, 2.0],
                [4.0, 2.0],
                [8.0, 2.0],
                [0.0, 4.0],
                [4.0, 4.0],
                [8.0, 4.0],
            ]
        ),
    )


def test_consecutive():
    array = np.array([0, 3, 4, 5, 9, 12, 13, 14])

    groups = nept.consecutive(array, stepsize=1)

    assert len(groups) == 4
    assert np.allclose(groups[0], [0])
    assert np.allclose(groups[1], [3, 4, 5])
    assert np.allclose(groups[2], [9])
    assert np.allclose(groups[3], [12, 13, 14])


def test_consecutive_equal_stepsize():
    array = np.arange(0, 10, 1)

    groups = nept.consecutive(array, stepsize=1)

    assert np.all(groups == np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]))


def test_consecutive_all_split():
    array = np.arange(0, 10, 1)

    groups = nept.consecutive(array, stepsize=0.9)

    assert np.all(
        groups == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    )


def test_get_edges_no_lastbin():
    edges = nept.get_edges(0.0, 4.1, binsize=0.5, lastbin=False)

    assert np.allclose(edges, np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]))


def test_get_edges_simple():
    edges = nept.get_edges(0.0, 4.1, binsize=0.5)

    assert np.allclose(
        edges, np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.1])
    )


def test_perievent_slice_simple():
    data = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    analogsignal = nept.AnalogSignal(data, time)

    events = np.array([1.0])
    perievent_lfp = nept.perievent_slice(
        analogsignal, events, t_before=1.0, t_after=1.0
    )

    assert np.allclose(perievent_lfp.data, np.array([[9.0], [7.0], [5.0]]))
    assert np.allclose(perievent_lfp.time, np.array([-1.0, 0.0, 1.0]))


def test_perievent_slice_with_dt():
    data = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    analogsignal = nept.AnalogSignal(data, time)

    events = np.array([1.0])
    perievent_lfp = nept.perievent_slice(
        analogsignal, events, t_before=1.0, t_after=1.0, dt=0.5
    )

    assert np.allclose(
        perievent_lfp.data, np.array([[9.0], [8.0], [7.0], [6.0], [5.0]])
    )
    assert np.allclose(perievent_lfp.time, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))


def test_perievent_slice_2d():
    x = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    y = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    data = np.hstack([np.array(x)[..., np.newaxis], np.array(y)[..., np.newaxis]])
    analogsignal = nept.AnalogSignal(data, time)

    events = np.array([1.0])

    with pytest.raises(ValueError) as excinfo:
        perievent_lfp = nept.perievent_slice(
            analogsignal, events, t_before=1.0, t_after=1.0
        )

    assert str(excinfo.value) == "AnalogSignal must be 1D."


def test_rest_threshold_simple():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.array([0.0, 0.5, 1.0, 0.7, 1.7, 2.0])

    position = nept.Position(data, times)

    rest_epoch = nept.rest_threshold(position, thresh=0.4, t_smooth=None)

    assert np.allclose(rest_epoch.starts, np.array([2.0, 4.0]))
    assert np.allclose(rest_epoch.stops, np.array([3.0, 5.0]))


def test_rest_threshold_gap():
    times = np.array([0.5, 1.0, 1.5, 4.5, 5.0, 5.5, 6.0, 6.5, 9.0, 9.5, 10.0])
    data = np.array([0.0, 0.5, 1.5, 10.0, 9.9, 9.8, 8.5, 7.0, 0.0, 0.5, 1.0])

    position = nept.Position(data, times)

    rest_epoch = nept.rest_threshold(position, thresh=0.4, t_smooth=None)

    assert np.allclose(rest_epoch.starts, np.array([4.5]))
    assert np.allclose(rest_epoch.stops, np.array([5.5]))


def test_run_threshold_simple():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.array([0.0, 0.5, 1.0, 0.7, 1.7, 2.0])

    position = nept.Position(data, times)

    run_epoch = nept.run_threshold(position, thresh=0.4, t_smooth=None)

    assert np.allclose(run_epoch.starts, np.array([0.0, 3.0]))
    assert np.allclose(run_epoch.stops, np.array([2.0, 4.0]))


def test_run_threshold_gap():
    times = np.array([0.5, 1.0, 1.5, 4.5, 5.0, 5.5, 6.0, 6.5, 9.0, 9.5, 10.0])
    data = np.array([0.0, 0.5, 1.5, 10.0, 9.9, 9.8, 8.5, 7.0, 0.0, 0.5, 1.0])

    position = nept.Position(data, times)

    run_epoch = nept.run_threshold(position, thresh=0.4, t_smooth=None)

    assert np.allclose(run_epoch.starts, np.array([0.5, 5.5, 9.0]))
    assert np.allclose(run_epoch.stops, np.array([1.5, 6.5, 10.0]))


def test_gaussian_filter_unchanged():
    signal = np.array([1.0, 3.0, 7.0])
    std = 0.1

    filtered_signal = nept.gaussian_filter(signal, std, dt=1.0)

    assert filtered_signal.all() == signal.all()
