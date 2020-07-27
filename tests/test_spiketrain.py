import numpy as np
import pytest
import nept


def test_spiketrain_labels():
    spikes = [
        nept.SpikeTrain(np.array([1.0, 3.0, 5.0, 7.0, 9.0])),
        nept.SpikeTrain(np.array([1.3, 3.3, 5.3]), "check_label"),
    ]

    assert spikes[0].label == None
    assert spikes[1].label == "check_label"


def test_spiketrain_sort_times():
    spikes = [
        nept.SpikeTrain(np.array([9.0, 7.0, 5.0, 3.0, 1.0])),
        nept.SpikeTrain(np.array([1.3, 5.3, 3.3])),
    ]

    assert np.allclose(spikes[0].time, np.array([1.0, 3.0, 5.0, 7.0, 9.0]))
    assert np.allclose(spikes[1].time, np.array([1.3, 3.3, 5.3]))


def test_spiketrain_time_slice():
    spikes_a = nept.SpikeTrain(np.arange(1, 100, 5), "test")
    spikes_b = nept.SpikeTrain(np.arange(24, 62, 1), "test")
    spikes_c = nept.SpikeTrain(
        np.hstack((np.arange(0, 24, 3), np.arange(61, 100, 3))), "test"
    )

    t_start = 25.0
    t_stop = 60.0

    sliced_spikes_a = spikes_a.time_slice(t_start, t_stop)
    sliced_spikes_b = spikes_b.time_slice(t_start, t_stop)
    sliced_spikes_c = spikes_c.time_slice(t_start, t_stop)

    assert np.allclose(sliced_spikes_a.time, np.array([26, 31, 36, 41, 46, 51, 56]))
    assert np.allclose(
        sliced_spikes_b.time,
        np.array(
            [
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
            ]
        ),
    )
    assert np.allclose(sliced_spikes_c.time, np.array([]))


def test_spiketrain_time_slices():
    spikes = [
        nept.SpikeTrain(np.array([1.0, 3.0, 5.0, 7.0, 9.0])),
        nept.SpikeTrain(np.array([1.3, 3.3, 5.3])),
    ]

    starts = np.array([1.0, 7.0])
    stops = np.array([4.0, 10.0])

    sliced_spikes = [spike.time_slice(starts, stops) for spike in spikes]

    assert np.allclose(sliced_spikes[0].time, np.array([1.0, 3.0, 7.0, 9.0]))
    assert np.allclose(sliced_spikes[1].time, np.array([1.3, 3.3]))


def test_spiketrain_timeslice_single():
    spikes = nept.SpikeTrain(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "test")

    starts = 2.0
    stops = 5.0
    sliced_spikes = spikes.time_slice(starts, stops)

    assert np.allclose(sliced_spikes.time, np.array([2.0, 3.0, 4.0, 5.0]))


def test_spiketrain_timeslice_none_start():
    spikes = nept.SpikeTrain(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "test")

    starts = None
    stops = 5.0
    sliced_spikes = spikes.time_slice(starts, stops)

    assert np.allclose(sliced_spikes.time, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_spiketrain_timeslice_none_stop():
    spikes = nept.SpikeTrain(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "test")

    starts = 2.0
    stops = None
    sliced_spikes = spikes.time_slice(starts, stops)

    assert np.allclose(sliced_spikes.time, np.array([2.0, 3.0, 4.0, 5.0, 6.0]))


def test_spiketrain_timeslice_none_list_start():
    spikes = nept.SpikeTrain(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "test")

    starts = [None, 4.5]
    stops = [3.0, 5.0]
    sliced_spikes = spikes.time_slice(starts, stops)

    assert np.allclose(sliced_spikes.time, np.array([1.0, 2.0, 3.0, 5.0]))


def test_spiketrain_timeslice_none_list_stop():
    spikes = nept.SpikeTrain(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "test")

    starts = [1.5, 4.5]
    stops = [3.0, None]
    sliced_spikes = spikes.time_slice(starts, stops)

    assert np.allclose(sliced_spikes.time, np.array([2.0, 3.0, 5.0, 6.0]))


def test_spiketrain_wrong_label():
    with pytest.raises(ValueError) as excinfo:
        spikes = nept.SpikeTrain(np.array([1.0, 3.0, 5.0, 7.0, 9.0]), 1)

    assert str(excinfo.value) == "label must be a string"


def test_spiketrain_time_notvector():
    with pytest.raises(ValueError) as excinfo:
        spikes = nept.SpikeTrain(np.array([[1.0, 2.0], [2.0, 3.0]]))

    assert str(excinfo.value) == "time must be a vector"


def test_spiketrain_timeslice_uneven_startstop():
    spikes = nept.SpikeTrain(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "test")

    starts = [0.1]
    stops = [1.5, 6.0]
    with pytest.raises(ValueError) as excinfo:
        sliced_spikes = spikes.time_slice(starts, stops)

    assert str(excinfo.value) == "must have same number of start and stop times"


def test_spiketrain_n_spikes_simple():
    spikes = [
        nept.SpikeTrain(np.array([1.0, 3.0, 5.0, 7.0, 9.0])),
        nept.SpikeTrain(np.array([1.3, 3.3, 5.3])),
    ]

    assert np.allclose(spikes[0].n_spikes, 5)
    assert np.allclose(spikes[1].n_spikes, 3)


def test_spiketrain_n_spikes_empty():
    spikes = [nept.SpikeTrain(np.array([]))]

    assert np.allclose(spikes[0].n_spikes, 0)
