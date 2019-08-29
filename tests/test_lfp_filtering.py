import numpy as np
import pytest
import nept


@pytest.mark.parametrize("fs", [800, 2000, 5000])
@pytest.mark.parametrize("z_thresh", [0.1, 0.4, 0.9, 1.9])
@pytest.mark.parametrize("min_length", [0.09, 0.11])
@pytest.mark.parametrize("noise_std", [None, 0.2, 0.35])
def test_swr_basic(fs, z_thresh, min_length, noise_std, rng, plt):
    time = np.arange(0, 0.5, 1./fs)
    freq = np.ones(len(time)) * 100
    freq[(time >= 0.2) & (time <= 0.3)] = 180
    freq[(time >= 0.35) & (time <= 0.45)] = 260
    data = np.sin(2.*np.pi*freq*time)

    lfp = nept.LocalFieldPotential(data, time)
    if noise_std is not None:
        lfp.data += rng.normal(scale=noise_std, size=(lfp.n_samples, 1))

    swrs = nept.detect_swr_hilbert(lfp, fs=fs, z_thresh=z_thresh, merge_thresh=0., min_length=min_length)

    plt.plot(lfp.time, lfp.data)
    for start, stop in zip(swrs.starts, swrs.stops):
        plt.plot(lfp.time[nept.find_nearest_idx(lfp.time, start)], 0, "ro")
        plt.plot(lfp.time[nept.find_nearest_idx(lfp.time, stop)], 0, "ro")

    if min_length < 0.1 and z_thresh < 1.9:
        assert swrs.n_epochs == 1
        assert np.allclose(swrs.start, 0.2, atol=20./fs)
        assert np.allclose(swrs.stop, 0.3, atol=20./fs)
    else:
        assert swrs.n_epochs == 0


@pytest.mark.parametrize("merge_thresh", [0.01, 0.05, 0.1])
def test_swr_merge_thresh(merge_thresh, plt):
    fs = 2000
    time = np.arange(0, 0.5, 1./fs)
    freq = np.ones(len(time)) * 100

    # First SWR
    t1 = 0.1
    freq[(time >= t1) & (time <= t1+0.05)] = 180
    t2 = t1 + 0.05 + merge_thresh - 50./fs
    # Should still be part of first SWR
    freq[(time >= t2) & (time <= t2+0.05)] = 180
    t3 = t2 + 0.05 + merge_thresh + 50./fs
    # Second SWR
    freq[(time >= t3) & (time <= t3+0.05)] = 180

    data = np.sin(2.*np.pi*freq*time)
    lfp = nept.LocalFieldPotential(data, time)
    swrs = nept.detect_swr_hilbert(lfp, fs=fs, z_thresh=0.4, merge_thresh=merge_thresh, min_length=0.04)

    plt.plot(lfp.time, lfp.data)
    for start, stop in zip(swrs.starts, swrs.stops):
        plt.plot(lfp.time[nept.find_nearest_idx(lfp.time, start)], 0, "ro")
        plt.plot(lfp.time[nept.find_nearest_idx(lfp.time, stop)], 0, "ro")

    assert swrs.n_epochs == 2
    assert np.allclose(swrs.starts[0], t1, atol=5./fs)
    assert np.allclose(swrs.stops[0], t2+0.05, atol=5./fs)
    assert np.allclose(swrs.starts[1], t3, atol=5./fs)
    assert np.allclose(swrs.stops[1], t3+0.05, atol=5./fs)


@pytest.mark.parametrize("merge_thresh,n_swrs", [(0.03, 3), (0.08, 2), (0.13, 1)])
def test_swr_merge_thresh_fixed_gaps(merge_thresh, n_swrs, plt):
    fs = 2000
    time = np.arange(0, 0.5, 1./fs)
    freq = np.ones(len(time)) * 100

    swr_edges = [(0.1, 0.14), (0.2, 0.35), (0.45, 0.47)]
    for start, stop in swr_edges:
        freq[(time >= start) & (time <= stop)] = 180

    data = np.sin(2.*np.pi*freq*time)
    lfp = nept.LocalFieldPotential(data, time)
    swrs = nept.detect_swr_hilbert(lfp, fs=fs, z_thresh=0.4, merge_thresh=merge_thresh, min_length=0.01)

    plt.plot(lfp.time, lfp.data)
    for start, stop in zip(swrs.starts, swrs.stops):
        plt.plot(lfp.time[nept.find_nearest_idx(lfp.time, start)], 0, "ro")
        plt.plot(lfp.time[nept.find_nearest_idx(lfp.time, stop)], 0, "ro")

    assert swrs.n_epochs == n_swrs
    if merge_thresh == 0.03:
        for i in range(3):
            assert np.allclose(swrs.starts[i], swr_edges[i][0], atol=5./fs)
            assert np.allclose(swrs.stops[i], swr_edges[i][1], atol=5./fs)
    elif merge_thresh == 0.05:
        assert np.allclose(swrs.starts[0], swr_edges[0][0], atol=5./fs)
        assert np.allclose(swrs.stops[0], swr_edges[1][1], atol=5./fs)
        assert np.allclose(swrs.starts[1], swr_edges[2][0], atol=5./fs)
        assert np.allclose(swrs.stops[1], swr_edges[2][1], atol=5./fs)
    elif merge_thresh == 0.07:
        assert np.allclose(swrs.starts[0], swr_edges[0][0], atol=5./fs)
        assert np.allclose(swrs.stops[0], swr_edges[2][1], atol=5./fs)


def test_next_regular_basic():
    regular = nept.next_regular(11)
    assert regular == 12


def test_next_regular_smaller6():
    regular = nept.next_regular(4)
    assert regular == 4


def test_next_regular_already():
    regular = nept.next_regular(16)
    assert regular == 16


def test_next_regular_p35():
    regular = nept.next_regular(9)
    assert regular == 9


def test_next_regular_p5_nomatch():
    regular = nept.next_regular(25)
    assert regular == 25


def test_next_regular_p5_match():
    regular = nept.next_regular(7)
    assert regular == 8


def test_next_regular_p5_matching():
    regular = nept.next_regular(121)
    assert regular == 125


def test_power_in_db_simple():
    power = np.array([1.0e10, 0.5, 2.0])
    db_power = nept.power_in_db(power)

    assert np.allclose(db_power, np.array([100., -3.01029996, 3.01029996]))


def test_mean_psd_simple():
    fs = 500
    dt = 1./fs
    time = np.arange(0, 2+dt, dt)

    cycles = 100
    data1 = np.sin(2 * np.pi * cycles * time)
    perievent1 = nept.AnalogSignal(data1, time)

    window = 250
    fs = 125
    freq, psd = nept.mean_psd(perievent1, window, fs)

    assert freq[np.where(psd == np.max(psd))[0][0]] == 25.0


def test_mean_csd_with_itself():
    fs = 500
    dt = 1./fs
    time = np.arange(0, 2+dt, dt)

    cycles = 100
    data1 = np.sin(2 * np.pi * cycles * time)
    perievent1 = nept.AnalogSignal(data1, time)

    window = 250
    fs = 125
    freq, csd = nept.mean_csd(perievent1, perievent1, window, fs)

    assert freq[np.where(csd == np.max(csd))[0][0]] == 25.0


def test_mean_coherence_with_itself():
    fs = 500
    dt = 1./fs
    time = np.arange(0, 2+dt, dt)

    cycles = 100
    data1 = np.sin(2 * np.pi * cycles * time)
    perievent1 = nept.AnalogSignal(data1, time)

    window = 250
    fs = 125
    freq, coherence = nept.mean_coherence(perievent1, perievent1, window, fs)

    assert np.allclose(np.ones(len(coherence)), coherence)


def test_mean_coherencegram_with_itself():
    fs = 500
    dt = 1./fs
    time = np.arange(0, 2+dt, dt)

    cycles = 100
    data1 = np.sin(2 * np.pi * cycles * time)
    perievent1 = nept.AnalogSignal(data1, time)

    window = 250
    fs = 125
    dt = 1
    timebins, freq, coherencegram = nept.mean_coherencegram(perievent1, perievent1, dt, window, fs)

    assert np.allclose(coherencegram[0], np.array([1., 0., 0.]))
