import numpy as np
import nept


def test_swr():
    fs = 2000
    time = np.arange(0, 0.5, 1./fs)
    freq = np.ones(len(time))*100
    freq[int(len(time)*0.4):int(len(time)*0.6)] = 180
    freq[int(len(time)*0.7):int(len(time)*0.9)] = 260
    data = np.sin(2.*np.pi*freq*time)

    lfp = nept.LocalFieldPotential(data, time)

    swrs = nept.detect_swr_hilbert(lfp, fs=2000, thresh=(140.0, 250.0), z_thresh=0.4,
                                   merge_thresh=0.02, min_length=0.01)
    assert swrs.start == 0.19950000000000001
    assert swrs.stop == 0.30049999999999999


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
