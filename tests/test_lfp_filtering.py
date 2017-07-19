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

    swrs = nept.detect_swr_hilbert(lfp, fs=2000, thresh=(140.0, 250.0), power_thresh=0.5, z_thresh=0.4)
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
    data = np.array([9., 7., 5., 3., 1.])
    time = np.array([0., 1., 2., 3., 4.])

    analogsignal = nept.AnalogSignal(data, time)

    events = np.array([1.])
    perievent_lfp = nept.perievent_slice(analogsignal, events, t_before=1., t_after=1.)

    window = 2
    fs = 1

    freq, psd = nept.mean_psd(perievent_lfp, window, fs)

    assert np.allclose(freq, np.array([0., 0.25, 0.5]))
    assert np.allclose(psd, np.array([72., 74., 2.]))
