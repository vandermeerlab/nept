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
