import numpy as np

import vdmlab as vdm


def test_swr():
    fs = 2000
    time = np.arange(0, 0.5, 1./fs)
    freq = np.ones(len(time))*100
    freq[int(len(time)*0.4):int(len(time)*0.6)] = 180
    freq[int(len(time)*0.7):int(len(time)*0.9)] = 260
    data = np.sin(2.*np.pi*freq*time)

    lfp = vdm.LocalFieldPotential(data, time)

    swrs = vdm.detect_swr_hilbert(lfp, fs=2000, thresh=(140.0, 250.0), power_thresh=0.5, z_thresh=0.4)
    assert np.allclose(swrs.start, 0.19950000000000001)
    assert np.allclose(swrs.stop, 0.30049999999999999)
