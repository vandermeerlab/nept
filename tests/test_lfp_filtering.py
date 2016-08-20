import numpy as np

import vdmlab as vdm


def test_swr():
    fs = 2000
    time = np.arange(0, 0.5, 1./fs)
    freq = np.ones(len(time))*100
    freq[int(len(time)*0.4):int(len(time)*0.6)] = 180
    freq[int(len(time)*0.7):int(len(time)*0.9)] = 260
    data = np.sin(2.*np.pi*freq*time)

    lfp = vdm.LFP(data, time)

    swrs = vdm.detect_swr_hilbert(lfp, fs=2000, thresh=(140.0, 250.0), power_thres=0.5, z_thres=0.4)
    assert np.allclose(swrs[0].time[0], 0.1995)
    assert np.allclose(swrs[0].time[-1], 0.3)
