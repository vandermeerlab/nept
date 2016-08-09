import numpy as np

import vdmlab as vdm


tuning = [np.array([0., 1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0.]),
              np.array([0., 1., 2., 3., 3., 1., 0., 0., 0., 0., 0., 0.]),
              np.array([0., 3., 3., 3., 0., 2., 0., 0., 3., 2., 0., 0.]),
              np.array([0., 1., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0.])]


def test_find_fields():
    with_fields = vdm.find_fields(tuning, hz_thresh=2, min_length=1, max_length=5, max_mean_firing=8)

    assert np.allclose(len(with_fields), 3)
    assert np.allclose(with_fields[1], [np.array([3, 4])])
    assert np.allclose(with_fields[2][0], np.array([1, 2, 3]))
    assert np.allclose(with_fields[2][1], np.array([8]))
    assert np.allclose(with_fields[3], [np.array([7])])

def test_get_single_fields():
    with_fields = vdm.find_fields(tuning, hz_thresh=2, min_length=1, max_length=5, max_mean_firing=8)
    single_fields = vdm.get_single_field(with_fields)

    assert np.allclose(len(single_fields), 2)
    assert np.allclose(single_fields[1], [np.array([3, 4])])
    assert np.allclose(single_fields[3], [np.array([7])])
