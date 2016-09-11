import numpy as np

import vdmlab as vdm


def test_spike_counts():
    intervals = vdm.Epoch(np.array([[2., 6.], [7., 10.]]))

    spikes = [vdm.SpikeTrain(np.array([0., 3., 4., 8.]), 'test'),
              vdm.SpikeTrain(np.array([0., 3., 4., 8.]), 'test'),
              vdm.SpikeTrain(np.array([1., 7., 11.])),
              vdm.SpikeTrain(np.array([0., 3., 4., 8.]), 'test')]

    count_matrix = vdm.spike_counts(spikes, intervals)

    assert np.allclose(np.mean(count_matrix), 1.25)
    assert np.allclose(count_matrix[0], count_matrix[1], count_matrix[3])


def test_compute_cooccur():
    count_matrix = np.array([[3., 2., 1.],
                             [0., 0., 0.],
                             [1., 0., 2.],
                             [0., 2., 0.]])

    tetrode_mask = np.array([[True, True, False, False],
                             [True, True, False, False],
                             [False, False, True, False],
                             [False, False, False, True]])

    prob = vdm.compute_cooccur(count_matrix, tetrode_mask, num_shuffles=100)

    assert np.allclose(prob['active'], np.array([1., 0., 2. / 3., 1. / 3.]))
    assert np.isnan(prob['expected'][0])
    assert np.allclose(prob['expected'][1:], np.array([2. / 3., 0., 1. / 3., 0., (2. / 3. * 1. / 3.)]))
    assert np.isnan(prob['observed'][0])
    assert np.allclose(prob['observed'][1:], np.array([2. / 3., 0., 1. / 3., 0., 0.]))


def test_get_tetrode_mask():
    spikes = [vdm.SpikeTrain([1., 2., 3.], 'a'),
              vdm.SpikeTrain([1., 2., 3.], 'b'),
              vdm.SpikeTrain([1., 2., 3.], 'a'),
              vdm.SpikeTrain([1., 2., 3.], 'b'),
              vdm.SpikeTrain([1., 2., 3.], 'c')]

    tetrode_mask = vdm.get_tetrode_mask(spikes)

    true_compare = np.array([[1, 0, 1, 0, 0],
                             [0, 1, 0, 1, 0],
                             [1, 0, 1, 0, 0],
                             [0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1]])

    assert np.all(true_compare == tetrode_mask)
