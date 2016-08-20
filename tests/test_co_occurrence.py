import numpy as np

import vdmlab as vdm


def test_spike_counts():
    intervals = np.array([[2., 7.], [6., 10.]])

    spikes = [vdm.SpikeTrain(np.array([0., 3., 4., 8.]), 'test'),
              vdm.SpikeTrain(np.array([0., 3., 4., 8.]), 'test'),
              vdm.SpikeTrain(np.array([1., 7., 11.]), 'test'),
              vdm.SpikeTrain(np.array([0., 3., 4., 8.]), 'test')]

    count_matrix = vdm.spike_counts(spikes, intervals)

    assert np.allclose(np.mean(count_matrix), 1.25)
    assert np.allclose(count_matrix[0], count_matrix[1], count_matrix[3])


def test_compute_cooccur():
    count_matrix = np.array([[3., 2., 1.],
                             [0., 0., 0.],
                             [1., 0., 2.],
                             [0., 2., 0.]])

    prob_active, prob_expected, prob_observed, prob_zscore = vdm.compute_cooccur(count_matrix, num_shuffles=100)

    assert np.allclose(prob_active, np.array([1., 0., 2. / 3., 1. / 3.]))
    assert np.allclose(prob_expected, np.array([0., 2. / 3., 0., 1. / 3., 0., (2. / 3. * 1. / 3.)]))
    assert np.allclose(prob_observed, np.array([0., 2. / 3., 0., 1. / 3., 0., 0.]))
