import numpy as np

import vdmlab as vdm


def test_spike_counts():
    interval_times = dict()
    interval_times['start'] = [2., 7.]
    interval_times['stop'] = [6., 10.]

    spikes = dict()
    spikes['time'] = [[0., 3., 4., 8.], [0., 3., 4., 8.], [1., 7., 11.], [0., 3., 4., 8.]]

    count_matrix = vdm.spike_counts(spikes, interval_times, window=None)

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
