import numpy as np
import pytest

import vdmlab as vdm


def test_bayesian_prob_smalltc():
    tuning_curve = np.array([[0., 0., 1.]])
    counts = np.array([[10.]])
    binsize = 1.0

    likelihood = vdm.bayesian_prob(counts, tuning_curve, binsize)

    assert np.sum(np.isnan(likelihood)) == likelihood.size


def test_bayesian_prob_onetime():
    tuning_curve = np.array([[0., 0., 2.]])
    counts = np.array([[10.]])
    binsize = 1.0

    likelihood = vdm.bayesian_prob(counts, tuning_curve, binsize)

    assert np.allclose(likelihood[0][2], 1.0)


def test_bayesian_prob_nospike():
    tuning_curve = np.array([[2., 0., 0.]])
    counts = np.array([[0.]])
    binsize = 1.0

    likelihood = vdm.bayesian_prob(counts, tuning_curve, binsize)

    assert np.sum(np.isnan(likelihood)) == likelihood.size


def test_bayesian_prob_multtc():
    tuning_curve = np.array([[2., 0., 0.],
                             [0., 5., 0.],
                             [0., 0., 5.]])
    counts = np.array([[0.],
                       [4.],
                       [2.]])

    binsize = 1.0
    likelihood = vdm.bayesian_prob(counts, tuning_curve, binsize)

    assert np.allclose(likelihood, np.array([[0.02997459, 0.93271674, 0.03730867]]))


def test_bayesian_prob_emptytcbin():
    tuning_curve = np.array([[0., 1., 0.],
                             [0., 5., 0.],
                             [0., 0., 5.]])
    counts = np.array([[0.],
                       [2.],
                       [2.]])
    binsize = 1.0
    likelihood = vdm.bayesian_prob(counts, tuning_curve, binsize)

    assert np.isnan(likelihood[0][0])
    assert np.allclose(likelihood[0][1], 0.5)
    assert np.allclose(likelihood[0][2], 0.5)


def test_bayesian_prob_onepos():
    tuning_curve = np.array([[10.]])
    counts = np.array([[0., 2., 4.]])
    binsize = 1.0
    likelihood = vdm.bayesian_prob(counts, tuning_curve, binsize)

    assert np.isnan(likelihood[0][0])
    assert np.allclose(likelihood[1][0], 1.0)
    assert np.allclose(likelihood[2][0], 1.0)


def test_bayesian_prob_multtimepos():
    tuning_curve = np.array([[3., 0., 0.]])
    counts = np.array([[0., 2., 4.]])
    binsize = 1.0
    likelihood = vdm.bayesian_prob(counts, tuning_curve, binsize)

    assert np.sum(np.isnan(likelihood[0])) == 3
    assert np.allclose(likelihood[1][0], 1.0)
    assert np.sum(np.isnan(likelihood[1])) == 2
    assert np.allclose(likelihood[2][0], 1.0)
    assert np.sum(np.isnan(likelihood[2])) == 2


def test_bayesian_prob_multneurons():
    tuning_curve = np.array([[2., 0., 0.],
                             [0., 5., 0.],
                             [0., 0., 5.]])
    counts = np.array([[0., 8, 0.],
                       [4., 0., 1.],
                       [0., 1., 3.]])
    binsize = 1.0
    likelihood = vdm.bayesian_prob(counts, tuning_curve, binsize)

    assert np.allclose(likelihood[0], np.array([0.0310880460, 0.967364171, 0.00154778267]))
    assert np.allclose(likelihood[1], np.array([0.998834476, 0.000194254064, 0.000971270319]))
    assert np.allclose(likelihood[2], np.array([0.133827265, 0.0333143360, 0.832858399]))


def test_decode_location():
    likelihood = np.array([[0.1, 0.8, 0.1],
                           [0.4, 0.3, 0.3],
                           [0.15, 0.15, 0.7]])

    linear = vdm.Position(np.array([1., 2., 3.]), np.array([0., 1., 2.]))
    decoded = vdm.decode_location(likelihood, linear)

    assert np.allclose(decoded.x, np.array([2., 1., 3.]))


def test_decode_location_equal():
    likelihood = np.array([[0.5, 0.5, 0.],
                           [0., 0.5, 0.5],
                           [0.5, 0., 0.5]])
    linear = vdm.Position(np.array([1., 2., 3.]), np.array([0., 1., 2.]))
    decoded = vdm.decode_location(likelihood, linear)

    assert np.allclose(decoded.x, np.array([1., 2., 1.]))


def test_find_sequences():
    decoded = vdm.Position(np.array([1., 1.5, 2., 3., 15.5, 17., 21., 22., 23.]),
                           np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.]))

    decoded_sequences = vdm.find_sequences(decoded, min_length=3, max_jump=4)

    assert len(decoded_sequences) == 2
    assert np.allclose(decoded_sequences[0].time, np.array([0., 1., 2., 3.]))
    assert np.allclose(decoded_sequences[0].x, np.array([1., 1.5, 2., 3.]))
    assert np.allclose(decoded_sequences[1].time, np.array([6., 7., 8.]))
    assert np.allclose(decoded_sequences[1].x, np.array([21., 22., 23.]))


def test_find_sequences_empty():
    decoded = vdm.Position(np.array([10., 20., 30., 40.]), np.array([0., 1., 2., 3.]))
    decoded_sequences = vdm.find_sequences(decoded, min_length=3, max_jump=9)
    assert len(decoded_sequences) == 0
