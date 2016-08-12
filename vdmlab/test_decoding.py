import numpy as np
import pytest

import vdmlab as vdm


def test_bayesian_prob_smalltc():
    tuning_curve = np.array([[0., 0., 1.]])
    counts = np.array([[10.]])
    binsize = 1.0
    prob = vdm.bayesian_prob(counts, tuning_curve, binsize)
    assert np.sum(np.isnan(prob)) == prob.size


def test_bayesian_prob_onetime():
    tuning_curve = np.array([[0., 0., 2.]])
    counts = np.array([[10.]])
    binsize = 1.0
    prob = vdm.bayesian_prob(counts, tuning_curve, binsize)
    assert np.allclose(prob[0][2], 1.0)


def test_bayesian_prob_nospike():
    tuning_curve = np.array([[2., 0., 0.]])
    counts = np.array([[0.]])
    binsize = 1.0
    prob = vdm.bayesian_prob(counts, tuning_curve, binsize)
    assert np.sum(np.isnan(prob)) == prob.size


def test_bayesian_prob_multtc():
    tuning_curve = np.array([[2., 0., 0.],
                             [0., 5., 0.],
                             [0., 0., 5.]])
    counts = np.array([[0.],
                       [4.],
                       [2.]])
    binsize = 1.0
    prob = vdm.bayesian_prob(counts, tuning_curve, binsize)
    assert np.allclose(prob, np.array([[0.02997459, 0.93271674, 0.03730867]]))


def test_bayesian_prob_emptytcbin():
    tuning_curve = np.array([[0., 1., 0.],
                             [0., 5., 0.],
                             [0., 0., 5.]])
    counts = np.array([[0.],
                       [2.],
                       [2.]])
    binsize = 1.0
    prob = vdm.bayesian_prob(counts, tuning_curve, binsize)
    assert np.isnan(prob[0][0])
    assert np.allclose(prob[0][1], 0.5)
    assert np.allclose(prob[0][2], 0.5)


def test_bayesian_prob_onepos():
    tuning_curve = np.array([[10.]])
    counts = np.array([[0., 2., 4.]])
    binsize = 1.0
    prob = vdm.bayesian_prob(counts, tuning_curve, binsize)
    assert np.isnan(prob[0][0])
    assert np.allclose(prob[1][0], 1.0)
    assert np.allclose(prob[2][0], 1.0)


def test_bayesian_prob_multtimepos():
    tuning_curve = np.array([[3., 0., 0.]])
    counts = np.array([[0., 2., 4.]])
    binsize = 1.0
    prob = vdm.bayesian_prob(counts, tuning_curve, binsize)
    assert np.sum(np.isnan(prob[0])) == 3
    assert np.allclose(prob[1][0], 1.0)
    assert np.sum(np.isnan(prob[1])) == 2
    assert np.allclose(prob[2][0], 1.0)
    assert np.sum(np.isnan(prob[2])) == 2


def test_bayesian_prob_multneurons():
    tuning_curve = np.array([[2., 0., 0.],
                             [0., 5., 0.],
                             [0., 0., 5.]])
    counts = np.array([[0., 8, 0.],
                       [4., 0., 1.],
                       [0., 1., 3.]])
    binsize = 1.0
    prob = vdm.bayesian_prob(counts, tuning_curve, binsize)

    assert np.allclose(prob[0], np.array([0.0310880460, 0.967364171, 0.00154778267]))
    assert np.allclose(prob[1], np.array([0.998834476, 0.000194254064, 0.000971270319]))
    assert np.allclose(prob[2], np.array([0.133827265, 0.0333143360, 0.832858399]))
