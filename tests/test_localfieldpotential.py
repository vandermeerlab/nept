import numpy as np
import pytest
import nept


def test_localfieldpotential_nsamples():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([1.1, 0.9, 2.3])
    lfp = nept.LocalFieldPotential(data, times)

    assert np.allclose(lfp.n_samples, 3)


def test_localfieldpotential_too_many():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([1.1, 0.9, 2.3])
    other = np.array([3.1, 2.0, 1.4])

    with pytest.raises(ValueError) as excinfo:
        lfp = nept.LocalFieldPotential([data, other], times)

    assert str(excinfo.value) == 'can only contain one LFP'


def test_localfieldpotential_slice():
    times = np.array([1.0, 2.0, 3.0])
    data = np.array([1.1, 0.9, 2.3])

    lfp = nept.LocalFieldPotential(data, times)
    sliced = lfp[:2]

    assert np.allclose(sliced.time, np.array([1.0, 2.0]))
    assert np.allclose(sliced.data, np.array([[1.1], [0.9]]))
