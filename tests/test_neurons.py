import numpy as np
import pytest
import nept


def test_neurons_basic():
    spikes = np.array(
        [
            nept.SpikeTrain(np.array([0.5]), "test"),
            nept.SpikeTrain(np.array([1.5]), "test"),
            nept.SpikeTrain(np.array([2.5]), "test"),
        ]
    )

    tuning = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    neurons = nept.Neurons(spikes, tuning)

    assert np.allclose(neurons.spikes[0].time, spikes[0].time)
    assert np.allclose(neurons.tuning_curves, tuning)


def test_neurons_n_wrong():
    spikes = np.array(
        [
            nept.SpikeTrain(np.array([0.5]), "test"),
            nept.SpikeTrain(np.array([1.5]), "test"),
            nept.SpikeTrain(np.array([2.5]), "test"),
        ]
    )

    tuning = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    with pytest.raises(ValueError) as excinfo:
        neurons = nept.Neurons(spikes, tuning)

    assert (
        str(excinfo.value)
        == "spikes and tuning curves must have the same number of neurons"
    )


def test_neurons_getitem_single():
    spikes = np.array(
        [
            nept.SpikeTrain(np.array([0.5]), "test"),
            nept.SpikeTrain(np.array([1.5]), "test"),
            nept.SpikeTrain(np.array([2.5]), "test"),
        ]
    )

    tuning = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    neurons = nept.Neurons(spikes, tuning)

    sliced = neurons[1]

    assert np.allclose(sliced.spikes[0].time, np.array([1.5]))
    assert np.allclose(sliced.tuning_curves[0], np.array([0.0, 1.0, 0.0, 0.0]))


def test_neurons_getitem_multiple():
    spikes = np.array(
        [
            nept.SpikeTrain(np.array([0.5]), "test"),
            nept.SpikeTrain(np.array([1.5]), "test"),
            nept.SpikeTrain(np.array([2.5]), "test"),
        ]
    )

    tuning = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    neurons = nept.Neurons(spikes, tuning)

    sliced = neurons[0:2]

    assert np.allclose(
        sliced.tuning_curves, np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    )
    assert np.allclose(sliced.spikes[0].time, np.array([0.5]))
    assert np.allclose(sliced.spikes[1].time, np.array([1.5]))


def test_neurons_slicing_specified_startstop():
    spikes = np.array(
        [
            nept.SpikeTrain(np.array([0.5]), "test"),
            nept.SpikeTrain(np.array([1.5]), "test"),
            nept.SpikeTrain(np.array([2.5]), "test"),
        ]
    )

    tuning = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    neurons = nept.Neurons(spikes, tuning)

    t_start = 1.0
    t_stop = 2.0

    sliced_neurons = neurons.time_slice(t_start, t_stop)

    assert np.allclose(sliced_neurons.spikes[0].time, np.array([]))
    assert np.allclose(sliced_neurons.spikes[1].time, np.array([1.5]))
    assert np.allclose(sliced_neurons.spikes[2].time, np.array([]))
    assert np.allclose(neurons.tuning_curves, tuning)


def test_neurons_slicing_specified_stop():
    spikes = np.array(
        [
            nept.SpikeTrain(np.array([0.5]), "test"),
            nept.SpikeTrain(np.array([1.5]), "test"),
            nept.SpikeTrain(np.array([2.5]), "test"),
        ]
    )

    tuning = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    neurons = nept.Neurons(spikes, tuning)

    t_stop = 2.0

    sliced_neurons = neurons.time_slice(None, t_stop)

    assert np.allclose(sliced_neurons.spikes[0].time, np.array([0.5]))
    assert np.allclose(sliced_neurons.spikes[1].time, np.array([1.5]))
    assert np.allclose(sliced_neurons.spikes[2].time, np.array([]))
    assert np.allclose(neurons.tuning_curves, tuning)


def test_neurons_slicing_specified_start():
    spikes = np.array(
        [
            nept.SpikeTrain(np.array([0.5]), "test"),
            nept.SpikeTrain(np.array([1.5]), "test"),
            nept.SpikeTrain(np.array([2.5]), "test"),
        ]
    )

    tuning = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    neurons = nept.Neurons(spikes, tuning)

    t_start = 1.0

    sliced_neurons = neurons.time_slice(t_start, None)

    assert np.allclose(sliced_neurons.spikes[0].time, np.array([]))
    assert np.allclose(sliced_neurons.spikes[1].time, np.array([1.5]))
    assert np.allclose(sliced_neurons.spikes[2].time, np.array([2.5]))
    assert np.allclose(neurons.tuning_curves, tuning)


def test_neurons_slicing_mult():
    spikes = np.array(
        [
            nept.SpikeTrain(np.array([0.5]), "test"),
            nept.SpikeTrain(np.array([1.5]), "test"),
            nept.SpikeTrain(np.array([2.5]), "test"),
        ]
    )

    tuning = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    neurons = nept.Neurons(spikes, tuning)

    t_starts = [0.0, 2.0]
    t_stops = [1.0, 3.0]

    sliced_neurons = neurons.time_slice(t_starts, t_stops)

    assert np.allclose(sliced_neurons.spikes[0].time, np.array([0.5]))
    assert np.allclose(sliced_neurons.spikes[1].time, np.array([]))
    assert np.allclose(sliced_neurons.spikes[2].time, np.array([2.5]))
    assert np.allclose(neurons.tuning_curves, tuning)


def test_neurons_get_num():
    spikes = np.array(
        [
            nept.SpikeTrain(np.array([0.5]), "test"),
            nept.SpikeTrain(np.array([1.5]), "test"),
            nept.SpikeTrain(np.array([2.5]), "test"),
        ]
    )

    tuning = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    neurons = nept.Neurons(spikes, tuning)

    assert np.allclose(neurons.n_neurons, spikes.shape[0])


def test_neurons_get_tuning_shape():
    spikes = np.array(
        [
            nept.SpikeTrain(np.array([0.5]), "test"),
            nept.SpikeTrain(np.array([1.5]), "test"),
            nept.SpikeTrain(np.array([2.5]), "test"),
        ]
    )

    tuning = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    neurons = nept.Neurons(spikes, tuning)

    assert np.allclose(neurons.tuning_shape, tuning[0].shape)
