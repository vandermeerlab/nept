import numpy as np
from scipy import signal

import nept


def find_nearest_idx(array, val):
    """Finds nearest index in array to value.

    Parameters
    ----------
    array : np.array
    val : float

    Returns
    -------
    Index into array that is closest to val

    """
    return (np.abs(array-val)).argmin()


def find_nearest_indices(array, vals):
    """Finds nearest index in array to value.

    Parameters
    ----------
    array : np.array
        This is the array you wish to index into.
    vals : np.array
        This is the array that you are getting your indices from.

    Returns
    -------
    Indices into array that is closest to vals.

    Notes
    -----
    Wrapper around find_nearest_idx().

    """
    return np.array([find_nearest_idx(array, val) for val in vals], dtype=int)


def find_multi_in_epochs(spikes, epochs, min_involved):
    """Finds epochs with minimum number of participating neurons.

    Parameters
    ----------
    spikes: np.array
        Of nept.SpikeTrain objects
    epochs: nept.Epoch
    min_involved: int

    Returns
    -------
    multi_epochs: nept.Epoch

    """
    multi_starts = []
    multi_stops = []

    n_neurons = len(spikes)
    for start, stop in zip(epochs.starts, epochs.stops):
        involved = 0
        for this_neuron in spikes:
            if ((start <= this_neuron.time) & (this_neuron.time <= stop)).sum() >= 1:
                involved += 1
        if involved >= min_involved:
            multi_starts.append(start)
            multi_stops.append(stop)

    multi_starts = np.array(multi_starts)
    multi_stops = np.array(multi_stops)

    multi_epochs = nept.Epoch(np.hstack([np.array(multi_starts)[..., np.newaxis],
                                        np.array(multi_stops)[..., np.newaxis]]))

    return multi_epochs


def get_sort_idx(tuning_curves):
    """Finds indices to sort neurons by max firing in tuning curve.

    Parameters
    ----------
    tuning_curves : list of lists
        Where each inner list is the tuning curves for an individual
        neuron.

    Returns
    -------
    sorted_idx : list
        List of integers that correspond to the neuron in sorted order.

    """
    tc_max_loc = []
    for i, neuron_tc in enumerate(tuning_curves):
        tc_max_loc.append((i, np.where(neuron_tc == np.max(neuron_tc))[0][0]))
    sorted_by_tc = sorted(tc_max_loc, key=lambda x: x[1])

    sorted_idx = []
    for idx in sorted_by_tc:
        sorted_idx.append(idx[0])

    return sorted_idx


def get_counts(spikes, edges, gaussian_std=None, n_gaussian_std=5):
    """Finds the number of spikes in each bin.

    Parameters
    ----------
    spikes : list
        Contains nept.SpikeTrain for each neuron
    edges : np.array
        Bin edges for computing spike counts.
    gaussian_std : float
        Standard deviation for gaussian filter. Default is None.

    Returns
    -------
    counts : nept.AnalogSignal
        Where each inner array is the number of spikes (int) in each bin for an individual neuron.

    """
    dt = np.median(np.diff(edges))

    if gaussian_std is not None:
        n_points = n_gaussian_std * gaussian_std * 2 / dt
        n_points = max(n_points, 1.0)
        if n_points % 2 == 0:
            n_points += 1
        if n_points > len(edges):
            raise ValueError("gaussian_std is too large for these times")
        gaussian_filter = signal.gaussian(n_points, gaussian_std/dt)
        gaussian_filter /= np.sum(gaussian_filter)

    counts = np.zeros((len(spikes), len(edges)-1))
    for idx, spiketrain in enumerate(spikes):
        counts[idx] = np.histogram(spiketrain.time, bins=edges)[0]
        if gaussian_std is not None and gaussian_std > dt:
            counts[idx] = np.convolve(counts[idx], gaussian_filter, mode='same')

    return nept.AnalogSignal(counts.T, edges[:-1])


def cartesian(xcenters, ycenters):
    """Finds every combination of elements in two arrays.

    Parameters
    ----------
    xcenters : np.array
    ycenters : np.array

    Returns
    -------
    cartesian : np.array
        With shape(n_sample, 2).

    """
    return np.transpose([np.tile(xcenters, len(ycenters)), np.repeat(ycenters, len(xcenters))])


def get_xyedges(position, binsize=3):
    """Gets edges based on position min and max.

    Parameters
    ----------
    position: 2D nept.Position
    binsize: int

    Returns
    -------
    xedges: np.array
    yedges: np.array

    """
    xedges = np.arange(position.x.min(), position.x.max() + binsize, binsize)
    yedges = np.arange(position.y.min(), position.y.max() + binsize, binsize)

    return xedges, yedges
