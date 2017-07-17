import numpy as np
from scipy import signal
import warnings

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


def get_edges(position, binsize, lastbin=True):
    """Finds edges based on linear time

    Parameters
    ----------
    position : nept.AnalogSignal
    binsize : float
        This is the desired size of bin.
        Typically set around 0.020 to 0.040 seconds.
    lastbin : boolean
        Determines whether to include the last bin. This last bin may
        not have the same binsize as the other bins.

    Returns
    -------
    edges : np.array

    """
    edges = np.arange(position.time[0], position.time[-1], binsize)

    if lastbin:
        if edges[-1] != position.time[-1]:
            edges = np.hstack((edges, position.time[-1]))

    return edges


def bin_spikes(spikes, position, window_size, window_advance,
               gaussian_std=None, n_gaussian_std=5, normalized=True):
    """Bins spikes using a sliding window.

    Parameters
    ----------
    spikes: list
        Of nept.SpikeTrain
    position: nept.AnalogSignal
    window_size: float
    window_advance: float
    gaussian_std: float or None
    n_gaussian_std: int
    normalized: boolean

    Returns
    -------
    binned_spikes: nept.AnalogSignal

    """
    bin_edges = get_edges(position, window_advance, lastbin=True)

    given_n_bins = window_size / window_advance
    n_bins = int(round(given_n_bins))
    if abs(n_bins - given_n_bins) > 0.01:
        warnings.warn("window advance does not divide the window size evenly. "
                      "Using window size %g instead." % (n_bins*window_advance))

    if normalized:
        square_filter = np.ones(n_bins) * (1 / n_bins)
    else:
        square_filter = np.ones(n_bins)

    counts = np.zeros((len(spikes), len(bin_edges)))
    for idx, spiketrain in enumerate(spikes):
        counts[idx] = np.convolve(np.hstack([np.histogram(spiketrain.time, bins=bin_edges)[0],
                                             np.array([0])]),
                                  square_filter, mode='same')

    if gaussian_std is not None and gaussian_std > window_advance:
        n_points = n_gaussian_std * gaussian_std * 2 / window_advance
        n_points = max(n_points, 1.0)
        if n_points % 2 == 0:
            n_points += 1
        n_points = int(round(n_points))
        gaussian_filter = signal.gaussian(n_points, gaussian_std / window_advance)
        gaussian_filter /= np.sum(gaussian_filter)

        if len(gaussian_filter) < counts.shape[1]:
            for idx, spiketrain in enumerate(spikes):
                counts[idx] = np.convolve(counts[idx], gaussian_filter, mode='same')
        else:
            raise ValueError("gaussian filter too long for this length of time")

    return nept.AnalogSignal(counts.T, bin_edges)


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


def expand_line(start_pt, stop_pt, line, expand_by=6):
    """ Creates buffer zone around a line.

    Parameters
    ----------
    start_pt : Shapely's Point object
    stop_pt : Shapely's Point object
    line : Shapely's LineString object
    expand_by : int
        This sets by how much you wish to expand the line.
        Defaults to 6.

    Returns
    -------
    zone : Shapely's Polygon object

    """
    line_expanded = line.buffer(expand_by)
    zone = start_pt.union(line_expanded).union(stop_pt)
    return zone


def perievent_slice(analogsignal, events, t_before, t_after, dt=None):
    """Slices the analogsignal data into perievent chunks.
    Unlike time_slice, the resulting AnalogSignal will be multidimensional.
    Only works for 1D signals.

    Parameters
    ----------
    analogsignal : nept.AnalogSignal
    events : np.array
    t_before : float
    t_after : float
    dt : float

    Returns
    -------
    nept.AnalogSignal

    """

    if analogsignal.dimensions != 1:
        raise ValueError("AnalogSignal must be 1D.")

    if dt is None:
        dt = np.median(np.diff(analogsignal.time))

    time = np.arange(-t_before, t_after+dt, dt)

    data = np.zeros((len(time), len(events)))
    for i, event in enumerate(events):
        sliced = analogsignal.time_slice(event-t_before, event+t_after)
        data[:,i] = np.interp(time+event, sliced.time, np.squeeze(sliced.data))

    return nept.AnalogSignal(data, time)