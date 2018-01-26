import numpy as np
from scipy import signal

from .utils import find_nearest_idx, gaussian_filter


def binned_position(position, binsize):
    """Bins 1D position by the binsize.

    Parameters
    ----------
    position : nept.Position
        Must be a 1D position
    binsize : int

    Returns
    -------
    edges : np.array

    """
    if not position.dimensions == 1:
        raise ValueError("position must be linear")

    pos_start = np.min(position.x)
    pos_stop = np.max(position.x)
    edges = np.arange(pos_start, pos_stop, binsize)

    if edges[-1] < pos_stop:
        edges = np.hstack([edges, pos_stop])

    return edges


def tuning_curve_1d(position, spikes, binsize, gaussian_std=None):
    """ Computes tuning curves for neurons relative to linear position.

    Parameters
    ----------
    position : nept.Position
        Must be a linear position (1D).
    spikes : list
        Containing nept.SpikeTrain for each neuron.
    binsize : int
    gaussian_std : int or None
        No smoothing if None.

    Returns
    -------
    out_tc : list of np.arrays
        Where each inner array contains the tuning curves for an
        individual neuron.

    Notes
    -----
    Input position and spikes should be from the same time
    period. Eg. when the animal is running on the track.

    """
    if not position.dimensions == 1:
        raise ValueError("position must be linear")

    sampling_rate = np.median(np.diff(position.time))

    edges = binned_position(position, binsize)

    position_counts = np.histogram(position.x, bins=edges)[0]
    position_counts = position_counts.astype(float)
    position_counts *= sampling_rate
    occupied_idx = position_counts > 0

    tc = []
    for spiketrain in spikes:
        counts_idx = []
        for spike_time in spiketrain.time:
            bin_idx = find_nearest_idx(position.time, spike_time)
            if np.abs(position.time[bin_idx] - spike_time) < sampling_rate:
                counts_idx.append(position.x[bin_idx])
        spike_counts = np.histogram(counts_idx, bins=edges)[0]

        firing_rate = np.zeros(len(edges)-1)
        firing_rate[occupied_idx] = spike_counts[occupied_idx] / position_counts[occupied_idx]
        if gaussian_std is not None:
            firing_rate = gaussian_filter(firing_rate, gaussian_std, dt=binsize)
        tc.append(firing_rate)

    return np.array(tc, dtype=float)


def tuning_curve_2d(position, spikes, xedges, yedges, occupied_thresh=0, gaussian_std=None):
    """Creates 2D tuning curves based on spikes and 2D position.

    Parameters
    ----------
    position : nept.Position
        Must be a 2D position.
    spikes : list
        Containing nept.SpikeTrain for each neuron.
    xedges : np.array
    yedges : np.array
    sampling_rate : float
    occupied_thresh: float
    gaussian_sigma : float
        Sigma used in gaussian filter if filtering.

    Returns
    -------
    tuning_curves : np.array
        Where each inner array is the tuning curve for an individual neuron.

    """
    sampling_rate = np.median(np.diff(position.time))

    position_2d, pos_xedges, pos_yedges = np.histogram2d(position.y, position.x, bins=[yedges, xedges])
    position_2d *= sampling_rate
    shape = position_2d.shape
    occupied_idx = position_2d > occupied_thresh

    tuning_curves = np.zeros((len(spikes),) + shape)
    for i, spiketrain in enumerate(spikes):
        spikes_x = np.interp(spiketrain.time, position.time, position.x)
        spikes_y = np.interp(spiketrain.time, position.time, position.y)
        
        spikes_2d, spikes_xedges, spikes_yedges = np.histogram2d(spikes_y, spikes_x, bins=[yedges, xedges])
        tuning_curves[i, occupied_idx] = spikes_2d[occupied_idx] / position_2d[occupied_idx]

    if gaussian_std is not None:
        xbinsize = xedges[1] - xedges[0]
        ybinsize = yedges[1] - yedges[0]
        tuning_curves = gaussian_filter(tuning_curves, gaussian_std, dt=xbinsize, axis=1)
        tuning_curves = gaussian_filter(tuning_curves, gaussian_std, dt=ybinsize, axis=2)

    return tuning_curves
