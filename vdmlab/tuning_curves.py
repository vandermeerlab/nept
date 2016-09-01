import numpy as np
from scipy import signal
from shapely.geometry import Point
from scipy.ndimage.filters import gaussian_filter

from .objects import AnalogSignal
from .utils import find_nearest_idx


def binned_position(position, binsize):
    """Bins 1D position by the binsize.

    Parameters
    ----------
    position : vdm.Position
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


def tuning_curve(position, spikes, binsize, sampling_rate=1/30., gaussian_std=3):
    """ Computes tuning curves for neurons relative to linear position.

    Parameters
    ----------
    position : vdmlab.Position
        Must be a linear position (1D).
    spikes : list
        Containing vdmlab.SpikeTrain for each neuron.
    sampling_rate : float
        Default set to 1/30.
    binsize : int
    gaussian_std : int
        Defaults to 3. No smoothing if None.

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
            counts_idx.append(position.x[bin_idx])
        spike_counts = np.histogram(counts_idx, bins=edges)[0]

        firing_rate = np.zeros(len(edges)-1)
        firing_rate[occupied_idx] = spike_counts[occupied_idx] / position_counts[occupied_idx]
        tc.append(firing_rate)

    if gaussian_std is not None:
        filter_multiplier = 6
        out_tc = []
        gaussian_filter = signal.get_window(('gaussian', gaussian_std), gaussian_std*filter_multiplier)
        normalized_gaussian = gaussian_filter / np.sum(gaussian_filter)
        for firing_rate in tc:
            out_tc.append(np.convolve(firing_rate, normalized_gaussian, mode='same'))
    else:
        print('Tuning curve with no filter.')
        out_tc = tc

    return np.array(out_tc, dtype=float)


def tuning_curve_2d(position, spikes, xedges, yedges, sampling_rate=1/30., gaussian_sigma=None):
    """Creates 2D tuning curves based on spikes and 2D position.

    Parameters
    ----------
    position : vdmlab.Position
        Must be a 2D position.
    spikes : list
        Containing vdmlab.SpikeTrain for each neuron.
    xedges : np.array
    yedges : np.array
    sampling_rate : float
    gaussian_sigma : float
        Sigma used in gaussian filter if filtering.

    Returns
    -------
    tuning_curves : np.array
        Where each inner array is the tuning curve for an individual neuron.

    """
    position_2d, pos_xedges, pos_yedges = np.histogram2d(position.y, position.x, bins=[yedges, xedges])
    position_2d *= sampling_rate
    shape = position_2d.shape
    occupied_idx = position_2d > 0

    tc = []
    for spiketrain in spikes:
        spikes_x = []
        spikes_y = []
        for spike_time in spiketrain.time:
            spike_idx = find_nearest_idx(position.time, spike_time)
            spikes_x.append(position.x[spike_idx])
            spikes_y.append(position.y[spike_idx])
        spikes_2d, spikes_xedges, spikes_yedges = np.histogram2d(spikes_y, spikes_x, bins=[yedges, xedges])

        firing_rate = np.zeros(shape)
        firing_rate[occupied_idx] = spikes_2d[occupied_idx] / position_2d[occupied_idx]

        tc.append(firing_rate)

    if gaussian_sigma is not None:
        tuning_curves = []
        for firing_rate in tc:
            tuning_curves.append(gaussian_filter(firing_rate, gaussian_sigma))
    else:
        print('Tuning curves with no filter.')
        tuning_curves = tc

    return tuning_curves
