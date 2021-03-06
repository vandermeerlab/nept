import numpy as np
import scipy

from .utils import gaussian_filter


def get_bin_edges(position, binsize):
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
    leftover = (pos_stop - pos_start) % binsize

    return np.arange(pos_start - leftover * 0.5, pos_stop + binsize, binsize)


def tuning_curve_1d(position, spikes, edges, gaussian_std=None, min_occupancy=0):
    """Computes tuning curves for neurons relative to linear position.

    Parameters
    ----------
    position : nept.Position
        Must be a linear position (1D).
    spikes : list
        Containing nept.SpikeTrain for each neuron.
    edges : np.array
        Edges for position bins. All bins must be the same size.
    gaussian_std : int or None
        No smoothing if None.

    Returns
    -------
    tuning_curves : list of np.arrays
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

    binsize = edges[1] - edges[0]
    assert np.allclose(np.diff(edges), binsize), "All bins must be the same size"

    occupancy = np.histogram(position.x, bins=edges)[0].astype(float)
    occupancy *= sampling_rate
    occupied = occupancy > (np.median(occupancy[occupancy > 0]) * min_occupancy)
    occupancy[~occupied] = np.nan

    tc = []
    for spiketrain in spikes:
        f_xy = scipy.interpolate.interp1d(
            position.time, position.data.T, kind="nearest", fill_value="extrapolate"
        )
        spikes_x = f_xy(spiketrain.time)

        spike_counts = np.histogram(spikes_x, bins=edges)[0]

        firing_rate = np.ones(len(edges) - 1) * np.nan
        firing_rate[occupied] = spike_counts[occupied] / occupancy[occupied]
        if gaussian_std is not None:
            firing_rate = gaussian_filter(
                firing_rate, gaussian_std, dt=binsize, boundary="extend"
            )

        tc.append(firing_rate)

    tuning_curves = np.array(tc, dtype=float)

    return tuning_curves, occupancy


def get_occupancy(position, yedges, xedges):
    sampling_rate = np.median(np.diff(position.time))

    position_2d, pos_xedges, pos_yedges = np.histogram2d(
        position.y, position.x, bins=[yedges, xedges]
    )
    position_2d *= sampling_rate

    return position_2d


def tuning_curve_2d(
    position, spikes, xedges, yedges, occupied_thresh=0, gaussian_std=None
):
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
    position_2d = get_occupancy(position, yedges, xedges)
    shape = position_2d.shape
    occupied_idx = position_2d > occupied_thresh

    tuning_curves = np.full(((len(spikes),) + shape), np.nan)
    for i, spiketrain in enumerate(spikes):
        f_xy = scipy.interpolate.interp1d(
            position.time, position.data.T, kind="nearest"
        )
        spikes_xy = f_xy(spiketrain.time)

        spikes_2d, spikes_xedges, spikes_yedges = np.histogram2d(
            spikes_xy[1], spikes_xy[0], bins=[yedges, xedges]
        )
        tuning_curves[i, occupied_idx] = (
            spikes_2d[occupied_idx] / position_2d[occupied_idx]
        )

    if gaussian_std is not None:
        xbinsize = xedges[1] - xedges[0]
        ybinsize = yedges[1] - yedges[0]
        # TODO: use Gaussian2DKernel
        tuning_curves = gaussian_filter(
            tuning_curves, gaussian_std, dt=xbinsize, axis=1
        )
        tuning_curves = gaussian_filter(
            tuning_curves, gaussian_std, dt=ybinsize, axis=2
        )

    return tuning_curves
