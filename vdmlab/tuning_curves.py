import numpy as np
from scipy import signal
from shapely.geometry import Point
from scipy.ndimage.filters import gaussian_filter

from .utils import find_nearest_idx


def linear_trajectory(pos, ideal_path, trial_start, trial_stop):
    """ Projects 2D positions into an 'ideal' linear trajectory.

    Parameters
    ----------
    pos : dict
        With x, y, time as keys. 2D position information.
    ideal_path : Shapely's LineString object
    trial_start : float
    trial_stop : float

    Returns
    -------
    z : dict
        With position, time as keys

    """
    t_start_idx = find_nearest_idx(np.array(pos['time']), trial_start)
    t_end_idx = find_nearest_idx(np.array(pos['time']), trial_stop)

    pos_trial = dict()
    pos_trial['x'] = pos['x'][t_start_idx:t_end_idx]
    pos_trial['y'] = pos['y'][t_start_idx:t_end_idx]
    pos_trial['time'] = pos['time'][t_start_idx:t_end_idx]

    z = dict(position=[])
    z['time'] = np.array(pos_trial['time'])
    for point_x, point_y in zip(pos_trial['x'], pos_trial['y']):
        z['position'].append(ideal_path.project(Point(point_x, point_y)))
    z['position'] = np.array(z['position'])
    return z


def tuning_curve(linear, spike_times, binsize, sampling_rate=1/30., gaussian_std=3):
    """ Computes tuning curves for neurons relative to linear position.

    Parameters
    ----------
    linear : dict
        With position, time as keys
    spike_times : list of arrays
        Each inner array contains the spike times (floats) for an individual neuron.
    sampling_rate : float
        Default set to 1/30.
    binsize : int
        Defaults to 3 if not specified
    gaussian_std : int
        Defaults to 3. No smoothing if None.

    Returns
    -------
    out_tc : list of np.arrays
        Where each inner array contains the tuning curves for an
        individual neuron.

    Notes
    -----
    Input linear and spikes should be from the same time
    period. Eg. when the animal is running on the track.

    """
    linear_start = np.min(linear['position'])
    linear_stop = np.max(linear['position'])
    edges = np.arange(linear_start, linear_stop, binsize)
    if edges[-1] < linear_stop:
        edges = np.hstack([edges, linear_stop])

    position_counts = np.histogram(linear['position'], bins=edges)[0]
    position_counts = position_counts.astype(float)
    position_counts *= sampling_rate
    occupied_idx = position_counts > 0

    tc = []
    for idx, neuron_spikes in enumerate(spike_times):
        counts_idx = []
        for spike_time in neuron_spikes:
            bin_idx = find_nearest_idx(linear['time'], spike_time)
            counts_idx.append(linear['position'][bin_idx])
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

    return out_tc


def get_speed(pos, smooth=True, t_smooth=0.5):
    """Finds the velocity of the animal from 2D position.

    Parameters
    ----------
    pos : dict
        With x, y, time as keys.
    smooth : bool
        Whether smoothing occurs. Default is True.
    t_smooth : float
        Range over which smoothing occurs in seconds. Default is 0.5 seconds.

    Returns
    -------
    speed : dict
        With time (floats), velocity (floats) as keys.

    """
    speed = dict()
    speed['time'] = pos['time']
    speed['velocity'] = np.sqrt((pos['x'][1:] - pos['x'][:-1]) ** 2 + (pos['y'][1:] - pos['y'][:-1]) ** 2)
    speed['velocity'] = np.hstack(([0], speed['velocity']))

    dt = np.median(np.diff(speed['time']))

    filter_length = np.ceil(t_smooth/dt)
    speed['smoothed'] = np.convolve(speed['velocity'], np.ones(int(filter_length))/filter_length, 'same')

    return speed


def tuning_curve_2d(spikes, position, xedges, yedges, sampling_rate=1/30., gaussian_sigma=None):
    """Creates 2D tuning curves based on spikes and 2D position.

    Parameters
    ----------
    spikes : dict
        With time (floats) and labels (str) as keys. Where each inner array
        represents the spike times for an individual neuron.
    position : dict
        With x (floats), y (floats), time (floats) as keys.
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
    position_2d, pos_xedges, pos_yedges = np.histogram2d(position['y'], position['x'], bins=[yedges, xedges])
    position_2d *= sampling_rate
    shape = position_2d.shape
    occupied_idx = position_2d > 0

    tc = []
    for neuron_spikes in spikes['time']:
        spikes_x = []
        spikes_y = []
        for spike_time in neuron_spikes:
            spike_idx = find_nearest_idx(position['time'], spike_time)
            spikes_x.append(position['x'][spike_idx])
            spikes_y.append(position['y'][spike_idx])
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
