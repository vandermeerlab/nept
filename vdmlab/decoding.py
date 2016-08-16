import numpy as np

def bayesian_prob(counts, tuning_curves, binsize, min_neurons=1, min_spikes=1):
    """Computes the bayesian probability of location based on spike counts.

    Parameters
    ----------
    counts : np.array
        Where each inner array is the number of spikes (int) in each bin for an individual neuron.
    tuning_curves : np.array
        Where each inner array is the tuning curve (floats) for an individual neuron.
    binsize : float
        Size of the time bins.
    min_neurons : int
        Mininum number of neurons active in a given bin. Default is 1.
    min_spikes : int
        Mininum number of spikes in a given bin. Default is 1.

    Returns
    -------
    prob : np.array
        Where each inner array is the probability (floats) for an individual neuron by location bins.

    Notes
    -----
    If a bin does not meet the min_neuron/min_spikes requirement, that bin's probability
    is set to nan. To convert it to 0s instead, use : prob[np.isnan(prob)] = 0 on the output.

    """
    n_time_bins = np.shape(counts)[1]
    n_position_bins = np.shape(tuning_curves)[1]

    likelihood = np.empty((n_time_bins, n_position_bins)) * np.nan

    # Ignore warnings when inf created in this loop
    error_settings = np.seterr(over='ignore')
    for idx in range(n_position_bins):
        valid_idx = tuning_curves[:, idx] > 1  # log of 1 or less is negative or invalid
        if np.any(valid_idx):
            # event_rate is the lambda in this poisson distribution
            event_rate = tuning_curves[valid_idx, idx][..., np.newaxis] ** counts[valid_idx]
            prior = np.exp(-binsize * np.sum(tuning_curves[valid_idx, idx]))

            # Below is the same as
            # likelihood[:, idx] = np.prod(event_rate, axis=0) * prior * (1/n_position_bins)
            # only less likely to have floating point issues, though slower
            likelihood[:, idx] = np.exp(np.sum(np.log(event_rate), axis=0)) * prior * (1/n_position_bins)
    np.seterr(**error_settings)

    # Set any inf value to be largest float
    largest_float = np.finfo(float).max
    likelihood[np.isinf(likelihood)] = largest_float
    likelihood /= np.nansum(likelihood, axis=1)[..., np.newaxis]

    # Remove bins with too few neurons that that are active
    # a neuron is considered active by having at least min_spikes in a bin
    n_active_neurons = np.sum(counts >= min_spikes, axis=0)
    likelihood[n_active_neurons < min_neurons] = np.nan

    return likelihood


def decode_location(likelihood, linear):
    """Finds the decoded location based on a linear (1D) trajectory.

    Parameters
    ----------
    prob : np.array
        Where each inner array is the probability (floats) for an individual neuron by location bins.
    linear : dict
        With position (floats), time (floats) as keys.

    Returns
    -------
    decoded : np.array
        Estimate of decoded location (floats) for each time bin.

    """
    max_decoded_idx = np.argmax(likelihood, axis=1)
    decoded = max_decoded_idx * (np.max(linear['position'])-np.min(linear['position'])) / (np.shape(likelihood)[1]-1)
    decoded += np.min(linear['position'])

    nan_idx = np.sum(np.isnan(likelihood), axis=1) == (np.shape(likelihood)[1]-1)
    decoded[nan_idx] = np.nan

    return decoded


def decode_sequences(decoded, min_length=3, max_jump=20):
    """Finds intervals of decoded position that are within jump limits.

    Parameters
    ----------
    decoded : dict
        Estimate of decoded location (floats) for each time bin. Has position, time keys.
    min_length : int
        Minimum number of bins to be considered a sequence.
    max_jump : int
        Any jump greater than this amount will break a sequence.

    Returns
    -------
    sequences : dict
        With time (np.arrays) and position (np.arrays) as keys.

    """
    sequence = dict()
    sequence['position'] = np.split(decoded['position'], np.where(np.abs(np.diff(decoded['position']))>= max_jump)[0]+1)
    sequence['position'] = [i for i in sequence['position'] if i.size >= min_length]

    sequence['time'] = np.split(decoded['time'], np.where(np.abs(np.diff(decoded['position'])) >= max_jump)[0]+1)
    sequence['time'] = [i for i in sequence['time'] if i.size >= min_length]

    return sequence
