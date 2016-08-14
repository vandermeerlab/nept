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


def decoded_sequences(decoded, min_idx_length=3, max_idx_jump=20):
    """Finds intervals of decoded position that are within jump limits.

    Parameters
    ----------
    decoded : np.array
        Estimate of decoded location (floats) for each time bin.
    min_idx_length : int
        Minimum number of bins to be considered a sequence.
    max_idx_jump : int
        Maximum number of bins to break a sequence.

    Returns
    -------
    sequences : dict
        With time (tuple of floats), index (tuple of ints) as keys.
        Where start, stop are [0], [1] of each tuple.

    """
    previous_position = -100
    start_idx = []
    sequences = dict(time=[], index=[])

    for idx, position in enumerate(decoded['position']):
        this_jump = np.abs(position - previous_position)
        if this_jump <= max_idx_jump:
            if len(start_idx) < 1:
                start_idx.append(idx-1)
            else:
                if ((idx-1) - start_idx[0]) >= min_idx_length:
                    sequences['time'].append((decoded['time'][start_idx[0]], decoded['time'][idx-1]))
                    sequences['index'].append((start_idx[0], idx-1))
                    start_idx = []
        previous_position = position

    return sequences
