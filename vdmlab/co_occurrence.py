import numpy as np
import warnings


def spike_counts(spikes, interval_times, window=None):
    """Get spike counts for specific interval.

    Parameters
    ----------
    spike_times : np.array
    interval_times : dict
        With start(int or float), stop(int or float) as keys
    window : float
        When window is set, takes the spike times for this window length
        around the center of the interval times. The default is set to None.
        Suggested window size is 0.1.

    Returns
    -------
    count_matrix : np.array
        num_neurons x num_bins

    """
    intervals = np.vstack((interval_times['start'], interval_times['stop']))
    bin_centers = np.mean(intervals, axis=0)

    if window is not None:
        intervals = np.vstack([bin_centers-(window*0.5), bin_centers+(window*0.5)])

    num_neurons = len(spikes['time'])
    count_matrix = np.zeros((num_neurons, intervals.shape[1]))

    for i, (start, stop) in enumerate(zip(intervals[0], intervals[1])):
        for neuron in range(num_neurons):
            count_matrix[neuron][i] = ((start <= spikes['time'][neuron]) & (spikes['time'][neuron] <= stop)).sum()

    return count_matrix


def compute_cooccur(count_matrix, num_shuffles=10000):
    """Computes the probabilities for co-occurrence

    Parameters
    ----------
    count_matrix : np.array
        num_neurons by num_bins

    Returns
    -------
    prob_active : np.array
    prob_expected : ap.array
    prob_observed : ap.array
    prob_zscore : np.array

    Notes
    -----
    NOT IMPLEMENTED: Handle mask for neurons that were recorded on the same tetrode

    """
    activity_matrix = bool_counts(count_matrix)
    prob_active = prob_active_neuron(activity_matrix)
    prob_expected = expected_cooccur(prob_active)
    prob_observed = observed_cooccur(activity_matrix)
    prob_shuffle = shuffle_cooccur(activity_matrix, num_shuffles)
    prob_zscore = zscore_cooccur(prob_observed, prob_shuffle)

    prob_expected = vector_from_array(prob_expected)
    prob_observed = vector_from_array(prob_observed)
    prob_zscore = vector_from_array(prob_zscore)

    return prob_active, prob_expected, prob_observed, prob_zscore


def vector_from_array(array):
    """Get triangle of output in vector from a correlation-type array

    Parameters
    ----------
    array : np.array

    Returns
    -------
    vector (np.array)

    Notes
    -----
    Old Matlab code indexes by column (aka.Fortran-style), so to get the indices
    of the top triangle, we have to do some reshaping.
    Otherwise, if the vector made up by rows is OK, then simply :
    triangle = np.triu_indices(array.size, k=1), out = array[triangle]

    """
    triangle_lower = np.tril_indices(array.shape[0], k=-1)
    flatten_idx = np.arange(array.size).reshape(array.shape)[triangle_lower]
    triangle = np.unravel_index(flatten_idx, array.shape, order='F')

    # triangle = np.triu_indices(array.size, k=1)
    # out = array[triangle]

    return array[triangle]


def bool_counts(count_matrix, min_spikes=1):
    """Converts count matrix to boolean of whether bin active or not

    Parameters
    ----------
    count_matrix : np.array
        num_neurons by num_bins
    min_spikes : int
        Minimum number of spikes in a bin for that bin to be considered active.
        The defaule is set to 1.

    Returns
    -------
    activity_matrix : np.array
        num_neurons by num_bins, boolean (1 or 0)

    """
    activity_matrix = np.zeros(count_matrix.shape)
    activity_matrix[count_matrix >= min_spikes] = 1

    return activity_matrix

def prob_active_neuron(activity_matrix):
    """Get expected co-occurrence under independence assumption

    Parameters
    ----------
    activity_matrix : np.array
        num_neurons by num_bins, boolean (1 or 0)

    Returns
    -------
    prob_active : np.array
        Fraction of bins each cell participates in individually

    """
    prob_active = np.mean(activity_matrix, axis=1)

    return prob_active


def expected_cooccur(prob_active):
    """Expected co-occurrence, multiply single cell probabilities

    Parameters
    ----------
    prob_active : np.array
        Fraction of bins each cell participates in individually

    Returns
    -------
    prob_expected : np.array
        .. math:: p(x|y)

    """
    prob_expected = np.outer(prob_active, prob_active)

    # Remove probability of cell co-occurring with itself
    prob_expected[np.eye(len(prob_expected), dtype=bool)] = np.nan

    return prob_expected


def observed_cooccur(activity_matrix):
    """Observed co-occurrences

    Parameters
    ----------
    activity_matrix : np.array
        num_neurons by num_bins, boolean (1 or 0)

    Returns
    -------
    prob_observed : np.array
        .. math:: p(x,y)

    """
    num_neurons = activity_matrix.shape[0]

    prob_observed = np.zeros((num_neurons, num_neurons))
    for i in range(num_neurons):
        neuron_activities = activity_matrix[i]
        prob_observed[i] = np.mean(neuron_activities * activity_matrix, axis=1)

    # Remove probability of cell co-occurring with itself
    prob_observed[np.eye(len(prob_observed), dtype=bool)] = np.nan

    return prob_observed


def shuffle_cooccur(activity_matrix, num_shuffles):
    """Compute shuffle matrices from experimental observations

    Parameters
    ----------
    activity_matrix : np.array
        num_neurons by num_bins, boolean (1 or 0)
    num_shuffle : int
        Number of times to shuffle the activity matrix.

    Returns
    -------
    prob_shuffle : np.array

    """
    shuffled_matrix = activity_matrix

    rows = shuffled_matrix.shape[0]
    cols = shuffled_matrix.shape[1]

    prob_shuffle = np.zeros((num_shuffles, rows, rows))

    for i in range(num_shuffles):
        this_matrix = shuffled_matrix
        for j in range(rows):
            this_matrix[j] = this_matrix[j, np.random.permutation(range(cols))]
        for k in range(rows):
            neuron_activities = this_matrix[k]
            prob_shuffle[i, k] = np.nanmean(neuron_activities * this_matrix, axis=1)

    return prob_shuffle


def zscore_cooccur(prob_observed, prob_shuffle):
    """Compare co-occurrence observed probabilities with shuffle

    Parameters
    ----------
    prob_observed : np.array
    prob_shuffle : np.array

    Returns
    -------
    prob_zscore : np.array

    """
    num_neurons = prob_observed.shape[0]

    prob_zscore = np.zeros((num_neurons, num_neurons))
    for i in range(num_neurons):
        for j in range(num_neurons):
            # if np.nanstd(np.squeeze(prob_shuffle[:, i, j])) > 0.0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                prob_zscore[i][j] = (prob_observed[i][j] -
                                     np.nanmean(np.squeeze(prob_shuffle[:, i, j]))) / np.nanstd(np.squeeze(prob_shuffle[:, i, j]))
            # else:
            #     prob_zscore[i][j] = prob_observed[i][j] - np.nanmean(np.squeeze(prob_shuffle[:, i, j]))

    return prob_zscore
