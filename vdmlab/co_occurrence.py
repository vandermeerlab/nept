import numpy as np
import warnings
import vdmlab as vdm


def spike_counts(spikes, epochs, window=None):
    """Get spike counts for specific interval.

    Parameters
    ----------
    spikes : list
        Containing vdmlab.SpikeTrain for each neuron.
    interval_times : vdmlab.Epoch
    window : float
        When window is set, takes the spike times for this window length
        around the center of the interval times. The default is set to None.
        Suggested window size is 0.1.

    Returns
    -------
    count_matrix : np.array
        num_neurons x num_bins

    """
    if window is not None:
        epochs = vdm.Epoch(np.array([epochs.centers-(window*0.5), epochs.centers+(window*0.5)]))

    n_neurons = len(spikes)
    count_matrix = np.zeros((n_neurons, epochs.n_epochs))

    for i, (start, stop) in enumerate(zip(epochs.starts, epochs.stops)):
        for neuron in range(n_neurons):
            count_matrix[neuron][i] = ((start <= spikes[neuron].time) & (spikes[neuron].time <= stop)).sum()

    return count_matrix


def get_tetrode_mask(spikes):
    tetrode_mask = np.zeros((len(spikes), len(spikes)), dtype=bool)

    labels = []
    for spiketrain in spikes:
        labels.append(spiketrain.label)

    for i, ilabel in enumerate(labels):
        for j, jlabel in enumerate(labels):
            if ilabel == jlabel:
                tetrode_mask[i][j] = True

    return tetrode_mask


def find_multi_in_epochs(spikes, epochs, min_involved):
    multi_starts = []
    multi_stops = []

    n_neurons = len(spikes)
    for start, stop in zip(epochs.starts, epochs.stops):
        involved = 0
        for neuron in range(n_neurons):
            if ((start <= spikes[neuron].time) & (spikes[neuron].time <= stop)).sum() > 1:
                involved += 1
        if involved > min_involved:
            multi_starts.append(start)
            multi_stops.append(stop)

    multi_starts = np.array(multi_starts)
    multi_stops = np.array(multi_stops)

    multi_epochs = vdm.Epoch(np.array([multi_starts, multi_stops]))

    return multi_epochs


def compute_cooccur(count_matrix, tetrode_mask, num_shuffles=10000):
    """Computes the probabilities for co-occurrence

    Parameters
    ----------
    count_matrix : np.array
        num_neurons by num_bins
    tetrode_mask : np.array
        n_neurons by n_neurons. Boolean of True (same tetrode) or
        False (different tetrode)

    Returns
    -------
    prob_active : np.array
    prob_expected : ap.array
    prob_observed : ap.array
    prob_zscore : np.array
    """
    prob = dict()
    activity_matrix = bool_counts(count_matrix)
    prob['active'] = prob_active_neuron(activity_matrix)
    prob['expected'] = expected_cooccur(prob['active'], tetrode_mask)
    prob['observed'] = observed_cooccur(activity_matrix, tetrode_mask)
    prob['shuffle'] = shuffle_cooccur(activity_matrix, num_shuffles)
    prob['zscore'] = zscore_cooccur(prob['observed'], prob['shuffle'])

    prob['expected'] = vector_from_array(prob['expected'])
    prob['observed'] = vector_from_array(prob['observed'])
    prob['zscore'] = vector_from_array(prob['zscore'])

    return prob


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


def expected_cooccur(prob_active, tetrode_mask):
    """Expected co-occurrence, multiply single cell probabilities

    Parameters
    ----------
    prob_active : np.array
        Fraction of bins each cell participates in individually
    tetrode_mask : np.array
        n_neurons by n_neurons. Boolean of True (same tetrode) or
        False (different tetrode).

    Returns
    -------
    prob_expected : np.array
        .. math:: p(x|y)
    """
    prob_expected = np.outer(prob_active, prob_active)

    # Remove probability of cell co-occurring with itself or
    # cells from the same tetrode due to inaccurate spike sorting.
    prob_expected[tetrode_mask] = np.nan

    return prob_expected


def observed_cooccur(activity_matrix, tetrode_mask):
    """Observed co-occurrences

    Parameters
    ----------
    activity_matrix : np.array
        num_neurons by num_bins, boolean (1 or 0)
    tetrode_mask : np.array
        n_neurons by n_neurons. Boolean of True (same tetrode) or
        False (different tetrode).

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

    # Remove probability of cell co-occurring with itself or
    # cells from the same tetrode due to inaccurate spike sorting.
    prob_observed[tetrode_mask] = np.nan

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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                prob_zscore[i][j] = (prob_observed[i][j] -
                                     np.nanmean(np.squeeze(prob_shuffle[:, i, j]))) / np.nanstd(np.squeeze(prob_shuffle[:, i, j]))

    return prob_zscore
