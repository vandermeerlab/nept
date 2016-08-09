import numpy as np

from .utils import find_nearest_idx


def consecutive(array, stepsize=1):
    """

    Parameters
    ----------
    array : np.array

    Returns
    -------
    List of np.arrays, split when jump greater than stepsize

    """

    return np.split(array, np.where(np.diff(array) != stepsize)[0]+1)


def find_fields(tuning, hz_thresh=5, min_length=1, max_length=20, max_mean_firing=10):
    """

    Parameters
    ----------
    tuning : list of np.arrays
        Where each inner array contains the tuning curves for an
        individual neuron.
    hz_thresh : int or float
        Any bin with firing above this value is considered to be part of a field.
    min_length : int
        Minimum length of field (in tuning curve bin units, eg. if bin size is 3cm,
        min_length=1 is 3cm.
    max_length : int
        Maximum length of field (in tuning curve bin units, eg. if bin size is 3cm,
        min_length=10 is 3cm*10 = 30cm.
    max_mean_firing : int or float
        Only neurons with a mean firing rate less than this amount are considered for
        having place fields. The default is set to 10.

    Returns
    -------
    with_fields : dict
        Where the key is the neuron number (int), value is a list of arrays (int)
        that are indices into the tuning curve where the field occurs.
        Each inner array contains the indices for a given place field.

    """

    fields = []
    for neuron_tc in tuning:
        if np.mean(neuron_tc) < max_mean_firing:
            neuron_field = np.zeros(neuron_tc.shape[0])
            for i, this_bin in enumerate(neuron_tc):
                if this_bin > hz_thresh:
                    neuron_field[i] = 1
            fields.append(neuron_field)
        else:
            fields.append(np.array([]))

    fields_idx = dict()
    for i, neuron_fields in enumerate(fields):
        field_idx = np.nonzero(neuron_fields)[0]
        fields_idx[i] = consecutive(field_idx)

    with_fields = dict()
    for key in fields_idx:
        for field in fields_idx[key]:
            if len(field) > max_length:
                continue
            elif min_length <= len(field):
                with_fields[key] = fields_idx[key]
                continue
    return with_fields


def get_single_field(fields):
    """Finds neurons with and indices of single fields.

    Parameters
    ----------
    fields : dict
        Where the key is the neuron number (int), value is a list of arrays (int).
        Each inner array contains the indices for a given place field.
        Eg. Neurons 7, 3, 11 that have 2, 1, and 3 place fields respectively would be:
        {7: [[field], [field]], 3: [[field]], 11: [[field], [field], [field]]}

    Returns
    -------
    fields : dict
        Where the key is the neuron number (int), value is a list of arrays (int).
        Each inner array contains the indices for a given place field.
        Eg. For the above input, only neuron 3 would be output in this dict:
        {3: [[field]]}

    """
    fields_single = dict()
    for neuron in fields.keys():
        if len(fields[neuron]) == 1:
            fields_single[neuron] = fields[neuron]
    return fields_single


def get_heatmaps(neuron_list, spikes, pos, num_bins=100):
    """ Gets the 2D heatmaps for firing of a given set of neurons.

    Parameters
    ----------
    neuron_list : list of ints
        These will be the indices into the full list of neuron spike times
    spikes : dict
        With times(float), labels (str) as keys
    pos : dict
        With time(float), x(float), y(float) as keys
    num_bins : int
        This will specify how the 2D space is broken up, the greater the number
        the more specific the heatmap will be. The default is set at 100.

    Returns
    -------
    heatmaps : dict of lists
        Where the key is the neuron number and the value is the heatmap for
        that individual neuron.

    """
    xedges = np.linspace(np.min(pos['x'])-2, np.max(pos['x'])+2, num_bins+1)
    yedges = np.linspace(np.min(pos['y'])-2, np.max(pos['y'])+2, num_bins+1)

    heatmaps = dict()
    count = 1
    for neuron in neuron_list:
        field_x = []
        field_y = []
        for spike in spikes['time'][neuron]:
            spike_idx = find_nearest_idx(pos['time'], spike)
            field_x.append(pos['x'][spike_idx])
            field_y.append(pos['y'][spike_idx])
            heatmap, out_xedges, out_yedges = np.histogram2d(field_x, field_y, bins=[xedges, yedges])
        heatmaps[neuron] = heatmap.T
        print(str(neuron) + ' of ' + str(len(neuron_list)))
        count += 1
    return heatmaps

