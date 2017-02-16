import os
import numpy as np
import nept


def load_mclust_header(filename):
    """Loads a mclust .t tetrode file.

    Parameters
    ----------
    filename: str

    Returns
    -------
    times: np.array

    """
    # The format for a .t file according the the mclust docs is
    # header - beginning with %%BEGINHEADER and ending with %%ENDHEADER
    # uint64 - timestamp in tenths of ms
    f = open(filename, 'rb')
    file_contents = f.read()

    # Here we separate the header from the timestamps (data).
    header_begin_idx = file_contents.find(b'%%BEGINHEADER')
    header_end_idx = file_contents.find(b'%%ENDHEADER') + len(b'%%ENDHEADER\n')
    header = file_contents[header_begin_idx:header_end_idx]

    f.close()

    return header


def load_mclust_t(filename):
    """Loads a mclust .t tetrode file.

    Parameters
    ----------
    filename: str

    Returns
    -------
    times: np.array

    """
    # The format for a .t file according the the mclust docs is
    # header - beginning with %%BEGINHEADER and ending with %%ENDHEADER
    # uint64 - timestamp in tenths of ms (big endian)
    f = open(filename, 'rb')
    file_contents = f.read()

    # Here we separate the mclust header from the timestamps (data).
    header_begin_idx = file_contents.find(b'%%BEGINHEADER')
    header_end_idx = file_contents.find(b'%%ENDHEADER') + len(b'%%ENDHEADER\n')
    header = file_contents[header_begin_idx:header_end_idx]

    data = file_contents[header_end_idx:]
    spike_times = np.fromstring(data, dtype=np.dtype('>Q'))

    # Spikes times are in timestamps (tenths of ms).
    # Let's convert the timestamps to seconds.
    spikes = spike_times / 1e4

    f.close()

    return spikes


def get_spiketrain(spike_times, label):
    """Converts spike times to nept.SpikeTrain.

    Parameters
    ----------
    spike_times: np.array
    label: str

    Returns
    -------
    spiketrain: nept.SpikeTrain

    """

    return nept.SpikeTrain(spike_times, label)


def load_spikes(filepath, load_questionable=True):
    """Loads spikes from multiple tetrode spike files from a given session.

    Parameters
    ----------
    filepath: str
        Session folder
    load_questionable: boolean
        Loads ``*.t`` and ``*._t`` spiketrains if True (default).

    Returns
    -------
    spikes: list of nept.SpikeTrain

    """
    spikes = []

    for file in os.listdir(filepath):
        if file.endswith(".t"):
            label = file[18:20]
            spiketrain = get_spiketrain(load_mclust_t(os.path.join(filepath, file)), label)
            spikes.append(spiketrain)

        if load_questionable:
            if file.endswith("._t"):
                label = file[18:20]
                spiketrain = get_spiketrain(load_mclust_t(os.path.join(filepath, file)), label)
                spikes.append(spiketrain)

    return np.array(spikes)
