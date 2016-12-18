# Adapted from nlxio written by Bernard Willards <https://github.com/bwillers/nlxio>

import numpy as np


def load_nev(filename):
    """Loads a neuralynx .nev file.

    Parameters
    ----------
    filename: str

    Returns
    -------
    nev_data: dict
        With time (uint64), id (uint16), nttl (uint16), and event_str (charx128) as the most usable keys.

    """

    f = open(filename, 'rb')

    # There's nothing useful in the header for .nev files, so skip past it
    f.seek(2 ** 14)

    # An event record is as follows:
    # int16 - nstx - reserved
    # int16 - npkt_id - id of the originating system
    # int16 - npkt_data_size - this value should always be 2
    # uint64 - timestamp, microseconds
    # int16 - nevent_id - ID value for event
    # int16 - nttl - decimal TTL value read from the TTL input port
    # int16 - ncrc - record crc check, not used in consumer applications
    # int16 - ndummy1 - reserved
    # int16 - ndummy2 - reserved
    # int32x8 - dnExtra - extra bit values for this event
    # string(128) - event string
    dt = np.dtype([('filler1', '<h', 3), ('time', '<Q'), ('id', '<h'),
                   ('nttl', '<h'), ('filler2', '<h', 3), ('extra', '<i', 8),
                   ('event_str', np.dtype('a128'))])
    nev_data = np.fromfile(f, dt)

    return nev_data


def load_events(filename, labels):
    nev_data = load_nev(filename)

    idx = {label: [] for label in labels}
    for label in labels:
        for i, event in enumerate(nev_data['event_str']):
            if event.decode() == label:
                idx[label].append(i)

    timestamps = {label: [] for label in labels}

    times = nev_data['time'].astype(float) * 1e-6

    for label in labels:
        timestamps[label] = times[idx[label]]

    return timestamps

