import warnings
import numpy as np


class SpikeTrain:
    """A set of spike times associated with an individual putative neuron.

    Parameters
    ----------
    time : np.array
    label : str or None, optional
        Information pertaining to the source of the spiketrain.

    Attributes
    ----------
    time : np.array
        With shape (n_samples,).
    label : str or None
        Information pertaining to the source of the spiketrain.

    """
    def __init__(self, time, label=None):
        time = np.squeeze(time).astype(float)

        if time.shape == ():
            time = time[..., np.newaxis]

        if time.ndim != 1:
            raise ValueError("time must be a vector")

        if label is not None and not isinstance(label, str):
            raise ValueError("label must be a string")

        self.time = np.sort(time)
        self.label = label

    def __getitem__(self, idx):
        return SpikeTrain(self.time[idx], self.label)

    def time_slice(self, t_starts, t_stops):
        """Creates a new object corresponding to the time slice of
        the original between (and including) times t_start and t_stop. Setting
        either parameter to None uses infinite endpoints for the time interval.

        Parameters
        ----------
        spiketrain : nept.SpikeTrain
        t_starts : float or list or None
        t_stops : float or list or None

        Returns
        -------
        sliced_spiketrain : nept.SpikeTrain
        """
        if t_starts is None:
            t_starts = [-np.inf]

        if t_stops is None:
            t_stops = [np.inf]

        if isinstance(t_starts, (int, float)):
            t_starts = [t_starts]

        if isinstance(t_stops, (int, float)):
            t_stops = [t_stops]

        if len(t_starts) != len(t_stops):
            raise ValueError("must have same number of start and stop times")

        indices = []
        for t_start, t_stop in zip(t_starts, t_stops):
            indices.append((self.time >= t_start) & (self.time <= t_stop))
        indices = np.any(np.column_stack(indices), axis=1)

        return self[indices]
