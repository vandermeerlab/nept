import warnings
import numpy as np
import nept


class AnalogSignal:
    """A continuous analog timestamped signal.

    Parameters
    ----------
    data : np.array
    time : np.array

    Attributes
    ----------
    data : np.array
        With shape (n_samples, dimensionality).
    time : np.array
        With shape (n_samples,).

    """
    def __init__(self, data, time):
        data = np.squeeze(data).astype(float)
        time = np.squeeze(time).astype(float)

        if time.ndim == 0:
            time = time[..., np.newaxis]
        elif time.ndim != 1:
            raise ValueError("time must be a vector")

        if data.ndim == 0:
            data = data.reshape((1, time.shape[0]))
        elif data.ndim == 1:
            if time.shape[0] == 1:
                data = data[np.newaxis, ...]
            elif time.shape[0] == data.shape[0]:
                data = data[..., np.newaxis]
            else:
                raise ValueError("data and time should be the same length")

        elif data.ndim > 2:
            raise ValueError("data must be vector or 2D array")

        if data.shape[0] != data.shape[1] and time.shape[0] == data.shape[1]:
            warnings.warn("data should be shape (timesteps, dimensionality); "
                          "got (dimensionality, timesteps). Correcting...")
            data = data.T

        if time.shape[0] != data.shape[0]:
            raise ValueError("must have same number of time and data samples")

        self.data = data
        self.time = time

    def __getitem__(self, idx):
        return AnalogSignal(self.data[idx], self.time[idx])

    @property
    def dimensions(self):
        """(int) Dimensionality of data attribute."""
        return self.data.shape[1]

    @property
    def n_samples(self):
        """(int) Number of samples."""
        return self.time.size

    @property
    def isempty(self):
        """(bool) Empty AnalogSignal."""
        if len(self.time) == 0:
            empty = True
        else:
            empty = False
        return empty

    def time_slice(self, t_starts, t_stops):
        """Creates a new object corresponding to the time slice of
        the original between (and including) times t_start and t_stop. Setting
        either parameter to None uses infinite endpoints for the time interval.

        Parameters
        ----------
        analogsignal : nept.AnalogSignal
        t_starts : float or list or None
        t_stops : float or list or None

        Returns
        -------
        sliced_analogsignal : nept.AnalogSignal
        """
        if isinstance(t_starts, (int, float)) or t_starts is None:
            t_starts = [t_starts]

        if any(element is None for element in t_starts):
            t_starts = [min(self.time) if t_start is None else t_start for t_start in t_starts]

        if isinstance(t_stops, (int, float)) or t_stops is None:
            t_stops = [t_stops]

        if any(element is None for element in t_stops):
            t_stops = [max(self.time) if t_start is None else t_start for t_start in t_stops]

        if len(t_starts) != len(t_stops):
            raise ValueError("must have same number of start and stop times")

        indices = []
        for t_start, t_stop in zip(t_starts, t_stops):
            indices.append((self.time >= t_start) & (self.time <= t_stop))
        indices = np.any(np.column_stack(indices), axis=1)

        return self[indices]
