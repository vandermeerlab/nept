import warnings

import numpy as np
from shapely.geometry import Point


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
            data = data[np.newaxis, ...]

        if time.ndim != 1:
            raise ValueError("time must be a vector")

        if data.ndim == 1:
            data = data[..., np.newaxis]

        if data.ndim > 2:
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

    def time_slice(self, t_start, t_stop):
        """Creates a new object corresponding to the time slice of
        the original between (and including) times t_start and t_stop. Setting
        either parameter to None uses infinite endpoints for the time interval.

        Parameters
        ----------
        analogsignal : vdmlab.AnalogSignal
        t_start : float
        t_stop : float

        Returns
        -------
        sliced_analogsignal : vdmlab.AnalogSignal
        """
        if t_start is None:
            t_start = -np.inf
        if t_stop is None:
            t_stop = np.inf

        indices = (self.time >= t_start) & (self.time <= t_stop)

        return self[indices]


class LocalFieldPotential(AnalogSignal):
    """Subclass of AnalogSignal.

    Parameters
    ----------
    data : np.array
    time : np.array

    Attributes
    ----------
    data : np.array
        With shape (n_samples, 1).
    time : np.array
        With shape (n_samples,).
    """
    def __init__(self, data, time):
        super().__init__(data, time)
        if self.dimensions > 1:
            raise ValueError("can only contain one LFP")

    def __getitem__(self, idx):
        return LocalFieldPotential(self.data[idx], self.time[idx])


class Position(AnalogSignal):
    """Subclass of AnalogSignal. Handles both 1D and 2d positions.

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
    def __getitem__(self, idx):
        return Position(self.data[idx], self.time[idx])

    @property
    def x(self):
        """(np.array) The 'x' position attribute."""
        return self.data[:, 0]

    @x.setter
    def x(self, val):
        self.data[:, 0] = val

    @property
    def y(self):
        """(np.array) The 'y' position attribute for 2D position data."""
        if self.dimensions < 2:
            raise ValueError("can't get 'y' of one-dimensional position")
        return self.data[:, 1]

    @y.setter
    def y(self, val):
        if self.dimensions < 2:
            raise ValueError("can't set 'y' of one-dimensional position")
        self.data[:, 1] = val

    def distance(self, pos):
        """ Return the euclidean distance from this position to the given 'pos'.

        Parameters
        ----------
        pos : vdmlab.Position

        Returns
        -------
        dist : np.array
        """

        if pos.n_samples != self.n_samples:
            raise ValueError("'pos' must have %d samples" % self.n_samples)

        dist = np.zeros(self.n_samples)
        for idx in range(self.data.shape[1]):
            dist += (self.data[:, idx] - pos.data[:, idx]) ** 2
        return np.sqrt(dist)

    def linearize(self, ideal_path):
        """ Projects 2D positions into an 'ideal' linear trajectory.

        Parameters
        ----------
        ideal_path : shapely.LineString

        Returns
        -------
        pos : vdmlab.Position
            1D position.

        """
        zpos = []
        for point_x, point_y in zip(self.x, self.y):
            zpos.append(ideal_path.project(Point(point_x, point_y)))
        zpos = np.array(zpos)

        return Position(zpos, self.time)

    def speed(self, t_smooth=None):
        """Finds the velocity of the animal from position.

        Parameters
        ----------
        pos : vdmlab.Position
        t_smooth : float or None
            Range over which smoothing occurs in seconds.
            Default is None (no smoothing).

        Returns
        -------
        speed : vdmlab.AnalogSignal
        """
        velocity = self[1:].distance(self[:-1])
        velocity = np.hstack(([0], velocity))

        if t_smooth is not None:
            dt = np.median(np.diff(self.time))
            filter_length = np.ceil(t_smooth / dt)
            velocity = np.convolve(velocity, np.ones(int(filter_length))/filter_length, 'same')

        return AnalogSignal(velocity, self.time)


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

        self.time = time
        self.label = label

    def __getitem__(self, idx):
        return SpikeTrain(self.time[idx], self.label)

    def time_slice(self, t_start, t_stop):
        """Creates a new vdmlab.SpikeTrain corresponding to the time slice of
        the original between (and including) times t_start and t_stop. Setting
        either parameter to None uses infinite endpoints for the time interval.

        Parameters
        ----------
        spikes : vdmlab.SpikeTrain
        t_start : float
        t_stop : float

        Returns
        -------
        sliced_spikes : vdmlab.SpikeTrain
        """
        if t_start is None:
            t_start = -np.inf
        if t_stop is None:
            t_stop = np.inf

        indices = (self.time >= t_start) & (self.time <= t_stop)

        return self[indices]
