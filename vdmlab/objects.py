import warnings

import numpy as np
from shapely.geometry import Point

from .utils import find_nearest_idx


class AnalogSignal:
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
                          "got (dimensionality, timesteps). Correcting.")
            data = data.T
        if time.shape[0] != data.shape[0]:
            raise ValueError("must have same number of time and data samples")

        self.data = data
        self.time = time

    def __getitem__(self, idx):
        return AnalogSignal(self.data[idx], self.time[idx])


class Position(AnalogSignal):

    def __getitem__(self, idx):
        return Position(self.data[idx], self.time[idx])

    @property
    def dimensions(self):
        return self.data.shape[1]

    @property
    def n_samples(self):
        return self.time.size

    @property
    def x(self):
        return self.data[:, 0]

    @property
    def y(self):
        if self.data.shape[1] < 2:
            raise ValueError("can't get 'y' of one-dimensional position")
        return self.data[:, 1]

    def distance(self, pos):
        """ Return the euclidean distance from this pos to the given 'pos'.

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

    def linearize(self, ideal_path, trial_start, trial_stop):
        """ Projects 2D positions into an 'ideal' linear trajectory.

        Parameters
        ----------
        ideal_path : shapely.LineString
        trial_start : float
        trial_stop : float

        Returns
        -------
        pos : vdmlab.Position
            1D position.

        """
        t_start_idx = find_nearest_idx(self.time, trial_start)
        t_end_idx = find_nearest_idx(self.time, trial_stop)
        pos_trial = self[t_start_idx:t_end_idx]

        zpos = []
        for point_x, point_y in zip(pos_trial.x, pos_trial.y):
            zpos.append(ideal_path.project(Point(point_x, point_y)))

        return Position(zpos, pos_trial.time)

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
