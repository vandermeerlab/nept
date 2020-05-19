import numpy as np
from shapely.geometry import Point
from nept.core.analogsignal import AnalogSignal
from nept.core.epoch import Epoch
from nept.utils import gaussian_filter


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
        if type(idx) == Epoch:
            if idx.isempty:
                return Position(np.array([[]]), np.array([]))
            else:
                return self.time_slice(idx.starts, idx.stops)
        else:
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

    def combine(self, pos):
        """Return the combined position from this position to the given 'pos'.

        Parameters
        ----------
        pos : nept.Position

        Returns
        -------
        dist : nept.Position
        """
        if self.dimensions != pos.dimensions:
            raise ValueError("'pos' must be %d dimensions" % self.dimensions)

        times = np.append(self.time, pos.time)
        sort_idx = np.argsort(times)

        times = times[sort_idx]
        data = np.concatenate((self.data, pos.data))[sort_idx]

        return Position(data, times)

    def distance(self, pos):
        """Return the euclidean distance from this position to the given 'pos'.

        Parameters
        ----------
        pos : nept.Position

        Returns
        -------
        dist : np.array
        """

        if pos.n_samples != self.n_samples:
            raise ValueError("'pos' must have %d samples" % self.n_samples)

        if self.dimensions != pos.dimensions:
            raise ValueError("'pos' must be %d dimensions" % self.dimensions)

        dist = np.zeros(self.n_samples)
        for idx in range(self.data.shape[1]):
            dist += (self.data[:, idx] - pos.data[:, idx]) ** 2
        return np.sqrt(dist)

    def linearize(self, ideal_path):
        """Projects 2D positions into an 'ideal' linear trajectory.

        Parameters
        ----------
        ideal_path : shapely.LineString

        Returns
        -------
        pos : nept.Position
            1D position.

        """
        zpos = []
        for point_x, point_y in zip(self.x, self.y):
            zpos.append(ideal_path.project(Point(point_x, point_y)))
        zpos = np.array(zpos)

        return Position(zpos, self.time)

    def speed(self, t_smooth=None):
        """Finds the speed of the animal from position.

        Parameters
        ----------
        pos : nept.Position
        t_smooth : float or None
            Range over which smoothing occurs in seconds.
            Default is None (no smoothing).

        Returns
        -------
        speed : nept.AnalogSignal
        """
        speed = self[1:].distance(self[:-1])
        speed /= np.diff(self.time)
        speed = np.hstack(([0], speed))

        if t_smooth is not None:
            dt = np.median(np.diff(self.time))
            speed = gaussian_filter(speed, std=t_smooth, dt=dt)

        return AnalogSignal(speed, self.time)
