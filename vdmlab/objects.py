import warnings

import numpy as np
from shapely.geometry import Point

import vdmlab as vdm


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


    def time_slices(self, t_starts, t_stops):
        """Creates a new object corresponding to the time slice of
        the original between (and including) times t_start and t_stop. Setting
        either parameter to None uses infinite endpoints for the time interval.

        Parameters
        ----------
        analogsignal : vdmlab.AnalogSignal
        t_starts : list of floats
        t_stops : list of floats

        Returns
        -------
        sliced_analogsignal : vdmlab.AnalogSignal
        """
        if len(t_starts) != len(t_stops):
            raise ValueError("must have same number of start and stop times")

        indices = []
        for t_start, t_stop in zip(t_starts, t_stops):
            indices.append((self.time >= t_start) & (self.time <= t_stop))
        indices = np.any(np.column_stack(indices), axis=1)

        return self[indices]


class Epoch:
    """An array of epochs, where each epoch has a start and stop time.

    Parameters
    ----------
    time : np.array
        If shape (n_epochs, 1) or (n_epochs,), the start time for each epoch.
        If shape (n_epochs, 2), the start and stop times for each epoch.
    duration : np.array or None, optional
        The length of the epoch.

    Attributes
    ----------
    time : np.array
        The start and stop times for each epoch. With shape (n_epochs, 2).

    """
    def __init__(self, time, duration=None):
        time = np.squeeze(time).astype(float)

        if time.ndim == 0:
            time = time[..., np.newaxis]

        if duration is not None:
            duration = np.squeeze(duration).astype(float)
            if duration.ndim == 0:
                duration = duration[..., np.newaxis]

            if time.ndim == 2 and duration.ndim == 1:
                raise ValueError("duration not allowed when using start and stop times")

            if time.ndim == 1 and time.shape[0] != duration.shape[0]:
                raise ValueError("must have same number of time and duration samples")

            if time.ndim == 1 and duration.ndim == 1:
                stop_epoch = time + duration
                time = np.hstack((time[..., np.newaxis], stop_epoch[..., np.newaxis]))

        if time.ndim == 1 and duration is None:
            time = time[..., np.newaxis]

        if time.ndim == 2 and time.shape[1] != 2:
            time = np.hstack((time[0][..., np.newaxis], time[1][..., np.newaxis]))

        if time.ndim > 2:
            raise ValueError("time cannot have more than 2 dimensions")

        if time[:, 0].shape[0] != time[:, 1].shape[0]:
            raise ValueError("must have the same number of start and stop times")

        if time.ndim == 2 and np.any(time[:, 1] - time[:, 0] <= 0):
            raise ValueError("start must be less than stop")

        sort_idx = np.argsort(time[:, 0])
        time = time[sort_idx]

        self.time = time

    def __getitem__(self, idx):
        return Epoch(np.array([self.starts[idx], self.stops[idx]]))

    @property
    def centers(self):
        """(np.array) The center of each epoch."""
        return np.mean(self.time, axis=1)

    @property
    def durations(self):
        """(np.array) The duration of each epoch."""
        return self.time[:, 1] - self.time[:, 0]

    @property
    def starts(self):
        """(np.array) The start of each epoch."""
        return self.time[:, 0]

    @property
    def start(self):
        """(np.array) The start of the first epoch."""
        return self.time[:, 0][0]

    @property
    def stops(self):
        """(np.array) The stop of each epoch."""
        return self.time[:, 1]

    @property
    def stop(self):
        """(np.array) The stop of the last epoch."""
        return self.time[:, 1][-1]

    @property
    def n_epochs(self):
        """(int) The number of epochs."""
        return len(self.time[:, 0])

    def copy(self):
        new_starts = np.array(self.starts)
        new_stops = np.array(self.stops)
        return Epoch(new_starts, new_stops-new_starts)


    def intersect(self, epoch):
        """Finds intersection between two sets of epochs.

        Parameters
        ----------
        epoch : vdmlab.Epoch

        Returns
        -------
        intersect_epochs : vdmlab.Epoch

        """
        if len(self.starts) == 0 or len(epoch.starts) == 0:
            return Epoch([], [])

        new_starts = []
        new_stops = []
        epoch_a = self.copy().merge()
        epoch_b = epoch.copy().merge()

        for aa in epoch_a.time:
            for bb in epoch_b.time:
                if (aa[0] <= bb[0] < aa[1]) and (aa[0] < bb[1] <= aa[1]):
                    new_starts.append(bb[0])
                    new_stops.append(bb[1])
                elif (aa[0] < bb[0] < aa[1]) and (aa[0] < bb[1] > aa[1]):
                    new_starts.append(bb[0])
                    new_stops.append(aa[1])
                elif (aa[0] > bb[0] < aa[1]) and (aa[0] < bb[1] < aa[1]):
                    new_starts.append(aa[0])
                    new_stops.append(bb[1])
                elif (aa[0] >= bb[0] < aa[1]) and (aa[0] < bb[1] >= aa[1]):
                    new_starts.append(aa[0])
                    new_stops.append(aa[1])

        return Epoch(np.hstack([np.array(new_starts)[..., np.newaxis],
                                np.array(new_stops)[..., np.newaxis]]))


    def overlaps(self, epoch):
        """Finds overlap between template epochs and epoch of interest.

        Parameters
        ----------
        epoch : vdmlab.Epoch

        Returns
        -------
        overlaps_epochs : vdmlab.Epoch

        """
        if len(self.starts) == 0 or len(epoch.starts) == 0:
            return Epoch([], [])

        new_starts = []
        new_stops = []
        template = self.copy().merge()
        epoch_interest = epoch.copy().merge()

        for aa in template.time:
            for bb in epoch_interest.time:
                if (aa[0] <= bb[0] < aa[1]) and (aa[0] < bb[1] <= aa[1]):
                    new_starts.append(bb[0])
                    new_stops.append(bb[1])
                elif (aa[0] < bb[0] < aa[1]) and (aa[0] < bb[1] > aa[1]):
                    new_starts.append(bb[0])
                    new_stops.append(bb[1])
                elif (aa[0] > bb[0] < aa[1]) and (aa[0] < bb[1] < aa[1]):
                    new_starts.append(bb[0])
                    new_stops.append(bb[1])
                elif (aa[0] >= bb[0] < aa[1]) and (aa[0] < bb[1] >= aa[1]):
                    new_starts.append(bb[0])
                    new_stops.append(bb[1])

        new_starts = np.unique(new_starts)
        new_stops = np.unique(new_stops)

        return Epoch(np.hstack([np.array(new_starts)[..., np.newaxis],
                                np.array(new_stops)[..., np.newaxis]]))

    def merge(self, gap=0.0):
        """Merges epochs that are close or overlapping.

        Parameters
        ----------
        gap : float, optional
            Amount (in time) to consider epochs close enough to merge.
            Defaults to 0.0 (no gap).

        Returns
        -------
        merged_epochs : vdmlab.Epoch

        """
        if gap < 0:
            raise ValueError("gap cannot be negative")

        epoch = self.copy()

        stops = epoch.stops[:-1] + gap
        starts = epoch.starts[1:]
        to_merge = (stops - starts) >= 0

        new_starts = [epoch.starts[0]]
        new_stops = []

        next_stop = epoch.stops[0]
        for i in range(epoch.time.shape[0] - 1):
            this_stop = epoch.stops[i]
            next_stop = max(next_stop, this_stop)
            if not to_merge[i]:
                new_stops.append(next_stop)
                new_starts.append(epoch.starts[i+1])

        new_stops.append(epoch.stops[-1])

        new_starts = np.array(new_starts)
        new_stops = np.array(new_stops)

        return Epoch(new_starts, new_stops-new_starts)

    def expand(self, amount, direction='both'):
        """Expands epoch by the given amount.

        Parameters
        ----------
        amount : float
            Amount (in time) to expand each epoch.
        direction : str
            Can be 'both', 'start', or 'stop'. This specifies
            which direction to resize epoch.

        Returns
        -------
        expanded_epochs : vdmlab.Epoch

        """
        if direction == 'both':
            resize_starts = self.time[:, 0] - amount
            resize_stops = self.time[:, 1] + amount
        elif direction == 'start':
            resize_starts = self.time[:, 0] - amount
            resize_stops = self.time[:, 1]
        elif direction == 'stop':
            resize_starts = self.time[:, 0]
            resize_stops = self.time[:, 1] + amount
        else:
            raise ValueError("direction must be 'both', 'start', or 'stop'")

        return Epoch(np.hstack((resize_starts[..., np.newaxis],
                                resize_stops[..., np.newaxis])))

    def shrink(self, amount, direction='both'):
        """Shrinks epoch by the given amount.

        Parameters
        ----------
        amount : float
            Amount (in time) to shrink each epoch.
        direction : str
            Can be 'both', 'start', or 'stop'. This specifies
            which direction to resize epoch.

        Returns
        -------
        shrinked_epochs : vdmlab.Epoch

        """
        both_limit = min(self.durations / 2)
        if amount > both_limit and direction == 'both':
            raise ValueError("shrink amount too large")

        single_limit = min(self.durations)
        if amount > single_limit and direction != 'both':
            raise ValueError("shrink amount too large")

        return self.expand(-amount, direction)

    def join(self, epoch):
        """Combines two sets of epochs.

        Parameters
        ----------
        epoch : vdmlab.Epoch

        Returns
        -------
        joined_epochs : vdmlab.Epoch

        """
        join_starts = np.concatenate((self.starts, epoch.starts))
        join_stops = np.concatenate((self.stops, epoch.stops))

        return Epoch(join_starts, join_stops-join_starts)

    def contains(self, value):
        """Checks whether value is in any epoch.

        Parameters
        ----------
        epochs: vdmlab.Epoch
        value: float or int

        Returns
        -------
        boolean

        """
        for start, stop in zip(self.starts, self.stops):
            if start <= value <= stop:
                return True
        return False


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


class Neurons:
    """ A grouping of spiketrains and tuning curves

    Parameters
    ----------
    spikes : np.array
    tuning_curves : np.array

    Attributes
    ----------
    spikes : np.array
    tuning_curves : np.array

    """
    def __init__(self, spikes, tuning_curves):

        if spikes.shape[0] != tuning_curves.shape[0]:
            raise ValueError("spikes and tuning curves must have the same number of neurons")

        self.spikes = spikes
        self.tuning_curves = tuning_curves

    def __getitem__(self, idx):
        return Neurons(self.spikes[idx], self.tuning_curves[idx])

    @property
    def n_neurons(self):
        """(int) The number of neurons."""
        return len(self.spikes)

    @property
    def tuning_shape(self):
        """(tuple) The shape of the tuning curves."""
        return self.tuning_curves[0].shape

    def time_slice(self, t_start, t_stop):
        """ Gets the neuron spikes corresponding to the time slice of
        the original between (and including) times t_start and t_stop. Setting
        either parameter to None uses infinite endpoints for the time interval.

        Parameters
        ----------
        spikes : vdmlab.Neurons
        t_start : float
        t_stop : float

        Returns
        -------
        sliced_spikes : list of vdmlab.SpikeTrain

        """
        sliced_spikes = [spiketrain.time_slice(t_start, t_stop) for spiketrain in self.spikes]
        return sliced_spikes

    def time_slices(self, t_starts, t_stops):
        """ Gets the neuron spikes corresponding to the time slices of
        the original between (and including) times t_starts and t_stops. Setting
        either parameter to None uses infinite endpoints for the time interval.

        Parameters
        ----------
        spikes : vdmlab.Neurons
        t_start : float
        t_stop : float

        Returns
        -------
        sliced_spikes : list of vdmlab.SpikeTrain

        """
        sliced_spikes = [spiketrain.time_slice(t_starts, t_stops) for spiketrain in self.spikes]
        return sliced_spikes


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
        if type(idx) == vdm.objects.Epoch:
            return self.time_slices(idx.starts, idx.stops)
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

        if self.dimensions != pos.dimensions:
            raise ValueError("'pos' must be %d dimensions" % self.dimensions)

        dist = np.zeros(self.n_samples)
        for idx in range(self.data.shape[1]):
            dist += (self.data[:, idx] - pos.data[:, idx]) ** 2
        return np.sqrt(dist)

    def linearize(self, ideal_path, zone):
        """ Projects 2D positions into an 'ideal' linear trajectory.

        Parameters
        ----------
        ideal_path : shapely.LineString
        zone : shapely.Polygon

        Returns
        -------
        pos : vdmlab.Position
            1D position.

        """
        zpos = []
        for point_x, point_y in zip(self.x, self.y):
            point = Point([point_x, point_y])
            if zone.contains(point):
                zpos.append(ideal_path.project(Point(point_x, point_y)))
        zpos = np.array(zpos)

        return Position(zpos, self.time)

    def speed(self, t_smooth=None):
        """Finds the speed of the animal from position.

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
        speed = self[1:].distance(self[:-1])
        speed /= np.diff(self.time)
        speed = np.hstack(([0], speed))

        dt = np.median(np.diff(self.time))

        if t_smooth is not None:
            filter_length = np.ceil(t_smooth / dt)
            speed = np.convolve(speed, np.ones(int(filter_length))/filter_length, 'same')

        speed = speed * dt

        return AnalogSignal(speed, self.time)


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


    def time_slices(self, t_starts, t_stops):
        """Creates a new object corresponding to the time slice of
        the original between (and including) times t_start and t_stop. Setting
        either parameter to None uses infinite endpoints for the time interval.

        Parameters
        ----------
        spiketrain : vdmlab.SpikeTrain
        t_starts : list of floats
        t_stops : list of floats

        Returns
        -------
        sliced_spiketrain : vdmlab.SpikeTrain
        """
        if len(t_starts) != len(t_stops):
            raise ValueError("must have same number of start and stop times")

        indices = []
        for t_start, t_stop in zip(t_starts, t_stops):
            indices.append((self.time >= t_start) & (self.time <= t_stop))
        indices = np.any(np.column_stack(indices), axis=1)

        return self[indices]
