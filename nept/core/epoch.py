import numpy as np
import nept


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
        return Epoch(np.hstack([np.array(self.starts[idx])[..., np.newaxis],
                                np.array(self.stops[idx])[..., np.newaxis]]))

    @property
    def centers(self):
        """(np.array) The center of each epoch."""
        return np.mean(self.time, axis=1)

    @property
    def durations(self):
        """(np.array) The duration of each epoch."""
        return self.time[:, 1] - self.time[:, 0]

    @property
    def isempty(self):
        """(boolean) Whether the epoch array is empty."""
        if self.time.size == 0:
            return True
        else:
            return False

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
        epoch : nept.Epoch

        Returns
        -------
        intersect_epochs : nept.Epoch

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
        epoch : nept.Epoch

        Returns
        -------
        overlaps_epochs : nept.Epoch

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
        merged_epochs : nept.Epoch

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
        expanded_epochs : nept.Epoch

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
        shrinked_epochs : nept.Epoch

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
        epoch : nept.Epoch

        Returns
        -------
        joined_epochs : nept.Epoch

        """
        join_starts = np.concatenate((self.starts, epoch.starts))
        join_stops = np.concatenate((self.stops, epoch.stops))

        return Epoch(join_starts, join_stops-join_starts)

    def contains(self, value):
        """Checks whether value is in any epoch.

        Parameters
        ----------
        epochs: nept.Epoch
        value: float or int

        Returns
        -------
        boolean

        """
        for start, stop in zip(self.starts, self.stops):
            if start <= value <= stop:
                return True
        return False
