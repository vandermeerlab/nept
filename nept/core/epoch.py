import numpy as np
import nept


class Epoch:
    """An array of epochs, where each epoch has a start and stop time.

    Parameters
    ----------
    starts : np.array
    stops : np.array

    Attributes
    ----------
    starts : np.array
        The start times for each epoch. With shape (n_epochs,).
    stops : np.array
        The stop times for each epoch. With shape (n_epochs,).

    """

    def __init__(self, starts, stops):
        starts = np.atleast_1d(np.squeeze(starts).astype(float))
        stops = np.atleast_1d(np.squeeze(stops).astype(float))

        if starts.shape[0] != stops.shape[0]:
            raise ValueError("must have the same number of start and stop times")

        if starts.ndim > 1 or stops.ndim > 1:
            raise ValueError("time cannot have more than 1 dimension.")

        if np.any(stops - starts <= 0):
            raise ValueError("start must be less than stop")

        sort_idx = np.argsort(starts)
        starts = starts[sort_idx]
        stops = stops[sort_idx]

        self.starts = starts
        self.stops = stops

    def __getitem__(self, idx):
        return Epoch(self.starts[idx], self.stops[idx])

    def __iter__(self):
        for start, stop in zip(self.starts, self.stops):
            yield Epoch(start, stop)

    @property
    def time(self):
        """(np.array) The times of the epochs."""
        return np.concatenate(
            np.array([self.starts, self.stops])[..., np.newaxis], axis=1
        )

    @property
    def centers(self):
        """(np.array) The center of each epoch."""
        return np.mean(self.time, axis=1)

    @property
    def durations(self):
        """(np.array) The duration of each epoch."""
        return self.stops - self.starts

    @property
    def isempty(self):
        """(boolean) Whether the epoch array is empty."""
        if self.time.size == 0:
            return True
        else:
            return False

    @property
    def start(self):
        """(np.array) The start of the first epoch."""
        return self.starts[0]

    @property
    def stop(self):
        """(np.array) The stop of the last epoch."""
        return self.stops[-1]

    @property
    def n_epochs(self):
        """(int) The number of epochs."""
        return len(self.starts)

    def copy(self):
        new_starts = np.array(self.starts)
        new_stops = np.array(self.stops)
        return Epoch(new_starts, new_stops)

    def contains(self, value, edge=True):
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
            if edge:
                if start <= value <= stop:
                    return True
            else:
                if start < value < stop:
                    return True
        return False

    def excludes(self, epoch):
        """Excludes the intersection between two sets of epochs.

        Parameters
        ----------
        epoch : nept.Epoch

        Returns
        -------
        without_intersect_epochs : nept.Epoch

        """
        if len(epoch.starts) == 0:
            return Epoch(np.array(self.starts), np.array(self.stops))

        new_starts = []
        new_stops = []
        epoch_a = self.copy().merge()
        epoch_b = epoch.copy().merge()

        for aa in epoch_a.time:
            aa = Epoch(aa[0], aa[1])
            for bb in epoch_b.time:
                bb = Epoch(bb[0], bb[1])
                if not aa.overlaps(bb).isempty:
                    if aa.contains(bb.start, edge=False) and aa.contains(bb.stop):
                        new_starts.append(aa.start)
                        new_stops.append(bb.start)
                        new_starts.append(bb.stop)
                        new_stops.append(aa.stop)
                    elif aa.contains(bb.start, edge=False) and not aa.contains(bb.stop):
                        new_starts.append(aa.start)
                        new_stops.append(bb.start)
                    elif not aa.contains(bb.start, edge=False) and aa.contains(bb.stop):
                        new_starts.append(bb.stop)
                        new_stops.append(aa.stop)
                    elif bb.contains(aa.start, edge=False) and bb.contains(aa.stop):
                        continue
                else:
                    new_starts.append(aa.start)
                    new_stops.append(aa.stop)

        return Epoch(np.array(new_starts), np.array(new_stops))

    def expand(self, amount, direction="both"):
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
        if direction == "both":
            resize_starts = self.time[:, 0] - amount
            resize_stops = self.time[:, 1] + amount
        elif direction == "start":
            resize_starts = self.time[:, 0] - amount
            resize_stops = self.time[:, 1]
        elif direction == "stop":
            resize_starts = self.time[:, 0]
            resize_stops = self.time[:, 1] + amount
        else:
            raise ValueError("direction must be 'both', 'start', or 'stop'")

        return Epoch(resize_starts, resize_stops)

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
            aa = Epoch(aa[0], aa[1])
            for bb in epoch_b.time:
                bb = Epoch(bb[0], bb[1])
                if bb.contains(aa.start) and bb.contains(aa.stop):
                    new_starts.append(aa.start)
                    new_stops.append(aa.stop)
                elif aa.contains(bb.start) and aa.contains(bb.stop):
                    new_starts.append(bb.start)
                    new_stops.append(bb.stop)
                elif aa.contains(bb.start, edge=False) and not aa.contains(bb.stop):
                    new_starts.append(bb.start)
                    new_stops.append(aa.stop)
                elif not aa.contains(bb.start) and aa.contains(bb.stop, edge=False):
                    new_starts.append(aa.start)
                    new_stops.append(bb.stop)

        return Epoch(np.array(new_starts), np.array(new_stops))

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

        return Epoch(join_starts, join_stops)

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

        if len(epoch.starts) == 0:
            return epoch

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
                new_starts.append(epoch.starts[i + 1])

        new_stops.append(max(epoch.stops))

        new_starts = np.array(new_starts)
        new_stops = np.array(new_stops)

        return Epoch(new_starts, new_stops)

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
            aa = Epoch(aa[0], aa[1])
            for bb in epoch_interest.time:
                bb = Epoch(bb[0], bb[1])
                if (
                    aa.contains(bb.start)
                    or aa.contains(bb.stop)
                    or bb.contains(aa.start)
                    or bb.contains(aa.stop)
                ):
                    new_starts.append(bb.start)
                    new_stops.append(bb.stop)

        new_starts = np.unique(new_starts)
        new_stops = np.unique(new_stops)

        return Epoch(np.array(new_starts), np.array(new_stops))

    def shrink(self, amount, direction="both"):
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
        if amount > both_limit and direction == "both":
            raise ValueError("shrink amount too large")

        single_limit = min(self.durations)
        if amount > single_limit and direction != "both":
            raise ValueError("shrink amount too large")

        return self.expand(-amount, direction)

    def time_slice(self, t_start, t_stop):
        """Creates a new object corresponding to the time slice of
            the original between (and including) times t_start and t_stop.

            Parameters
            ----------
            analogsignal : nept.Epoch
            t_start : float
            t_stop : float

            Returns
            -------
            sliced_epoch : nept.Epoch
            """
        new_starts = []
        new_stops = []

        for start, stop in zip(self.starts, self.stops):
            if (start >= t_start) and (start < t_stop):
                new_starts.append(start)
                if (stop > t_start) and (stop <= t_stop):
                    new_stops.append(stop)
                else:
                    new_stops.append(t_stop)
            elif (stop > t_start) and (stop <= t_stop):
                new_stops.append(stop)
                if (start >= t_start) and (start < t_stop):
                    new_starts.append(start)
                else:
                    new_starts.append(t_start)

        return Epoch(np.array(new_starts), np.array(new_stops))
