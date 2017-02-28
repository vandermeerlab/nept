import warnings
import numpy as np


class Neurons:
    """ A grouping of spiketrains and tuning curves

    Parameters
    ----------
    spikes : np.array
    tuning_curves : np.array

    Attributes
    ----------
    spikes : list of np.array
    tuning_curves : list of np.array

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

    def time_slice(self, t_starts, t_stops):
        """ Gets the neuron spikes corresponding to the time slices of
        the original between (and including) times t_starts and t_stops. Setting
        either parameter to None uses infinite endpoints for the time interval.

        Parameters
        ----------
        spikes : nept.Neurons
        t_starts : float or list or None
        t_stops : float or list or None

        Returns
        -------
        sliced_spikes : list of nept.SpikeTrain

        """
        if t_starts is None:
            t_starts = [-np.inf]

        if t_stops is None:
            t_stops = [np.inf]

        if isinstance(t_starts, (int, float)):
            t_starts = [t_starts]

        if isinstance(t_stops, (int, float)):
            t_stops = [t_stops]

        sliced_spikes = [spiketrain.time_slice(t_starts, t_stops) for spiketrain in self.spikes]

        return sliced_spikes
