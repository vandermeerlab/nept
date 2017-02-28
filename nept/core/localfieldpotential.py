from nept.core.analogsignal import AnalogSignal


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
