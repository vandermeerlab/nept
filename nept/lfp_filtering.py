import numpy as np
import scipy.signal
import scipy.stats
import matplotlib.mlab
import nept


def butter_bandpass(signal, thresh, fs, order=4):
    """Filters signal using butterworth filter.

    Parameters
    ----------
    signal : nept.LFP
    fs : int
        Eg. 2000. Should get this from experiment-specifics.
    thresh : tuple
        With format (lowcut, highcut).
        Typically (140.0, 250.0) for sharp-wave ripple detection.
    order : int
        Default set to 4.

    Returns
    -------
    filtered_butter : np.array

    """
    signal = np.squeeze(signal)
    nyquist = 0.5 * fs

    b, a = scipy.signal.butter(order, [thresh[0]/nyquist, thresh[1]/nyquist], btype='band')
    filtered_butter = scipy.signal.filtfilt(b, a, signal)

    return filtered_butter


def detect_swr_hilbert(lfp, fs, thresh, z_thresh, merge_thresh, min_length, times_for_z=None):
    """Finds sharp-wave ripple (SWR) times and indices.

    Parameters
    ----------
    lfp : nept.LocalFieldPotential
    fs : int
        Experiment-specific, something in the range of 2000 typical.
    thresh : tuple
        With format (lowcut, highcut).
        Typically (140.0, 250.0) for sharp-wave ripple detection.
    z_thres : int or float
    min_length : float
        Any sequence less than this amount is not considered a sharp-wave ripple.
    times_for_z : nept.Epoch or None
        Portion of LFP used for z thresh.

    Returns
    -------
    swrs : nept.Epoch
        Containing nept.LocalFieldPotential for each SWR event

    """
    # Filtering signal with butterworth fitler
    filtered_butter = butter_bandpass(lfp.data, thresh, fs)

    # Get LFP power (using Hilbert) and z-score the power
    # Zero padding to nearest regular number to speed up fast fourier transforms (FFT) computed in the hilbert function.
    # Regular numbers are composites of the prime factors 2, 3, and 5.
    hilbert_n = next_regular(lfp.n_samples)
    power = np.abs(scipy.signal.hilbert(filtered_butter, N=hilbert_n))

    # removing the zero padding now that the power is computed
    power_lfp = nept.AnalogSignal(power[:lfp.n_samples], lfp.time)

    swrs = get_epoch_from_zscored_thresh(power_lfp, thresh=z_thresh, times_for_z=times_for_z)

    # Merging epochs that are closer - in time - than the merge_threshold.
    swrs = swrs.merge(gap=merge_thresh)

    # Removing epochs that are shorter - in time - than the min_length value.
    keep_indices = swrs.durations >= min_length
    swrs = nept.Epoch(swrs.starts[keep_indices], swrs.stops[keep_indices])

    return swrs


def get_epoch_from_zscored_thresh(power_lfp, thresh, times_for_z):
    if times_for_z is not None:
        # Apply zscore thresh to restricted data to find power thresh
        sliced_power_lfp = power_lfp.time_slice(times_for_z.start, times_for_z.stop)
        zpower = scipy.stats.zscore(np.squeeze(sliced_power_lfp.data))

        zthresh_idx = (np.abs(zpower - thresh)).argmin()
        power_thresh = sliced_power_lfp.data[zthresh_idx][0]
    else:
        zpower = scipy.stats.zscore(np.squeeze(power_lfp.data))
        zthresh_idx = (np.abs(zpower - thresh)).argmin()
        power_thresh = power_lfp.data[zthresh_idx][0]

    # Finding locations where the power changes
    detect = np.squeeze(power_lfp.data) > power_thresh
    detect = np.hstack([0, detect, 0])  # pad to detect first or last element change
    signal_change = np.diff(detect.astype(int))

    start_swr_idx = np.where(signal_change == 1)[0]
    stop_swr_idx = np.where(signal_change == -1)[0]

    # Getting times associated with these power changes
    start_time = power_lfp.time[start_swr_idx]
    stop_time = power_lfp.time[stop_swr_idx]

    return nept.Epoch(start_time, stop_time)


def next_regular(target):
    """Finds the next regular number greater than or equal to target.

    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to fast-fourier transforms (FFTPACK).

    Parameters
    ----------
    target : positive int

    Returns
    -------
    match : int

    Notes
    -----
    This function was taken from the scipy.signal.signaltools module.
    See http://scipy.org/scipylib/
    """
    if target <= 6:
        print(target)
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            p2 = 2**((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def power_in_db(power):
    """Computes the power in dB for plotting

    Parameters
    ----------
    power : np.array

    Returns
    -------
    np.array
    """
    return 10*np.log10(power)


def mean_psd(perievent_lfps, window, fs):
    """Computes the mean Power Spectral Density (PSD) of perievent slices

    Parameters
    ----------
    perievent_lfps : nept.AnalogSignal
    window : int
    fs : int

    Returns
    -------
    freq : np.array
    power : np.array

    """
    power = np.zeros((window+1, perievent_lfps.dimensions))
    for i, lfp in enumerate(perievent_lfps.data.T):
        power[:, i], freq = matplotlib.mlab.psd(
            lfp, Fs=fs, NFFT=int(window*2), noverlap=int(window/2))

    return freq, np.mean(power, axis=1)


def mean_csd(perievent_lfp1, perievent_lfp2, window, fs):
    """Computes the mean Cross-Spectral Density (CSD) between perievent slices

    Parameters
    ----------
    perievent_lfp1 : nept.AnalogSignal
    perievent_lfp2 : nept.AnalogSignal
    window : int
    fs : int

    Returns
    -------
    freq : np.array
    power : np.array

    """
    freq, power = scipy.signal.csd(perievent_lfp1.data.T, perievent_lfp2.data.T,
                                   fs=fs, nperseg=window, nfft=int(window*2))

    return freq, np.mean(power, axis=0)


def mean_coherence(perievent_lfp1, perievent_lfp2, window, fs):
    """Computes the mean coherence between perievent slices

    Parameters
    ----------
    perievent_lfp1 : nept.AnalogSignal
    perievent_lfp2 : nept.AnalogSignal
    window : int
    fs : int

    Returns
    -------
    freq : np.array
    coherence : np.array

    """
    freq, coherence = scipy.signal.coherence(
        perievent_lfp1.data.T, perievent_lfp2.data.T, fs=fs, nperseg=window, nfft=int(window*2))

    return freq, np.mean(coherence, axis=0)


def mean_coherencegram(perievent_lfp1, perievent_lfp2, dt, window, fs, extend=0.3):
    """ Computes the mean coherence over time between perievent slices
        (e.g. "coherencegram" because it's a combination of a coherence and a spectrogram)

    Parameters
    ----------
    perievent_lfp1 : nept.AnalogSignal
    perievent_lfp2 : nept.AnalogSignal
    dt : float
    window : int
    fs : int
    extend : float
        Defaults to 0.3

    Returns
    -------
    timebins : np.array
    freq : np.array
    coherencegram : np.array

    """
    timebins = np.arange(perievent_lfp1.time[0], perievent_lfp1.time[-1]+dt, dt)
    coherencegram = np.zeros((window+1, len(timebins)))

    for i, (t_start, t_stop) in enumerate(zip(timebins[:-2], timebins[1:-1])):
        lfp1 = perievent_lfp1.time_slice(t_start-extend, t_stop+extend)
        lfp2 = perievent_lfp2.time_slice(t_start-extend, t_stop+extend)
        freq, coherencegram[:, i] = mean_coherence(lfp1, lfp2, window, fs)

    return timebins, freq, coherencegram
