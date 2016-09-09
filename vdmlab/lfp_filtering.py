import numpy as np
import scipy.signal
import scipy.stats as stats

import vdmlab as vdm


def butter_bandpass(signal, thresh, fs, order=4):
    """ Filters signal using butterworth filter

    Parameters
    ----------
    signal : vdmlad.LFP
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


def detect_swr_hilbert(lfp, fs, thresh, z_thresh=3, power_thresh=3, merge_thresh=0.02, min_length=0.01):
    """Finds sharp-wave ripple (SWR) times and indices.

    Parameters
    ----------
    lfp : vdmlab.LocalFieldPotential
    fs : int
        Experiment-specific, something in the range of 2000 typical.
    thresh : tuple
        With format (lowcut, highcut).
        Typically (140.0, 250.0) for sharp-wave ripple detection.
    z_thres : int or float
        The default is set to 3
    power_thres : int or float
        The default is set to 3
    merge_thres : int or float
        The default is set to 0.02
    min_length : float
        Any sequence less than this amount is not considered a sharp-wave ripple.
        The default is set to 0.01.

    Returns
    -------
    swrs : list
        Containing vdmlab.LocalFieldPotential for each SWR event

    """
    # Filtering signal with butterworth fitler
    filtered_butter = butter_bandpass(lfp.data, thresh, fs)

    # Get LFP power (using Hilbert) and z-score the power
    # Zero padding to nearest regular number to speed up fast fourier transforms (FFT) computed in the hilbert function.
    # Regular numbers are composites of the prime factors 2, 3, and 5.
    hilbert_n = next_regular(lfp.n_samples)
    power_lfp = np.abs(scipy.signal.hilbert(filtered_butter, N=hilbert_n))
    power_lfp = power_lfp[:lfp.n_samples]  # removing the zero padding now that the power is computed
    zpower_lfp = stats.zscore(power_lfp)

    # Finding locations where the power changes
    detect = zpower_lfp > z_thresh
    detect = np.hstack([0, detect, 0])  # pad to detect first or last element change
    signal_change = np.diff(detect.astype(int))

    start_swr_idx = np.where(signal_change == 1)[0]
    stop_swr_idx = np.where(signal_change == -1)[0] - 1

    # Getting times associated with these power changes
    start_time = lfp.time[start_swr_idx]
    stop_time = lfp.time[stop_swr_idx]

    # Merging ranges that are closer - in time - than the merge_threshold.
    no_double = start_time[1:] - stop_time[:-1]
    merge_idx = np.where(no_double < merge_thresh)[0]
    start_merged = np.delete(start_time, merge_idx + 1)
    stop_merged = np.delete(stop_time, merge_idx)
    start_merged_idx = np.delete(start_swr_idx, merge_idx + 1)
    stop_merged_idx = np.delete(stop_swr_idx, merge_idx)

    # Removing ranges that are shorter - in time - than the min_length value.
    swr_len = stop_merged - start_merged
    short_idx = np.where(swr_len < min_length)[0]
    start_merged = np.delete(start_merged, short_idx)
    stop_merged = np.delete(stop_merged, short_idx)
    start_merged_idx = np.delete(start_merged_idx, short_idx)
    stop_merged_idx = np.delete(stop_merged_idx, short_idx)

    # Removing ranges that have powers less than the power_threshold if sufficiently different.
    if power_thresh > z_thresh:
        max_z = []
        for start_idx, stop_idx in zip(start_merged_idx, stop_merged_idx):
            max_z.append(np.max(zpower_lfp[start_idx:stop_idx]))
        max_z = np.array(max_z)

        z_idx = np.where(max_z < power_thresh)[0]
        start_merged = np.delete(start_merged, z_idx)
        stop_merged = np.delete(stop_merged, z_idx)
        start_merged_idx = np.delete(start_merged_idx, z_idx)
        stop_merged_idx = np.delete(stop_merged_idx, z_idx)

    swrs = vdm.Epoch(np.array([start_merged, stop_merged]))

    return swrs


def next_regular(target):
    """
    Find the next regular number greater than or equal to target.
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
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)

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
