# -*- coding: utf-8 -*-
# Adapted from nlxio written by Bernard Willards <https://github.com/bwillers/nlxio>

import numpy as np
import nept


def load_events(filename, labels):
    """Loads neuralynx events

    Parameters
    ----------
    filename: str
    labels: dict
        With event name as the key and Neuralynx event string as the value.

    Returns
    -------
    timestamps: dict

    """
    nev_data = load_nev(filename)

    idx = {key: [] for key in labels}
    for key in labels:
        for i, event in enumerate(nev_data['event_str']):
            if event.decode() == labels[key]:
                idx[key].append(i)

    timestamps = {label: [] for label in labels}

    times = nev_data['time'].astype(float) * 1e-6

    for label in labels:
        timestamps[label] = times[idx[label]]

    return timestamps


def load_lfp(filename):
    """Loads LFP as nept.LocalFieldPotential

    Parameters
    ----------
    filename: str

    Returns
    -------
    lfp: nept.LocalFieldPotential

    """
    data, time = load_ncs(filename)

    return nept.LocalFieldPotential(data, time)


def load_position(filename, pxl_to_cm):
    """Loads videotracking position as nept.Position

    Parameters
    ----------
    filename: str
    pxl_to_cm: tuple
        With (x, y) conversion factors

    Returns
    -------
    position: nept.Position

    """
    nvt_data = load_nvt(filename)

    xy = np.hstack(np.array([nvt_data['x'] / pxl_to_cm[0], nvt_data['y'] / pxl_to_cm[1]])[..., np.newaxis])

    return nept.Position(xy, nvt_data['time'])


def load_neuralynx_header(filename):
    """Loads a neuralynx header.

    Parameters
    ----------
    filename: str

    Returns
    -------
    header: byte str

    """
    f = open(filename, 'rb')

    # Neuralynx files have a 16kbyte header
    header = f.read(16 * 2**10)

    f.close()

    return header


def load_ncs(filename):
    """Loads a neuralynx .ncs electrode file.

    Parameters
    ----------
    filename: str

    Returns
    -------
    cscs: np.array
        Voltage trace (V)
    times: np.array
        Timestamps (microseconds)

    """

    f = open(filename, 'rb')

    # Neuralynx files have a 16kbyte header
    header = f.read(16 * 2**10)

    # The format for a .ncs files according the the neuralynx docs is
    # uint64 - timestamp in microseconds
    # uint32 - channel number
    # uint32 - sample freq
    # uint32 - number of valid samples
    # int16 x 512 - actual csc samples
    dt = np.dtype([('time', '<Q'), ('channel', '<i'), ('freq', '<i'),
                   ('valid', '<i'), ('csc', '<h', (512,))])
    data = np.fromfile(f, dt)

    # unpack the csc matrix
    csc = data['csc'].reshape((data['csc'].size,))

    data_times = data['time'] * 1e-6

    # find the frequency
    frequency = np.unique(data['freq'])
    if len(frequency) > 1:
        raise IOError("only one frequency allowed")
    frequency = frequency[0]

    # .ncs files have a timestamp for every ~512 data points.
    # Here, we assign timestamps for each data sample based on the sampling frequency
    # for each of the 512 data points. Sometimes a block will have fewer than 512 data entries,
    # number is set in data['valid'].
    this_idx = 0
    n_block = 512.
    offsets = np.arange(0, n_block / frequency, 1. / frequency)
    times = np.zeros(csc.shape)
    for i, (time, n_valid) in enumerate(zip(data_times, data['valid'])):
        times[this_idx:this_idx + n_valid] = time + offsets[:n_valid]
        this_idx += n_valid

    # now find analog_to_digital conversion factor in the header
        analog_to_digital = None
    for line in header.split(b'\n'):
        if line.strip().startswith(b'-ADBitVolts'):
            analog_to_digital = np.array(float(line.split(b' ')[1].decode()))

    if analog_to_digital is None:
        raise IOError("ADBitVolts not found in .ncs header for " + filename)

    cscs = csc * analog_to_digital

    f.close()

    return cscs, times


def load_nev(filename):
    """Loads a neuralynx .nev file.

    Parameters
    ----------
    filename: str

    Returns
    -------
    nev_data: dict
        With time (uint64), id (uint16), nttl (uint16), and event_str (charx128) as the most usable keys.

    """

    f = open(filename, 'rb')

    # There's nothing useful in the header for .nev files, so skip past it
    f.seek(2 ** 14)

    # An event record is as follows:
    # int16 - nstx - reserved
    # int16 - npkt_id - id of the originating system
    # int16 - npkt_data_size - this value should always be 2
    # uint64 - timestamp, microseconds
    # int16 - nevent_id - ID value for event
    # int16 - nttl - decimal TTL value read from the TTL input port
    # int16 - ncrc - record crc check, not used in consumer applications
    # int16 - ndummy1 - reserved
    # int16 - ndummy2 - reserved
    # int32x8 - dnExtra - extra bit values for this event
    # string(128) - event string
    dt = np.dtype([('filler1', '<h', 3), ('time', '<Q'), ('id', '<h'),
                   ('nttl', '<h'), ('filler2', '<h', 3), ('extra', '<i', 8),
                   ('event_str', np.dtype('a128'))])
    nev_data = np.fromfile(f, dt)

    f.close()

    return nev_data


def load_ntt(filename):
    """Loads a neuralynx .ntt tetrode spike file.

    Parameters
    ----------
    filename: str

    Returns
    -------
    timestamps: np.array
        Spikes as (num_spikes, length_waveform, num_channels)
    spikes: np.array
        Spike times as uint64 (us)
    frequency: float
        Sampling frequency in waveforms (Hz)

    Usage:
    timestamps, spikes, frequency = load_ntt('TT13.ntt')

    """

    f = open(filename, 'rb')

    # A tetrode spike record is as folows:
    # uint64 - timestamp                    bytes 0:8
    # uint32 - acquisition entity number    bytes 8:12
    # uint32 - classified cel number        bytes 12:16
    # 8 * uint32- params                    bytes 16:48
    # 32 * 4 * int16 - waveform points
    # hence total record size is 2432 bits, 304 bytes

    # header is 16kbyte, i.e. 16 * 2^10 = 2^14
    header = f.read(16 * 2**10)

    # Read the header and find the conversion factors / sampling frequency
    analog_to_digital = None
    frequency = None

    for line in header.split(b'\n'):
        if line.strip().startswith(b'-ADBitVolts'):
            analog_to_digital = np.array(float(line.split(b' ')[1].decode()))
        if line.strip().startswith(b'-SamplingFrequency'):
            frequency = float(line.split(b' ')[1].decode())

    f.seek(2 ** 14)  # start of the spike, records
    # Neuralynx write little endian for some dumb reason
    dt = np.dtype([('time', '<Q'), ('filer', '<i', 10),
                   ('spikes', np.dtype('<h'), (32, 4))])
    data = np.fromfile(f, dt)

    if analog_to_digital is None:
        raise IOError("ADBitVolts not found in .ntt header for " + filename)
    if frequency is None:
        raise IOError("Frequency not found in .ntt header for " + filename)

    f.close()

    return data['time'], data['spikes'] * analog_to_digital, frequency


def load_nvt(filename):
    """Loads a neuralynx .nvt file.

    Parameters
    ----------
    filename: str

    Returns
    -------
    nvt_data: dict
        With time, x, and y as keys.

    """
    f = open(filename, 'rb')

    # Neuralynx files have a 16kbyte header
    header = f.read(16 * 2**10)

    # The format for .nvt files according the the neuralynx docs is
    # uint16 - beginning of the record
    # uint16 - ID for the system
    # uint16 - size of videorec in bytes
    # uint64 - timestamp in microseconds
    # uint32 x 400 - points with the color bitfield values
    # int16 - unused
    # int32 - extracted X location of target
    # int32 - extracted Y location of target
    # int32 - calculated head angle in degrees clockwise from the positive Y axis
    # int32 x 50 - colored targets using the same bitfield format used to extract colors earlier
    dt = np.dtype([('filler1', '<h', 3), ('time', '<Q'), ('points', '<i', 400),
                   ('filler2', '<h'), ('x', '<i'), ('y', '<i'), ('head_angle', '<i'),
                   ('targets', '<i', 50)])
    data = np.fromfile(f, dt)

    nvt_data = dict()
    nvt_data['time'] = data['time'] * 1e-6
    nvt_data['x'] = np.array(data['x'], dtype=float)
    nvt_data['y'] = np.array(data['y'], dtype=float)
    nvt_data['targets'] = np.array(data['targets'], dtype=float)

    empty_idx = (data['x'] == 0) & (data['y'] == 0)
    for key in nvt_data:
        nvt_data[key] = nvt_data[key][~empty_idx]

    return nvt_data


def write_ntt_header(nlx_headersize=16*2**10, name=None, t_open=None, t_close=None,
                     filetype=None, fileversion=None, recordsize=None,
                     cheetahrev=None, hardwaresubname=None, hardwaresubtype=None,
                     samplingfreq=None, admaxvalue=None, adbitvolts=None,
                     acqentname=None, numadchannels=None, adchannel=None,
                     inputrange=None, inputinverted=None, dsplowcutfilterenabled=None,
                     dsplowcutfreq=None, dsplowcutnumtaps=None, dsplowcutfiltertype=None,
                     dsphighcutfilterenabled=None, dsphighcutfreq=None, dsphighcutnumtaps=None,
                     dsphighcutfiltertype=None, dspdelaycomp=None,
                     dspfilterdelay=None, waveformlen=None, alignmentpt=None,
                     threshval=None, minretriggertime=None, spikeretriggertime=None,
                     dualthresh=None, featurepeak1=None, featurepeak2=None,
                     featurepeak3=None, featurepeak4=None, featurevalley1=None,
                     featurevalley2=None, featurevalley3=None, featurevalley4=None):
    """
    Returns a .ntt header

    Parameters
    ----------
    nlx_headersize: float
        Default is 16*2**10
    name: str
        Default is None
    ...


    Returns
    -------
    header: byte string

    """

    header = '######## Neuralynx Data File Header'
    if name is not None:
        header += '## File Name ' + name
    else:
        header += '## File Name '
    if t_open is not None:
        header += '## Time Opened (m/d/y): ' + t_open
    else:
        header += '## Time Opened (m/d/y): '
    if t_close is not None:
        header += '## Time Closed (m/d/y): ' + t_close
    else:
        header += '## Time Closed (m/d/y): '
    header += '\r\n'
    if filetype is not None:
        header += '-FileType ' + filetype
    else:
        header += '-FileType '
    if fileversion is not None:
        header += '-FileVersion ' + fileversion
    else:
        header += '-FileVersion '
    if recordsize is not None:
        header += '-RecordSize ' + recordsize
    else:
        header += '-RecordSize '
    header += '\r\n'
    if cheetahrev is not None:
        header += '-CheetahRev ' + cheetahrev
    else:
        header += '-CheetahRev '
    header += '\r\n'
    if hardwaresubname is not None:
        header += '-HardwareSubSystemName ' + hardwaresubname
    else:
        header += '-HardwareSubSystemName '
    if hardwaresubtype is not None:
        header += '-HardwareSubSystemType ' + hardwaresubtype
    else:
        header += '-HardwareSubSystemType '
    if samplingfreq is not None:
        header += '-SamplingFrequency ' + samplingfreq
    else:
        header += '-SamplingFrequency '
    if admaxvalue is not None:
        header += '-ADMaxValue ' + admaxvalue
    else:
        header += '-ADMaxValue '
    if adbitvolts is not None:
        header += '-ADBitVolts ' + adbitvolts
    else:
        header += '-ADBitVolts '
    header += '\r\n'
    if acqentname is not None:
        header += '-AcqEntName ' + acqentname
    else:
        header += '-AcqEntName '
    if numadchannels is not None:
        header += '-NumADChannels ' + numadchannels
    else:
        header += '-NumADChannels '
    if adchannel is not None:
        header += 'ADChannel ' + adchannel
    else:
        header += 'ADChannel '
    if inputrange is not None:
        header += '-InputRange ' + inputrange
    else:
        header += '-InputRange '
    if inputinverted is not None:
        header += '-InputInverted ' + inputinverted
    else:
        header += '-InputInverted '
    header += '\r\n'
    if dsplowcutfilterenabled is not None:
        header += '-DSPLowCutFilterEnabled ' + dsplowcutfilterenabled
    else:
        header += '-DSPLowCutFilterEnabled '
    if dsplowcutfreq is not None:
        header += '-DspLowCutFrequency ' + dsplowcutfreq
    else:
        header += '-DspLowCutFrequency '
    if dsplowcutnumtaps is not None:
        header += '-DspLowCutNumTaps ' + dsplowcutnumtaps
    else:
        header += '-DspLowCutNumTaps '
    if dsplowcutfiltertype is not None:
        header += '-DspLowCutFilterType ' + dsplowcutfiltertype
    else:
        header += '-DspLowCutFilterType '
    if dsphighcutfilterenabled is not None:
        header += '-DSPHighCutFilterEnabled ' + dsphighcutfilterenabled
    else:
        header += '-DSPHighCutFilterEnabled '
    if dsphighcutfreq is not None:
        header += '-DspHighCutFrequency ' + dsphighcutfreq
    else:
        header += '-DspHighCutFrequency '
    if dsphighcutnumtaps is not None:
        header += '-DspHighCutNumTaps ' + dsphighcutnumtaps
    else:
        header += '-DspHighCutNumTaps '
    if dsphighcutfiltertype is not None:
        header += '-DspHighCutFilterType ' + dsphighcutfiltertype
    else:
        header += '-DspHighCutFilterType '
    if dspdelaycomp is not None:
        header += '-DspDelayCompensation ' + dspdelaycomp
    else:
        header += '-DspDelayCompensation '
    if dspfilterdelay is not None:
        header += '-DspFilterDelay_µs ' + dspfilterdelay
    else:
        header += '-DspFilterDelay_µs '
    header += '\r\n'
    if waveformlen is not None:
        header += '-WaveformLength ' + waveformlen
    else:
        header += '-WaveformLength '
    if alignmentpt is not None:
        header += '-AlignmentPt ' + alignmentpt
    else:
        header += '-AlignmentPt '
    if threshval is not None:
        header += '-ThreshVal ' + threshval
    else:
        header += '-ThreshVal '
    if minretriggertime is not None:
        header += '-MinRetriggerSamples ' + minretriggertime
    else:
        header += '-MinRetriggerSamples '
    if spikeretriggertime is not None:
        header += '-SpikeRetriggerTime ' + spikeretriggertime
    else:
        header += '-SpikeRetriggerTime '
    if dualthresh is not None:
        header += '-DualThresholding ' + dualthresh
    else:
        header += '-DualThresholding '
    header += '\r\n'
    if featurepeak1 is not None:
        header += '-Feature Peak ' + featurepeak1
    else:
        header += '-Feature Peak '
    if featurepeak2 is not None:
        header += '-Feature Peak ' + featurepeak2
    else:
        header += '-Feature Peak '
    if featurepeak3 is not None:
        header += '-Feature Peak ' + featurepeak3
    else:
        header += '-Feature Peak '
    if featurepeak4 is not None:
        header += '-Feature Peak ' + featurepeak4
    else:
        header += '-Feature Peak '
    if featurevalley1 is not None:
        header += '-Feature Valley ' + featurevalley1
    else:
        header += '-Feature Valley '
    if featurevalley2 is not None:
        header += '-Feature Valley ' + featurevalley2
    else:
        header += '-Feature Valley '
    if featurevalley3 is not None:
        header += '-Feature Valley ' + featurevalley3
    else:
        header += '-Feature Valley '
    if featurevalley4 is not None:
        header += '-Feature Valley ' + featurevalley4
    else:
        header += '-Feature Valley '
    header += '\r\n'

    offset = nlx_headersize - len(header)
    header = header.ljust(offset, '\x00')

    return header.encode()


def write_ncs_header(nlx_headersize=16*2**10, name=None, t_open=None, t_close=None,
                     filetype=None, fileversion=None, recordsize=None,
                     cheetahrev=None, hardwaresubname=None, hardwaresubtype=None,
                     samplingfreq=None, admaxvalue=None, adbitvolts=None,
                     acqentname=None, numadchannels=None, adchannel=None,
                     inputrange=None, inputinverted=None, dsplowcutfilterenabled=None,
                     dsplowcutfreq=None, dsplowcutnumtaps=None, dsplowcutfiltertype=None,
                     dsphighcutfilterenabled=None, dsphighcutfreq=None, dsphighcutnumtaps=None,
                     dsphighcutfiltertype=None, dspdelaycomp=None, dspfilterdelay=None):
    """
    Returns a .ncs header

    Parameters
    ----------
    nlx_headersize: float
        Default is 16*2**10
    name: str
        Default is None
    ...


    Returns
    -------
    header: byte string

    """

    header = '######## Neuralynx Data File Header'
    if name is not None:
        header += '## File Name ' + name
    else:
        header += '## File Name '
    if t_open is not None:
        header += '## Time Opened (m/d/y): ' + t_open
    else:
        header += '## Time Opened (m/d/y): '
    if t_close is not None:
        header += '## Time Closed (m/d/y): ' + t_close
    else:
        header += '## Time Closed (m/d/y): '
    header += '\r\n'
    if filetype is not None:
        header += '-FileType ' + filetype
    else:
        header += '-FileType '
    if fileversion is not None:
        header += '-FileVersion ' + fileversion
    else:
        header += '-FileVersion '
    if recordsize is not None:
        header += '-RecordSize ' + recordsize
    else:
        header += '-RecordSize '
    header += '\r\n'
    if cheetahrev is not None:
        header += '-CheetahRev ' + cheetahrev
    else:
        header += '-CheetahRev '
    header += '\r\n'
    if hardwaresubname is not None:
        header += '-HardwareSubSystemName ' + hardwaresubname
    else:
        header += '-HardwareSubSystemName '
    if hardwaresubtype is not None:
        header += '-HardwareSubSystemType ' + hardwaresubtype
    else:
        header += '-HardwareSubSystemType '
    if samplingfreq is not None:
        header += '-SamplingFrequency ' + samplingfreq
    else:
        header += '-SamplingFrequency '
    if admaxvalue is not None:
        header += '-ADMaxValue ' + admaxvalue
    else:
        header += '-ADMaxValue '
    if adbitvolts is not None:
        header += '-ADBitVolts ' + adbitvolts
    else:
        header += '-ADBitVolts '
    header += '\r\n'
    if acqentname is not None:
        header += '-AcqEntName ' + acqentname
    else:
        header += '-AcqEntName '
    if numadchannels is not None:
        header += '-NumADChannels ' + numadchannels
    else:
        header += '-NumADChannels '
    if adchannel is not None:
        header += 'ADChannel ' + adchannel
    else:
        header += 'ADChannel '
    if inputrange is not None:
        header += '-InputRange ' + inputrange
    else:
        header += '-InputRange '
    if inputinverted is not None:
        header += '-InputInverted ' + inputinverted
    else:
        header += '-InputInverted '
    header += '\r\n'
    if dsplowcutfilterenabled is not None:
        header += '-DSPLowCutFilterEnabled ' + dsplowcutfilterenabled
    else:
        header += '-DSPLowCutFilterEnabled '
    if dsplowcutfreq is not None:
        header += '-DspLowCutFrequency ' + dsplowcutfreq
    else:
        header += '-DspLowCutFrequency '
    if dsplowcutnumtaps is not None:
        header += '-DspLowCutNumTaps ' + dsplowcutnumtaps
    else:
        header += '-DspLowCutNumTaps '
    if dsplowcutfiltertype is not None:
        header += '-DspLowCutFilterType ' + dsplowcutfiltertype
    else:
        header += '-DspLowCutFilterType '
    if dsphighcutfilterenabled is not None:
        header += '-DSPHighCutFilterEnabled ' + dsphighcutfilterenabled
    else:
        header += '-DSPHighCutFilterEnabled '
    if dsphighcutfreq is not None:
        header += '-DspHighCutFrequency ' + dsphighcutfreq
    else:
        header += '-DspHighCutFrequency '
    if dsphighcutnumtaps is not None:
        header += '-DspHighCutNumTaps ' + dsphighcutnumtaps
    else:
        header += '-DspHighCutNumTaps '
    if dsphighcutfiltertype is not None:
        header += '-DspHighCutFilterType ' + dsphighcutfiltertype
    else:
        header += '-DspHighCutFilterType '
    if dspdelaycomp is not None:
        header += '-DspDelayCompensation ' + dspdelaycomp
    else:
        header += '-DspDelayCompensation '
    if dspfilterdelay is not None:
        header += '-DspFilterDelay_µs ' + dspfilterdelay
    else:
        header += '-DspFilterDelay_µs '
    header += '\r\n'

    offset = nlx_headersize - len(header)
    header = header.ljust(offset, '\x00')

    return header.encode()
