import os
import numpy as np
import nept


def assign_medpc_label(data):
    """Assigns events to proper labels.

    Parameters
    ----------
    data: dict

    Returns
    -------
    rats_data: dict
        With mags, pellets, lights1, lights2, sounds1, sounds2, trial1, trial2, trial3, trial4 as keys.
        Each contains nept.Epoch objects

    """
    mag_start = np.array(data[1])
    mag_end = np.array(data[2])
    if len(mag_start) > len(mag_end):
        mag_start = np.array(data[1][:-1])
    pel_start = np.array(data[3])
    pel_end = pel_start + 1
    light1_start = np.array(data[4])
    light1_end = np.array(data[5])
    light2_start = np.array(data[6])
    light2_end = np.array(data[7])
    sound1_start = np.array(data[8])
    sound1_end = np.array(data[9])
    sound2_start = np.array(data[10])
    sound2_end = np.array(data[11])

    rats_data = dict()
    rats_data['mags'] = nept.Epoch(mag_start, mag_end-mag_start)
    rats_data['pellets'] = nept.Epoch(pel_start, pel_end-pel_start)
    rats_data['lights1'] = nept.Epoch(light1_start, light1_end-light1_start)
    rats_data['lights2'] = nept.Epoch(light2_start, light2_end-light2_start)
    rats_data['sounds1'] = nept.Epoch(sound1_start, sound1_end-sound1_start)
    rats_data['sounds2'] = nept.Epoch(sound2_start, sound2_end-sound2_start)

    return rats_data


thisdir = os.path.dirname(os.path.realpath(__file__))
roborats = os.path.join(thisdir, '!roborats')

rats_data = nept.load_medpc(roborats, assign_medpc_label)

rats = ['1', '2', '3', '4', '5', '6', '7', '8']
group1 = ['1', '3', '5', '7']
group2 = ['2', '4', '6', '8']

data = dict()
for rat in rats:
    data[rat] = nept.Rat(rat, group1, group2)
    data[rat].add_session_medpc(**rats_data[rat])

n_sessions = len(data['1'].sessions)
only_sound = False

df = nept.combine_rats(data, rats, n_sessions, only_sound=False)


def test_no_mags():
    rat = '1'
    for cue in ['light', 'sound']:
        for trial in [1, 2, 3, 4]:
            this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                       groupby(['trial_type']).get_group(trial)[['measure', 'value']])
            assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 0.0))
            assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 0.0))
            assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 10.0))
            assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 0.0))


def test_all_mags():
    rat = '2'
    for cue in ['light', 'sound']:
        for trial in [1, 2, 3, 4]:
            this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                       groupby(['trial_type']).get_group(trial)[['measure', 'value']])
            assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 10.0))
            assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 1.0))
            assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 0.0))
            assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 100.0))


def test_sound_only():
    rat = '3'
    cue = 'sound'
    for trial in [1, 2, 3, 4]:
        this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                   groupby(['trial_type']).get_group(trial)[['measure', 'value']])
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 10.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 1.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 100.0))

    cue = 'light'
    for trial in [1, 2, 3, 4]:
        this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                   groupby(['trial_type']).get_group(trial)[['measure', 'value']])
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 10.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 0.0))


def test_light_only():
    rat = '4'
    cue = 'sound'
    for trial in [1, 2, 3, 4]:
        this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                   groupby(['trial_type']).get_group(trial)[['measure', 'value']])
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 10.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 0.0))

    cue = 'light'
    for trial in [1, 2, 3, 4]:
        this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                   groupby(['trial_type']).get_group(trial)[['measure', 'value']])
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 10.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 1.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 100.0))


def test_rewarded_sound():
    rat = '5'
    cue = 'light'
    for trial in [1, 2, 3, 4]:
        this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                   groupby(['trial_type']).get_group(trial)[['measure', 'value']])
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 10.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 0.0))

    cue = 'sound'
    for trial in [2, 4]:
        this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                   groupby(['trial_type']).get_group(trial)[['measure', 'value']])
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 10.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 1.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 100.0))

    for trial in [1, 3]:
        this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                   groupby(['trial_type']).get_group(trial)[['measure', 'value']])
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 10.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 0.0))


def test_iti_only():
    rat = '6'
    for cue in ['light', 'sound']:
        for trial in [1, 2, 3, 4]:
            this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                       groupby(['trial_type']).get_group(trial)[['measure', 'value']])
            assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 0.0))
            assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 0.0))
            assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 10.0))
            assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 0.0))


def test_half_light():
    rat = '7'
    cue = 'sound'
    for trial in [1, 2, 3, 4]:
        this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                   groupby(['trial_type']).get_group(trial)[['measure', 'value']])
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 10.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 0.0))

    cue = 'light'
    for trial in [1, 3]:
        this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                   groupby(['trial_type']).get_group(trial)[['measure', 'value']])
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 5.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 1.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 0.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 100.0))

    for trial in [2, 4]:
        this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
                   groupby(['trial_type']).get_group(trial)[['measure', 'value']])
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 5.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 1.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 5.0))
        assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 100.0))


def test_complex():
    rat = '8'

    trial = 1  # are the trial numbers correct?
    cue = 'light'
    this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
               groupby(['trial_type']).get_group(trial)[['measure', 'value']])

    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 0.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 0.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 10.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 0.0))

    cue = 'sound'
    this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
               groupby(['trial_type']).get_group(trial)[['measure', 'value']])

    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 0.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 0.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 10.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 0.0))

    trial = 2
    cue = 'light'
    this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
               groupby(['trial_type']).get_group(trial)[['measure', 'value']])

    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 0.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 0.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 10.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 0.0))

    cue = 'sound'
    this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
               groupby(['trial_type']).get_group(trial)[['measure', 'value']])

    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 10.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 1.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 0.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 100.0))

    trial = 3
    cue = 'light'
    this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
               groupby(['trial_type']).get_group(trial)[['measure', 'value']])

    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 1.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 1.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 9.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 100.0))

    cue = 'sound'
    this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
               groupby(['trial_type']).get_group(trial)[['measure', 'value']])

    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 1.98))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 1.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 0.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 100.0))

    trial = 4  # or should it be trial 1?
    cue = 'light'
    this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
               groupby(['trial_type']).get_group(trial)[['measure', 'value']])

    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 2.5))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 2.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 0.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 100.0))

    cue = 'sound'
    this_df = (df.groupby(['rat']).get_group(rat).groupby(['cue']).get_group(cue).
               groupby(['trial_type']).get_group(trial)[['measure', 'value']])

    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'durations']['value']), 10.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'numbers']['value']), 1.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'latency']['value']), 0.0))
    assert (np.allclose(np.mean(this_df[this_df['measure'] == 'responses']['value']), 100.0))
