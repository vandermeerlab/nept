import numpy as np
import nept
import pandas as pd


class Session:
    def __init__(self, mags, pellets):
        self.mags = mags
        self.pellets = pellets
        self.trials = []

    def add_trial(self, epoch, cue, trial_type):
        """Adds trial to session
        epoch: nept.Epoch object
        cue: str
            Typically either 'light' or 'sound'
        trial_type: int
            Typically 1, 2, 3, or 4
        """
        cue_mag = epoch.intersect(self.mags)

        self.trials.append(
            Trial(cue=cue,
                  trial_type=trial_type,
                  durations=np.sum(cue_mag.durations),
                  numbers=cue_mag.n_epochs,
                  latency=cue_mag.start - epoch.start if cue_mag.n_epochs > 0 else 10.0,
                  responses=1 if cue_mag.n_epochs > 0 else 0))

    def add_missing_trial(self, cue, trial_type):
        """Adds trial placeholders for missing trials.
        cue: str
            Typically either 'light' or 'sound'
        trial_type: int
            Typically 1, 2, 3, or 4
        """
        self.trials.append(
            Trial(cue=cue,
                  trial_type=trial_type,
                  durations=np.nan,
                  numbers=np.nan,
                  latency=np.nan,
                  responses=np.nan))


class Trial:
    def __init__(self, cue, trial_type, durations, numbers, latency, responses):
        self.cue = cue
        self.trial_type = trial_type
        self.durations = durations
        self.numbers = numbers
        self.latency = latency
        self.responses = responses


class Rat:
    def __init__(self, rat_id, group1=None, group2=None):
        self.rat_id = rat_id
        self.group1 = group1
        self.group2 = group2
        self.sessions = []

        self.sound_trials = {1: 'sounds2',
                             2: 'sounds1',
                             3: 'sounds1',
                             4: 'sounds2'}

        if group1 is not None and rat_id in group1:
            self.light_trials = {1: 'lights1',
                                 2: 'lights1',
                                 3: 'lights2',
                                 4: 'lights2'}

        elif group2 is not None and rat_id in group2:
            self.light_trials = {1: 'lights2',
                                 2: 'lights2',
                                 3: 'lights1',
                                 4: 'lights1'}
        else:
            raise ValueError("rat id is incorrect. Should be in group1 or group2")

    def add_session(self, mags, pellets, lights1, lights2, sounds1, sounds2, trial1, trial2, trial3, trial4,
                    group=False):
        """Sorts cues into appropriate trials (1, 2, 3, 4), using intersect between trial and cue epochs."""
        session = Session(mags, pellets)

        if group == 1:
            for single_trial in trial1:
                session.add_trial(single_trial.intersect(lights1), 'light', 1)
                session.add_trial(single_trial.intersect(sounds2), 'sound', 1)
            for single_trial in trial2:
                session.add_trial(single_trial.intersect(lights1), 'light', 2)
                session.add_trial(single_trial.intersect(sounds1), 'sound', 2)
            for single_trial in trial3:
                session.add_trial(single_trial.intersect(lights2), 'light', 3)
                session.add_trial(single_trial.intersect(sounds1), 'sound', 3)
            for single_trial in trial4:
                session.add_trial(single_trial.intersect(lights2), 'light', 4)
                session.add_trial(single_trial.intersect(sounds2), 'sound', 4)

        elif group == 2:
            for single_trial in trial1:
                session.add_trial(single_trial.intersect(lights2), 'light', 1)
                session.add_trial(single_trial.intersect(sounds2), 'sound', 1)
            for single_trial in trial2:
                session.add_trial(single_trial.intersect(lights2), 'light', 2)
                session.add_trial(single_trial.intersect(sounds1), 'sound', 2)
            for single_trial in trial3:
                session.add_trial(single_trial.intersect(lights1), 'light', 3)
                session.add_trial(single_trial.intersect(sounds1), 'sound', 3)
            for single_trial in trial4:
                session.add_trial(single_trial.intersect(lights1), 'light', 4)
                session.add_trial(single_trial.intersect(sounds2), 'sound', 4)

        else:
            raise ValueError("must specify a group")

        self.sessions.append(session)

    def add_session_medpc(self, mags, pellets, lights1, lights2, sounds1, sounds2, n_unique=8, delay=5.02,
                          tolerance=1e-08):
        """Sorts cues into appropriate trials (1, 2, 3, 4), using specified delay between light and sound cues."""

        session = Session(mags, pellets)

        for trial in [1, 2, 3, 4]:
            if self.light_trials[trial] == 'lights1':
                light_cues = lights1
            elif self.light_trials[trial] == 'lights2':
                light_cues = lights2
            if self.sound_trials[trial] == 'sounds1':
                sound_cues = sounds1
            elif self.sound_trials[trial] == 'sounds2':
                sound_cues = sounds2

            n_trials = 0
            for light in light_cues:
                for sound in sound_cues:
                    if np.allclose(sound.start - light.stop, delay, atol=tolerance):
                        session.add_trial(light, 'light', trial)
                        session.add_trial(sound, 'sound', trial)
                        n_trials += 1

            for _ in range(n_unique - n_trials):
                session.add_missing_trial('light', trial)
                session.add_missing_trial('sound', trial)

        self.sessions.append(session)


def f_analyze(trial, measure):
    """Extracts appropriate analysis metric.

    Parameters
    ----------
    trial: emi_biconditional Trial object
    measure: str
        One of 'durations', 'numbers', 'latency', or 'responses'

    Returns
    --------
    output: analysis metric for a given trial

    """
    if measure not in ['durations', 'numbers', 'latency', 'responses']:
        raise ValueError("measure must be one of 'durations', 'numbers', 'latency', or 'responses'")
    if measure == 'durations':
        output = trial.durations
    if measure == 'numbers':
        output = trial.numbers
    if measure == 'latency':
        output = trial.latency
    if measure == 'responses':
        output = trial.responses * 100.

    return output


def combine_rats(data, rats, n_sessions, only_sound=False):
    """Combines behavioral measures from multiple rats, sessions and trials.

    data: dict
        With rat (str) as key, contains Rat objects for each rat
    rats: list
        With rat_id (str)
    n_sessions: int
    only_sound: boolean

    Returns
    -------
    df: pd.DataFrame

    """
    measures = ['durations', 'numbers', 'latency', 'responses']
    together = dict(trial=[], rat=[], session=[], trial_type=[], rewarded=[],
                    cue=[], value=[], measure=[], condition=[])

    for session in range(n_sessions):
        for rat in rats:
            for i, trial in enumerate(data[rat].sessions[session].trials):
                for measure in measures:
                    if not only_sound or trial.cue == 'sound':
                        together['trial'].append("%s, %d" % (rat, i))
                        together['rat'].append(rat)
                        together['session'].append(session+1)
                        together['trial_type'].append(trial.trial_type)
                        together['rewarded'].append("%s %s" %
                                                    (trial.cue, 'rewarded' if trial.trial_type % 2 == 0 else 'unrewarded'))
                        together['cue'].append(trial.cue)
                        together['condition'].append("%s %d" % (trial.cue, trial.trial_type))
                        together['measure'].append(measure)
                        together['value'].append(f_analyze(trial, measure))

    df = pd.DataFrame(data=together)

    fix_missing_trials(df)

    return df


def fix_missing_trials(df):
    """Replaces nan values with mean for that trial type

    Parameters
    ----------
    df: pd.DataFrame

    Note: this is a hack to handle sessions where there were fewer trials than expected.
    This function finds those trials and replaces the values with the mean for that
    trial type across the session.

    """
    nan_idx = np.where(np.isnan(df['value']))[0]
    for idx in nan_idx:
        row = df.loc[idx]
        value = df.loc[(df['rat'] == row['rat']) &
                       (df['session'] == row['session']) &
                       (df['condition'] == row['condition']) &
                       (df['measure'] == row['measure'])].mean()['value']

        df.set_value(idx, 'value', value)
