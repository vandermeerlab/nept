import os
import numpy as np
import nept

thisdir = os.path.dirname(os.path.realpath(__file__))
roborats = os.path.join(thisdir, '!roborats')


def assign_medpc_label(data):
    """Assigns events to labels.

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
    rats_data['mags'] = nept.Epoch(mag_start, mag_end)
    rats_data['pellets'] = nept.Epoch(pel_start, pel_end)
    rats_data['lights1'] = nept.Epoch(light1_start, light1_end)
    rats_data['lights2'] = nept.Epoch(light2_start, light2_end)
    rats_data['sounds1'] = nept.Epoch(sound1_start, sound1_end)
    rats_data['sounds2'] = nept.Epoch(sound2_start, sound2_end)

    return rats_data


def test_medpc_roborats():
    rats_data = nept.load_medpc(roborats, assign_medpc_label)
    for subject in ['1', '2', '3', '4', '5', '6', '7', '8']:
        assert np.allclose(rats_data[subject]['lights1'].starts, np.array([10.0, 100.0]))
        assert np.allclose(rats_data[subject]['lights1'].stops, np.array([20.0, 110.0]))
        assert np.allclose(rats_data[subject]['lights2'].starts, np.array([200.0, 300.0]))
        assert np.allclose(rats_data[subject]['lights2'].stops, np.array([210.0, 310.0]))
        assert np.allclose(rats_data[subject]['sounds1'].starts, np.array([115.02, 215.02]))
        assert np.allclose(rats_data[subject]['sounds1'].stops, np.array([125.02, 225.02]))
        assert np.allclose(rats_data[subject]['sounds2'].starts, np.array([25.02, 315.02]))
        assert np.allclose(rats_data[subject]['sounds2'].stops, np.array([35.02, 325.02]))

    assert np.allclose(rats_data['1']['mags'].durations, np.array([]))
    assert np.allclose(rats_data['2']['mags'].durations, np.array([321.]))
    assert np.allclose(rats_data['3']['mags'].durations, np.array([10., 10., 10., 10.]))
    assert np.allclose(rats_data['4']['mags'].durations, np.array([10., 10., 10., 10.]))
    assert np.allclose(rats_data['5']['mags'].durations, np.array([10., 10.]))
    assert np.allclose(rats_data['6']['mags'].durations,
                       np.array([1., 5.01, 64.97, 5.01, 74.97, 5.01, 74.97, 5.01, 4.97]))
    assert np.allclose(rats_data['7']['mags'].durations, np.array([5., 5., 5., 5.]))
    assert np.allclose(rats_data['8']['mags'].durations, np.array([2., 1.5, 10., 8., 12.]))
