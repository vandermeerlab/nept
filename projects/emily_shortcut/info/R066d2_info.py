import os

from startup import load_csc, load_videotrack, load_events, load_spikes, convert_to_cm

session_id = 'R066d2'

thisdir = os.path.dirname(os.path.realpath(__file__))
dataloc = os.path.abspath(os.path.join(thisdir, '..', 'cache', 'data'))

species = 'rat'
behavior = 'shortcut'
target = 'dCA1'
experimenter = 'Emily Irvine'


def get_csc(lfp_mat):
    return load_csc(os.path.join(dataloc, lfp_mat))


def get_pos(pxl_to_cm):
    pos = load_videotrack(os.path.join(dataloc, 'R066-2014-11-28-vt.mat'))
    pos['x'] = pos['x'] / pxl_to_cm[0]
    pos['y'] = pos['y'] / pxl_to_cm[1]
    return pos


def get_events():
    return load_events(os.path.join(dataloc, 'R066-2014-11-28-event.mat'))


def get_spikes():
    return load_spikes(os.path.join(dataloc, 'R066-2014-11-28-spike.mat'))


# plt.plot(pos['x'], pos['y'], 'b.', ms=1)
# plt.show()

# Experimental session-specific task times for R066 day 2
task_times = dict()
task_times['prerecord'] = [11850.0, 12155.0]
task_times['phase1'] = [12210.0, 12840.0]
task_times['pauseA'] = [12900.0, 13501.0]
task_times['phase2'] = [13574.0, 14776.0]
task_times['pauseB'] = [14825.0, 16633.0]
task_times['phase3'] = [16684.0, 19398.0]
task_times['postrecord'] = [19436.0, 19742.0]

pxl_to_cm = (7.5460, 7.2192)

fs = 2000

run_threshold = 0.35

good_lfp = ['R066-2014-11-28-CSC02c.ncs']
good_swr = ['R066-2014-11-28-CSC02.mat']
good_theta = ['R066-2014-11-28-CSC07.mat']

# Session-specific path trajectory points
path_pts = dict()
path_pts['feeder1'] = [530, 460]
path_pts['turn1'] = [525, 382]
path_pts['pt1'] = [472, 375]
# path_pts['pt2'] = [425, 397]
# path_pts['pt3'] = [404, 359]
path_pts['pt4'] = [439, 379]
path_pts['pt5'] = [410, 382]
# path_pts['pt6'] = [307, 357]
path_pts['pt7'] = [366, 387]
path_pts['pt8'] = [316, 384]
path_pts['pt9'] = [249, 368]
path_pts['turn2'] = [205, 343]
path_pts['pt10'] = [194, 299]
path_pts['pt11'] = [199, 158]
path_pts['pt12'] = [207, 92]
path_pts['turn3'] = [220, 66]
path_pts['pt13'] = [253, 48]
path_pts['pt14'] = [412, 43]
path_pts['feeder2'] = [623, 54]
# path_pts['pt15'] = [665, 51]
path_pts['shortcut1'] = [525, 382]
path_pts['spt1'] = [528, 220]
path_pts['spt2'] = [540, 181]
path_pts['spt3'] = [568, 168]
path_pts['spt4'] = [614, 153]
path_pts['spt5'] = [630, 119]
path_pts['shortcut2'] = [631, 53]
path_pts['novel1'] = [204, 365]
path_pts['npt1'] = [89, 359]
path_pts['novel2'] = [98, 149]
path_pts['pedestal'] = [331, 206]

path_pts = convert_to_cm(path_pts, pxl_to_cm)

u_trajectory = [path_pts['feeder1'], path_pts['turn1'],
                path_pts['pt1'], path_pts['pt4'],
                path_pts['pt5'], path_pts['pt7'],
                path_pts['pt8'], path_pts['pt9'], path_pts['turn2'], path_pts['pt10'],
                path_pts['pt11'], path_pts['pt12'], path_pts['turn3'],
                path_pts['pt13'], path_pts['pt14'], path_pts['feeder2']]

shortcut_trajectory = [path_pts['shortcut1'], path_pts['spt1'], path_pts['spt2'],
                       path_pts['spt3'], path_pts['spt4'], path_pts['spt5'], path_pts['shortcut2']]

novel_trajectory = [path_pts['novel1'], path_pts['npt1'], path_pts['novel2']]

sequence = dict(u=dict(), shortcut=dict())
sequence['u']['swr_start'] = [19482.6, 19613.0, 19719.95]
sequence['u']['swr_stop'] = [19483.0,  19613.4, 19720.5]
sequence['u']['run_start'] = [19220.0, 14370.0, 14130.0]
sequence['u']['run_stop'] = [19260.0, 14440.0, 14160.0]
sequence['u']['ms'] = 10

sequence['shortcut']['swr_start'] = [19710.0, 16584.8]
sequence['shortcut']['swr_stop'] = [19710.6, 16585.2]
sequence['shortcut']['run_start'] = [17960.0, 18800.0]
sequence['shortcut']['run_stop'] = [17990.0, 18830.0]
sequence['shortcut']['ms'] = 10
