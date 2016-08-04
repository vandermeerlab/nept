import os

from startup import load_csc, load_videotrack, load_events, load_spikes, convert_to_cm

session_id = 'R063d4'

thisdir = os.path.dirname(os.path.realpath(__file__))
dataloc = os.path.abspath(os.path.join(thisdir, '..', 'cache', 'data'))

species = 'rat'
behavior = 'shortcut'
target = 'dCA1'
experimenter = 'Emily Irvine'


def get_csc(lfp_mat):
    return load_csc(os.path.join(dataloc, lfp_mat))


def get_pos(pxl_to_cm):
    pos = load_videotrack(os.path.join(dataloc, 'R063-2015-03-23-vt.mat'))
    pos['x'] = pos['x'] / pxl_to_cm[0]
    pos['y'] = pos['y'] / pxl_to_cm[1]
    return pos


def get_events():
    return load_events(os.path.join(dataloc, 'R063-2015-03-23-event.mat'))


def get_spikes():
    return load_spikes(os.path.join(dataloc, 'R063-2015-03-23-spike.mat'))


# Experimental session-specific task times for R063 day 4
task_times = dict()
task_times['prerecord'] = [1074.6, 1378.7]
task_times['phase1'] = [1415.9, 1847.2]
task_times['pauseA'] = [1860.6, 2486.0]
task_times['phase2'] = [2504.6, 3704.5]
task_times['pauseB'] = [3725.3, 5600.7]
task_times['phase3'] = [5627.4, 8638.8]
task_times['postrecord'] = [8656.4, 9000.0]

pxl_to_cm = (7.9628, 7.2755)

fs = 2000

run_threshold = 0.5

good_lfp = ['R063-2015-03-23-CSC04a.ncs']
good_swr = ['R063-2015-03-23-CSC04.mat']
good_theta = ['R063-2015-03-23-CSC10.mat']

# Session-specific path trajectory points
path_pts = dict()
path_pts['feeder1'] = [552, 461]
path_pts['point1'] = [551, 409]
path_pts['turn1'] = [547, 388]
path_pts['point2'] = [535, 383]
path_pts['point3'] = [479, 381]
path_pts['point4'] = [370, 400]
path_pts['point5'] = [274, 384]
path_pts['turn2'] = [230, 370]
path_pts['point6'] = [219, 325]
path_pts['point7'] = [234, 98]
path_pts['turn3'] = [244, 66]
path_pts['point8'] = [275, 51]
path_pts['point9'] = [334, 46]
path_pts['feeder2'] = [662, 55]
path_pts['shortcut1'] = [546, 387]
path_pts['spt1'] = [620, 373]
path_pts['spt2'] = [652, 358]
path_pts['spt3'] = [662, 313]
path_pts['spt4'] = [661, 150]
path_pts['shortcut2'] = [662, 55]
path_pts['novel1'] = [331, 392]
path_pts['npt1'] = [324, 460]
path_pts['npt2'] = [316, 471]
path_pts['npt3'] = [289, 478]
path_pts['novel2'] = [113, 470]
path_pts['pedestal'] = [412, 203]

path_pts = convert_to_cm(path_pts, pxl_to_cm)

u_trajectory = [path_pts['feeder1'], path_pts['point1'], path_pts['turn1'], path_pts['point2'],
                path_pts['point3'], path_pts['point4'], path_pts['point5'], path_pts['turn2'],
                path_pts['point6'], path_pts['point7'], path_pts['turn3'], path_pts['point8'],
                path_pts['point9'], path_pts['feeder2']]

shortcut_trajectory = [path_pts['shortcut1'], path_pts['spt1'], path_pts['spt2'],
                       path_pts['spt3'], path_pts['spt4'], path_pts['shortcut2']]

novel_trajectory = [path_pts['novel1'], path_pts['npt1'], path_pts['npt2'],
                    path_pts['npt3'], path_pts['novel2']]


sequence = dict(u=dict(), shortcut=dict())
sequence['u']['swr_start'] = [8876.4, 8855.0, 8965.4]
sequence['u']['swr_stop'] = [8876.7, 8855.4, 8966.0]
sequence['u']['run_start'] = [2577, 2668.0, 3632.0]
sequence['u']['run_stop'] = [2607, 2698.0, 3667.0]
sequence['u']['ms'] = 10

sequence['shortcut']['swr_start'] = [8872.65, 8578.9, 8575.37, 8119.5, 8206.82]
sequence['shortcut']['swr_stop'] = [8873.15, 8579.5, 8575.55, 8119.8, 8207.38]
sequence['shortcut']['run_start'] = [5760.0, 6214.0, 5900.0, 6460.0, 6510.0]
sequence['shortcut']['run_stop'] = [5790.0, 6244.0, 5945.0, 6490.0, 6550.0]
sequence['shortcut']['ms'] = 10
