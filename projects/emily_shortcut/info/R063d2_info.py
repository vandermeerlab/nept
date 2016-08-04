import os

from startup import load_csc, load_videotrack, load_events, load_spikes, convert_to_cm

session_id = 'R063d2'

thisdir = os.path.dirname(os.path.realpath(__file__))
dataloc = os.path.abspath(os.path.join(thisdir, '..', 'cache', 'data'))

species = 'rat'
behavior = 'shortcut'
target = 'dCA1'
experimenter = 'Emily Irvine'


def get_csc(lfp_mat):
    return load_csc(os.path.join(dataloc, lfp_mat))


def get_pos(pxl_to_cm):
    pos = load_videotrack(os.path.join(dataloc, 'R063-2015-03-20-vt.mat'))
    pos['x'] = pos['x'] / pxl_to_cm[0]
    pos['y'] = pos['y'] / pxl_to_cm[1]
    return pos


def get_events():
    return load_events(os.path.join(dataloc, 'R063-2015-03-20-event.mat'))


def get_spikes():
    return load_spikes(os.path.join(dataloc, 'R063-2015-03-20-spike.mat'))

# plt.plot(pos['x'], pos['y'])
# plt.show()

# Experimental session-specific task times for R063 day 2
task_times = dict()
task_times['prerecord'] = [721.9412, 1027.1]
task_times['phase1'] = [1075.8, 1569.6]
task_times['pauseA'] = [1593.9, 2219.0]
task_times['phase2'] = [2243.4, 3512.4]
task_times['pauseB'] = [3556.1, 5441.3]
task_times['phase3'] = [5469.7, 8794.6]
task_times['postrecord'] = [8812.7, 9143.4]

pxl_to_cm = (8.8346, 7.1628)

fs = 2000

run_threshold = 0.2

good_lfp = ['R063-2015-03-20-CSC15a.ncs']
good_swr = ['R063-2015-03-20-CSC15.mat']
good_theta = ['R063-2015-03-20-CSC10.mat']

# Session-specific path trajectory points
path_pts = dict()
path_pts['feeder1'] = [468, 471]
path_pts['point1'] = [466, 397]
path_pts['turn1'] = [465, 380]
path_pts['point2'] = [416, 380]
path_pts['point3'] = [397, 370]
path_pts['point4'] = [368, 391]
path_pts['point5'] = [337, 376]
path_pts['point6'] = [293, 406]
path_pts['point7'] = [173, 367]
path_pts['turn2'] = [148, 359]
path_pts['point8'] = [138, 319]
path_pts['point9'] = [140, 103]
path_pts['turn3'] = [155, 69]
path_pts['point10'] = [203, 58]
path_pts['feeder2'] = [661, 54]
path_pts['shortcut1'] = [467, 378]
path_pts['point11'] = [450, 164]
path_pts['point12'] = [496, 166]
path_pts['point13'] = [645, 164]
path_pts['point14'] = [669, 162]
path_pts['point15'] = [672, 146]
path_pts['shortcut2'] = [661, 55]
path_pts['novel1'] = [146, 359]
path_pts['novel2'] = [49, 351]
path_pts['pedestal'] = [295, 200]

path_pts = convert_to_cm(path_pts, pxl_to_cm)

u_trajectory = [path_pts['feeder1'], path_pts['point1'], path_pts['turn1'],
                path_pts['point2'], path_pts['point3'], path_pts['point4'],
                path_pts['point5'], path_pts['point6'], path_pts['point7'],
                path_pts['turn2'], path_pts['point8'], path_pts['point9'],
                path_pts['turn3'], path_pts['point10'], path_pts['feeder2']]

shortcut_trajectory = [path_pts['shortcut1'], path_pts['point11'], path_pts['point12'],
                       path_pts['point13'], path_pts['point14'], path_pts['point15'],
                       path_pts['shortcut2']]

novel_trajectory = [path_pts['novel1'], path_pts['novel2']]

sequence = dict(u=dict(), shortcut=dict())
sequence['u']['swr_start'] = [9120.8, 9000.8]
sequence['u']['swr_stop'] = [9121.4, 9001.3]
sequence['u']['run_start'] = [2243.4, 2535]
sequence['u']['run_stop'] = [2273.4, 2565]
sequence['u']['ms'] = 15
sequence['u']['loc'] = 1
sequence['u']['colours'] = ['#bd0026', '#fc4e2a', '#ef3b2c', '#ec7014', '#fe9929',
                            '#78c679', '#41ab5d', '#238443', '#66c2a4', '#41b6c4',
                            '#1d91c0', '#8c6bb1', '#225ea8', '#88419d', '#ae017e']

sequence['shortcut']['swr_start'] = [9024.7, 9040.05, 9089.1]
sequence['shortcut']['swr_stop'] = [9025.3, 9040.65, 9089.9]
sequence['shortcut']['run_start'] = [8370, 8450, 8660]
sequence['shortcut']['run_stop'] = [8410, 8480, 8690]
sequence['shortcut']['ms'] = 10
sequence['shortcut']['loc'] = 1
sequence['shortcut']['colours'] = ['#bd0026', '#fc4e2a', '#ef3b2c', '#ec7014', '#fe9929',
                                   '#78c679', '#41ab5d', '#238443', '#66c2a4', '#41b6c4',
                                   '#1d91c0', '#8c6bb1', '#225ea8']
