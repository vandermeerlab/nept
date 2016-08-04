import os

from startup import load_csc, load_videotrack, load_events, load_spikes, convert_to_cm

session_id = 'R063d3'

thisdir = os.path.dirname(os.path.realpath(__file__))
dataloc = os.path.abspath(os.path.join(thisdir, '..', 'cache', 'data'))


species = 'rat'
behavior = 'shortcut'
target = 'dCA1'
experimenter = 'Emily Irvine'


def get_csc(lfp_mat):
    return load_csc(os.path.join(dataloc, lfp_mat))


def get_pos(pxl_to_cm):
    pos = load_videotrack(os.path.join(dataloc, 'R063-2015-03-22-vt.mat'))
    pos['x'] = pos['x'] / pxl_to_cm[0]
    pos['y'] = pos['y'] / pxl_to_cm[1]
    return pos


def get_events():
    return load_events(os.path.join(dataloc, 'R063-2015-03-22-event.mat'))


def get_spikes():
    return load_spikes(os.path.join(dataloc, 'R063-2015-03-22-spike.mat'))

# plt.plot(pos['x'], pos['y'])
# plt.show()

# Experimental session-specific task times for R063 day 3
task_times = dict()
task_times['prerecord'] = [837.4714, 1143.1]
task_times['phase1'] = [1207.9, 2087.5]
task_times['pauseA'] = [2174.3, 2800.8]
task_times['phase2'] = [2836.2, 4034.1]
task_times['pauseB'] = [4051.3, 6185.6]
task_times['phase3'] = [6249.5, 9373.7]
task_times['postrecord'] = [9395.4, 9792.5]

pxl_to_cm = (7.3452, 7.2286)

fs = 2000

run_threshold = 0.5

good_lfp = ['R063-2015-03-22-CSC13a.ncs']
good_swr = ['R063-2015-03-22-CSC13.mat']
good_theta = ['R063-2015-03-22-CSC15.mat']

# Session-specific path trajectory points
path_pts = dict()
path_pts['feeder1'] = [547, 457]
path_pts['point1'] = [558, 451]
path_pts['point2'] = [545, 401]
path_pts['turn1'] = [542, 374]
path_pts['point3'] = [511, 380]
path_pts['point4'] = [443, 399]
path_pts['point5'] = [362, 418]
path_pts['point6a'] = [340, 407]
path_pts['point6'] = [310, 385]
path_pts['point7'] = [292, 404]
path_pts['point8'] = [255, 379]
path_pts['turn2'] = [217, 375]
path_pts['point9'] = [217, 316]
path_pts['point10'] = [236, 84]
path_pts['turn3'] = [249, 59]
path_pts['point11'] = [289, 51]
path_pts['point12'] = [532, 47]
path_pts['feeder2'] = [670, 56]
path_pts['shortcut1'] = [446, 391]
path_pts['spt1'] = [438, 334]
path_pts['spt2'] = [449, 295]
path_pts['spt3'] = [471, 277]
path_pts['spt4'] = [621, 269]
path_pts['spt5'] = [648, 280]
path_pts['spt6'] = [671, 266]
path_pts['spt7'] = [660, 240]
path_pts['shortcut2'] = [654, 56]
path_pts['novel1'] = [247, 61]
path_pts['npt1'] = [146, 53]
path_pts['npt2'] = [132, 83]
path_pts['novel2'] = [130, 266]
path_pts['pedestal'] = [368, 188]

path_pts = convert_to_cm(path_pts, pxl_to_cm)

u_trajectory = [path_pts['feeder1'], path_pts['point1'], path_pts['point2'], path_pts['turn1'],
                path_pts['point3'], path_pts['point4'], path_pts['point5'], path_pts['point6a'], path_pts['point6'],
                path_pts['point7'], path_pts['point8'], path_pts['turn2'], path_pts['point9'],
                path_pts['point10'], path_pts['turn3'], path_pts['point11'], path_pts['point12'],
                path_pts['feeder2']]

shortcut_trajectory = [path_pts['shortcut1'], path_pts['spt1'], path_pts['spt2'],
                       path_pts['spt3'], path_pts['spt4'], path_pts['spt5'], path_pts['spt6'],
                       path_pts['spt7'], path_pts['shortcut2']]

novel_trajectory = [path_pts['novel1'], path_pts['npt1'],
                    path_pts['npt2'], path_pts['novel2']]

sequence = dict(u=dict(), shortcut=dict())
sequence['u']['swr_start'] = [9692.2, 9735.65]
sequence['u']['swr_stop'] = [9692.5, 9736.1]
sequence['u']['run_start'] = [2950, 3285]
sequence['u']['run_stop'] = [2975, 3315]
sequence['u']['ms'] = 15
sequence['u']['loc'] = 2
sequence['u']['colours'] = ['#bd0026', '#fc4e2a', '#ef3b2c', '#ec7014', '#fe9929',
                            '#78c679', '#41ab5d', '#238443', '#66c2a4', '#41b6c4',
                            '#1d91c0']

sequence['shortcut']['swr_start'] = [9692.2, 9450.01, 9735.75]
sequence['shortcut']['swr_stop'] = [9692.7, 9450.4, 9736.1]
sequence['shortcut']['run_start'] = [8000, 7950, 8035]
sequence['shortcut']['run_stop'] = [8030, 7980, 8065]
sequence['shortcut']['ms'] = 10
sequence['shortcut']['loc'] = 1
sequence['shortcut']['colours'] = ['#bd0026', '#fc4e2a', '#ef3b2c', '#ec7014', '#fe9929',
                                   '#78c679', '#41ab5d', '#238443', '#66c2a4', '#41b6c4',
                                   '#1d91c0', '#8c6bb1', '#225ea8', '#88419d', '#ae017e',
                                   '#dd3497', '#f768a1', '#fcbba1', '#fc9272', '#fb6a4a',
                                   '#e31a1c', '#fb6a4a', '#993404', '#b30000', '#800026']
