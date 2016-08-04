import os

from startup import load_csc, load_videotrack, load_events, load_spikes, convert_to_cm

session_id = 'R066d3'

thisdir = os.path.dirname(os.path.realpath(__file__))
dataloc = os.path.abspath(os.path.join(thisdir, '..', 'cache', 'data'))

species = 'rat'
behavior = 'shortcut'
target = 'dCA1'
experimenter = 'Emily Irvine'


def get_csc(lfp_mat):
    return load_csc(os.path.join(dataloc, lfp_mat))


def get_pos(pxl_to_cm):
    pos = load_videotrack(os.path.join(dataloc, 'R066-2014-11-29-vt.mat'))
    pos['x'] = pos['x'] / pxl_to_cm[0]
    pos['y'] = pos['y'] / pxl_to_cm[1]
    return pos


def get_events():
    return load_events(os.path.join(dataloc, 'R066-2014-11-29-event.mat'))


def get_spikes():
    return load_spikes(os.path.join(dataloc, 'R066-2014-11-29-spike.mat'))


# plt.plot(pos['x'], pos['y'], 'b.', ms=1)
# plt.show()

# Experimental session-specific task times for R066 day 2
task_times = dict()
task_times['prerecord'] = [20258.0, 20593.0]
task_times['phase1'] = [20688.0, 21165.0]
task_times['pauseA'] = [21218.0, 21828.0]
task_times['phase2'] = [21879.0, 23081.0]
task_times['pauseB'] = [23136.0, 24962.0]
task_times['phase3'] = [25009.0, 27649.0]
task_times['postrecord'] = [27698.0, 28011.0]

pxl_to_cm = (7.2599, 7.2286)

fs = 2000

run_threshold = 0.35

good_lfp = ['R066-2014-11-29-CSC12c.ncs']
good_swr = ['R066-2014-11-29-CSC12.mat']
good_theta = ['R066-2014-11-29-CSC07.mat']

# Session-specific path trajectory points
path_pts = dict()
path_pts['feeder1'] = [525, 471]
path_pts['pt1'] = [528, 399]
path_pts['turn1'] = [522, 373]
path_pts['pt2'] = [511, 367]
path_pts['pt3'] = [498, 377]
path_pts['pt4'] = [480, 373]
path_pts['pt5'] = [437, 387]
path_pts['pt6'] = [357, 394]
path_pts['pt7'] = [322, 389]
path_pts['pt8'] = [225, 365]
path_pts['turn2'] = [197, 358]
path_pts['pt10'] = [198, 333]
path_pts['pt11'] = [204, 83]
path_pts['turn3'] = [211, 61]
path_pts['pt13'] = [230, 53]
path_pts['pt14'] = [395, 40]
path_pts['feeder2'] = [650, 57]
path_pts['shortcut1'] = [422, 380]
path_pts['spt1'] = [420, 321]
path_pts['spt2'] = [432, 284]
path_pts['spt3'] = [461, 262]
path_pts['spt4'] = [600, 270]
path_pts['spt5'] = [634, 270]
path_pts['spt6'] = [638, 245]
path_pts['spt7'] = [630, 85]
path_pts['shortcut2'] = [650, 57]
path_pts['novel1'] = [211, 61]
path_pts['npt1'] = [162, 37]
path_pts['npt2'] = [124, 43]
path_pts['npt3'] = [107, 75]
path_pts['novel2'] = [104, 156]
path_pts['pedestal'] = [338, 186]

path_pts = convert_to_cm(path_pts, pxl_to_cm)

u_trajectory = [path_pts['feeder1'], path_pts['pt1'], path_pts['turn1'], path_pts['pt2'],
                path_pts['pt3'], path_pts['pt4'], path_pts['pt5'],
                path_pts['pt6'], path_pts['pt7'], path_pts['pt8'],
                path_pts['turn2'], path_pts['pt10'], path_pts['pt11'],
                path_pts['turn3'], path_pts['pt13'], path_pts['pt14'],
                path_pts['feeder2']]

shortcut_trajectory = [path_pts['shortcut1'], path_pts['spt1'], path_pts['spt2'],
                       path_pts['spt3'], path_pts['spt4'], path_pts['spt5'],
                       path_pts['spt6'], path_pts['spt7'], path_pts['shortcut2']]

novel_trajectory = [path_pts['novel1'], path_pts['npt1'], path_pts['npt2'],
                    path_pts['npt2'], path_pts['novel2']]

sequence = dict(u=dict(), shortcut=dict())
sequence['u']['swr_start'] = [27888.2, 27924.45]
sequence['u']['swr_stop'] = [27888.6, 27924.85]
sequence['u']['run_start'] = [25360, 25320]
sequence['u']['run_stop'] = [25390, 25350]
sequence['u']['ms'] = 15
sequence['u']['loc'] = 1
sequence['u']['colours'] = ['#bd0026', '#fc4e2a', '#ef3b2c', '#ec7014', '#fe9929',
                            '#78c679', '#41ab5d', '#238443', '#66c2a4', '#41b6c4',
                            '#1d91c0', '#8c6bb1', '#225ea8']

sequence['shortcut']['swr_start'] = [27791.2, 27833.7]
sequence['shortcut']['swr_stop'] = [27791.8, 27834.7]
sequence['shortcut']['run_start'] = [25170, 25265]
sequence['shortcut']['run_stop'] = [25210, 25305]
sequence['shortcut']['ms'] = 10
sequence['shortcut']['loc'] = 2
sequence['shortcut']['colours'] = ['#bd0026', '#fc4e2a', '#ef3b2c', '#ec7014', '#fe9929',
                                   '#78c679', '#41ab5d', '#238443', '#66c2a4', '#41b6c4',
                                   '#1d91c0', '#8c6bb1', '#225ea8', '#88419d', '#ae017e',
                                   '#dd3497', '#f768a1', '#fcbba1']
