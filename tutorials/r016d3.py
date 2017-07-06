import os
import numpy as np
import nept

rat_id = 'R016'
session_id = 'R016d3'
session = 'R016-2012-10-03'

species = 'rat'
behavior = 'value-risk'
target = 'dCA1'
experimenter = 'unknown'

event_filename = session + '-Events.nev'
event_labels = dict(start='Starting Recording',
                    stop='Stopping Recording')

position_filename = session + '-VT1.nvt'

lfp_gamma_filename1 = session + '-CSC04d.ncs'
lfp_gamma_filename2 = session + '-CSC03d.ncs'
lfp_theta_filename = session + '-CSC02b.ncs'

spikes_filepath = session

task_times = dict()
task_times['prerecord'] = nept.Epoch(np.array([1020, 1320]))
task_times['task-value'] = nept.Epoch(np.array([1428, 2674]))
task_times['task-reward'] = nept.Epoch(np.array([2700, 3321]))
task_times['postrecord'] = nept.Epoch(np.array([3328, 3649]))

pxl_to_cm = (3., 2.3)
