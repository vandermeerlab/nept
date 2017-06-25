import os
import numpy as np
import nept

rat_id = 'R042'
session_id = 'R042d3'
session = 'R042-2013-08-18'

species = 'rat'
behavior = 'motivational-t'
target = 'dCA1'
experimenter = 'Alyssa Carey and Matt van der Meer'

restriction_type = 'water'
layout = 'food_left'
pedestal_location = 'right'
n_pellets = 5
water_volume = []
n_trials = 18
forced_trials = None
non_consumption_trials = None
bad_trials = None

path_length = 257
path_arms = 254
track_dimensions = (139, 185)

event_filename = session + '-Events.nev'
event_labels = dict(start='Starting Recording',
                    stop='Stopping Recording',
                    one='TTL Input on AcqSystem1_0 board 0 port 1 value (0x0000).',
                    two='TTL Input on AcqSystem1_0 board 0 port 1 value (0x0020).',
                    three='TTL Input on AcqSystem1_0 board 0 port 1 value (0x0040).',
                    four='TTL Input on AcqSystem1_0 board 0 port 1 value (0x0080).',
                    five='TTL Output on AcqSystem1_0 board 0 port 0 value (0x0000).',
                    six='TTL Output on AcqSystem1_0 board 0 port 0 value (0x0004).',
                    seven='TTL Output on AcqSystem1_0 board 0 port 0 value (0x0040).')

position_filename = session + '-VT1.nvt'

lfp_swr_filename = session + '-CSC11a.ncs'
lfp_theta_filename = session + '-CSC07a.ncs'

spikes_filepath = session

fs = 2000

task_times = dict()
task_times['prerecord'] = nept.Epoch(np.array([2126.64553, 3214.07253]))
task_times['task'] = nept.Epoch(np.array([3238.67853, 5645.16153]))
task_times['on_track'] = nept.Epoch(np.array([3240, 5645]))
task_times['postrecord'] = nept.Epoch(np.array([5656.35353, 6563.46453]))

experiment_times = dict()
experiment_times['left_trials'] = nept.Epoch(np.array([[3240.5, 3282.1],
                                                       [3591.8, 3605.4],
                                                       [3744.1, 3754.9],
                                                       [3891.7, 3905.5],
                                                       [4145.1, 4170.3],
                                                       [4966.5, 4982.1],
                                                       [5085.7, 5106.4],
                                                       [5214.4, 5232.3],
                                                       [5330.3, 5357.6]]))
experiment_times['right_trials'] = nept.Epoch(np.array([[3433.5, 3448.2],
                                                        [4015.4, 4044.4],
                                                        [4267.6, 4284.5],
                                                        [4404.5, 4420.4],
                                                        [4540.3, 4583.4],
                                                        [4703.8, 4718.8],
                                                        [4822.6, 4870.3],
                                                        [5479.6, 5491.3],
                                                        [5583.6, 5622.4]]))

pxl_to_cm = (2.9176, 2.3794)
