import os
import matplotlib.pyplot as plt
import vdmlab as vdm

from tuning_curves_functions import get_tc, linearize
from plotting_functions import plot_sorted_tc

import info.R063d2_info as r063d2
import info.R063d3_info as r063d3
import info.R063d4_info as r063d4
import info.R063d5_info as r063d5
import info.R063d6_info as r063d6
import info.R066d1_info as r066d1
import info.R066d2_info as r066d2
import info.R066d4_info as r066d4


thisdir = os.path.dirname(os.path.realpath(__file__))


info = r066d4
# infos = [r063d2, r063d3, r063d4, r063d5, r063d6, r066d1, r066d2, r066d4]

pickle_filepath = os.path.join(thisdir, 'cache', 'pickled')
output_filepath = os.path.join(thisdir, 'plots', 'decode')

print(info.session_id)
pos = info.get_pos(info.pxl_to_cm)

t_start = info.task_times['phase3'][0]
t_stop = info.task_times['phase3'][1]

t_start_idx = vdm.find_nearest_idx(pos['time'], t_start)
t_end_idx = vdm.find_nearest_idx(pos['time'], t_stop)

sliced_pos = dict()
sliced_pos['x'] = pos['x'][t_start_idx:t_end_idx]
sliced_pos['y'] = pos['y'][t_start_idx:t_end_idx]
sliced_pos['time'] = pos['time'][t_start_idx:t_end_idx]

linear, zone = linearize(info, pos)

spikes = info.get_spikes()

tc = get_tc(info, sliced_pos, pickle_filepath)

# sort_idx = vdm.get_sort_idx(tc['u'])
# ordered_spikes = spikes['time'][sort_idx]

for thisposition in linear['u']['time'][1000:10500]:
    x = vdm.find_nearest_idx(pos['time'], thisposition)
    # plt.plot(pos['time'][x], pos['x'][x], 'b.')
    # plt.plot(pos['time'][x], pos['y'][x], 'g.')
    plt.plot(pos['x'][x], pos['y'][x], 'r.')
    plt.show()
