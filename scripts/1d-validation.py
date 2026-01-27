#
# Emre Alca
# University of Pennsylvania
# Created on Thu Jan 22 2026
# Last Modified: 2026/01/27 13:57:19
#

# -- import box --

import numpy as np
np.set_printoptions(formatter={'float': '{:.3f}'.format})
import matplotlib.pyplot as plt

from src import spindle_state as ss
import animate_spindle as anisp

test_1d_lattice = np.array([
    [1, 0, 0],
    [-1, 0, 0],
])

# test_1d_spindle_state = np.array([1,1]) # both pushing (stable, degenerate, buckling)
test_1d_spindle_state = np.array([3,3]) # both pulling (stable, degenerate)
# test_1d_spindle_state = np.array([3,1]) # unstable (towards positive)
# test_1d_spindle_state = np.array([1,3]) # unstable (towards negative)

initial_mtoc_pos = np.array([0.5,0.5,0])

test_spindle = ss.Spindle(initial_mtoc_pos, test_1d_spindle_state, test_1d_lattice, timestep_size=0.001, max_total_mt_length=(10*np.sqrt(3) + 0.001), mt_len_cost_punishment_degree=2)

# -- developing time development loop with spindle updates -- 

# set unstable position and set of MTS

test_spindle.add_microtubules([0,1])
# test_spindle.set_mtoc_pos(np.array([0, 0.2, 0.6]))

# set timer and max time, timestep size is set when initializing the Spindle
max_time = 200

file = '2d-pulling-degeneracy'
save=False 

data = test_spindle.simulate(max_time, readout=True, save=save, file_prefix=file, update_spindle=False)

ani = anisp.animate_2d_spindle(data, 0, 1, interval=500, save=save, file_prefix=file)

plt.show()