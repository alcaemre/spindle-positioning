#
# Emre Alca
# University of Pennsylvania
# Created on Thu Jan 08 2026
# Last Modified: 2026/01/16 13:52:10
#

# -- import box --

import numpy as np
np.set_printoptions(formatter={'float': '{:.3f}'.format})
# import trimesh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from rich.console import Console
from rich.live import Live
from rich.table import Table

import pickle
import os
from datetime import datetime

console = Console()

from src import spindle_state as ss

# -- initializing test_spindle --

test_spindle_lattice = np.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
])

expected_mt_vecs = np.array([
       [ 0.5,  0. ,  0. ],
       [-1.5,  0. ,  0. ],
       [-0.5,  1. ,  0. ],
       [-0.5, -1. ,  0. ],
       [-0.5,  0. ,  1. ],
       [-0.5,  0. , -1. ]])

# test_spindle_state = np.array([3, 3, 3, 3, 3, 3])
test_spindle_state = np.array([1, 1, 1, 1, 1, 1])
# test_spindle_state = np.array([1, 1, 3, 3, 1, 1])

test_spindle = ss.Spindle(np.array([0, 0, 0]), test_spindle_state, test_spindle_lattice, timestep_size=0.001, max_total_mt_length=(10*np.sqrt(3) + 0.001), mt_len_cost_punishment_degree=2)

# -- developing time development loop with spindle updates -- 

# set unstable position and set of MTS

test_spindle.add_microtubules([0,1])
test_spindle.set_mtoc_pos(np.array([0.4, 0.2, 0]))

# set timer and max time, timestep size is set when initializing the Spindle
max_time = 200

test_spindle.simulate(max_time, readout=True, save=False)