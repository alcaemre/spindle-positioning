#
# Emre Alca
# University of Pennsylvania
# Created on Thu Jan 08 2026
# Last Modified: 2026/01/08 16:14:12
#

# -- import box --

import numpy as np
np.set_printoptions(formatter={'float': '{:.3f}'.format})
import trimesh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from tqdm import tqdm
import sys
import time
from IPython.display import HTML

from rich.console import Console
from rich.live import Live
from rich.table import Table

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

test_spindle_state = np.array([1, 1, 3, 3, 1, 1])

test_spindle = ss.Spindle(np.array([0, 0, 0]), test_spindle_state, test_spindle_lattice, timestep_size=0.001)

# -- developing time development loop with spindle updates -- 

# set unstable position and set of MTS

test_spindle.add_microtubules([0,1])
test_spindle.set_mtoc_pos(np.array([0, 0.1, 0]))

# set timer and max time, timestep size is set when initializing the Spindle
t = 0 
max_time = 200

boundary_violated = False
with Live(console=console, refresh_per_second=4) as live:
    while t < (max_time - test_spindle.timestep_size) and not boundary_violated:

        # MTOC position and cost before time evolution
        old_mtoc_pos = test_spindle.mtoc_pos
        old_cost = test_spindle.calc_cost()

        # time evolution and saving MTOC position and cost after time evolution
        new_mtoc_pos, boundary_violated = test_spindle.mtoc_time_evolution()
        new_cost = test_spindle.calc_cost()

        # if new_cost > old_cost, change the spindle state

        if new_cost > old_cost:
            # change spindle state
            a = 1

        t = t + test_spindle.timestep_size

        # readout table
        table = Table(title="Simulation Status")
        table.add_column("Parameter", justify="left")
        table.add_column("Value", justify="right")
        table.add_row("Time", f"{t:.2f}")
        table.add_row("Progress", f"{(100 * t/max_time):.2f}%")
        table.add_row("Boundary Violated", str(boundary_violated))
        table.add_row("Current Position", str(test_spindle.mtoc_pos))
        live.update(table)
        
        time.sleep(0.0001)


