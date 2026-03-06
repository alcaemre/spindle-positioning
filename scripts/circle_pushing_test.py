import src.spindle_state as ss

import numpy as np
# np.set_printoptions(formatter={'float': '{:.3f}'.format})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# %matplotlib widget

from rich.console import Console
from rich.live import Live
from rich.table import Table
from tqdm import tqdm

console = Console()


# starting with only three points
import src.lattice as lat
import src.plotting_tools as pt

N = 10

lattice_circle_3 = lat.circle_lattice(N)
# print(lattice_circle_3)

# spindle_state_circle_3 = np.array([1,1,1,1])
spindle_state_circle_3 = np.ones(N)

initial_mtoc_pos_circle_3 = np.array([-0.0,0.0,0])

# setting critical length and consequently rigidity
critical_length = 0.5
stall_force = 1
rigidity = stall_force * np.square(critical_length) / np.square(np.pi) 

spindle_circle_3 = ss.Spindle(
    initial_mtoc_pos_circle_3, 
    spindle_state_circle_3, 
    lattice_circle_3, 
    timestep_size=0.001,
    stall_force=stall_force,
    rigidity=rigidity,
    save=True,
    # dir_prefix='circle-pushing'
    dir_path='/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/circle-pushing_2026-02-27_16-19-52'
    )

spindle_circle_3.spindle_state

spindle_circle_3.add_microtubules(np.arange(N))

# # -- relax --
# with Live(console=console, refresh_per_second=4) as live:
#     traj, boundary_violated = spindle_circle_3.relax(resolution=4,readout=True, live=live)

#     spindle_circle_3.trajectory = spindle_circle_3.trajectory | traj
#     spindle_circle_3.current_time = list(traj.keys())[-1]

# ani = spindle_circle_3.animate(interval=1000, save=True)

# plt.show()


# print(spindle_circle_3.lattice_sites[:,0])
# fig, ax = pt.phase_portrait_2D(spindle_circle_3, 0, (-1.3,1.3), (-1.3,1.3), n=50, quiver=False)
# plt.show()

fig, ax = pt.radial_velocity_heatmap(spindle_circle_3, 0.1, (-1.3,1.3), (-1.3,1.3), n=500,)
plt.show()

