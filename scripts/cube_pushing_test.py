import src.spindle_state as ss

import numpy as np
# np.set_printoptions(formatter={'float': '{:.3f}'.format})
import matplotlib.pyplot as plt
# %matplotlib widget

from rich.console import Console
from rich.live import Live
from rich.table import Table
import src.plotting_tools as pt

console = Console()

cube_lattice = np.array([
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [1, 1, -1],
    [-1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1],
])
cube_spindle_state = np.array([1,1,1,1,1,1,1,1])

initial_mtoc_pos_3d = np.array([0.1,0.1,0.1])

# setting critical length and consequently rigidity
critical_length = 0.5 * np.sqrt(3)
stall_force = 1
rigidity = stall_force * np.square(critical_length) / np.square(np.pi) 

cube_spindle = ss.Spindle(
    initial_mtoc_pos_3d, 
    cube_spindle_state, 
    cube_lattice, 
    timestep_size=0.001,
    stall_force=stall_force,
    rigidity=rigidity,
    save=True,
    # dir_prefix='cube'
    dir_path='/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/cube_2026-02-27_15-50-22'
    )

# cube_spindle.add_microtubules(np.array([0,3,4,7,]))
cube_spindle.add_microtubules(np.array([0,1,2,3,4,5,6,7]))
Z = 0
x_lim = (-1.3,1.3)
y_lim = (-1.3,1.3)
N = 250

# print(cube_spindle.mt_dirs)

# print(f'total pushing force: \n {cube_spindle.calculate_pushing_forces()}')
# # -- relax --
# with Live(console=console, refresh_per_second=4) as live:
#     traj, boundary_violated = cube_spindle.fast_relax(resolution=10, readout=True, live=live)
#     # traj, boundary_violated = cube_spindle.relax(resolution=10, readout=True, live=live)

#     cube_spindle.trajectory = cube_spindle.trajectory | traj
#     cube_spindle.current_time = list(traj.keys())[-1]

# ani = cube_spindle.animate(interval=5000, save=True)

# plt.show()

# fig, ax = pt.radial_velocity_heatmap(cube_spindle, Z, x_lim, y_lim, n=N,)
# plt.show()

# fix, ax = pt.phase_portrait_2D(cube_spindle, Z, xlim=x_lim, ylim=y_lim, n=N)
# plt.show()

# fig, ax, equilibria, labels = basin_of_attraction_3D(cube_spindle, n_seeds=300, seed=42)
# print(f"Found {len(equilibria)} attractors:\n{equilibria}")
fig, (ax_xy, ax_z) = pt.phase_portrait_2D_with_z(cube_spindle, z=Z, xlim=x_lim, ylim=y_lim, n=N)
plt.show()
