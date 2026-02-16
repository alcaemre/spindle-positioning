#
# Emre Alca
# University of Pennsylvania
# Created on Fri Feb 13 2026
# Last Modified: 2026/02/16 11:51:50
#

import numpy as np
import matplotlib.pyplot as plt

from src import spindle_state as ss

from rich.console import Console
console = Console()


if __name__ == "__main__":
    # --- Set basic numbers ---
    M = 200 # number of lattice sites 
    N = 10 # number of MTs
    num_positions = 50 # number of stable positions with decreasing cost before we kill the simulation

    # --- creating circular discrete lattice ---
    theta = np.arange(0, 2* np.pi, 2* np.pi/M)
    x = np.cos(theta)
    y = np.sin(theta)
    # theta
    (x,y)

    lattice = []
    spindle_state = []
    for i in range(len(theta)):
        lattice.append([x[i], y[i], 0])
        spindle_state.append(3)

    lattice = np.array(lattice)

    # --- initializing spindle ---
    spindle_state = np.array(spindle_state)

    initial_mtoc_pos = np.array([0., 0, 0])

    test_spindle = ss.Spindle(initial_mtoc_pos, spindle_state, lattice, tubulin_budget=N, timestep_size=0.001, rigidity=0.05, nucleation_rate=20, catastrophe_rate=0.5)

    initial_indices = np.random.randint(0, M, N)

    test_spindle.add_microtubules(initial_indices)

    # --- simulate spindle with 1 MT updates ---

    test_spindle.single_mt_update_simulation_to_equilibrium(num_positions=num_positions, resolution=3)

    file_prefix = f'2D-single-mt-update-{M}-points-{N}-mts-{num_positions}-positions'

    test_spindle.save_trajectory(file_prefix)

    ani = test_spindle.animate(interval=1000, save=True, file_prefix=file_prefix)
    plt.show()