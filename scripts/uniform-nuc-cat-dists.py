#
# Emre Alca
# University of Pennsylvania
# Created on Fri Feb 20 2026
# Last Modified: 2026/02/20 17:52:30
#

import numpy as np
import matplotlib.pyplot as plt

from src import spindle_state as ss

from rich.console import Console
console = Console()

import time


if __name__ == "__main__":
    # --- Set basic numbers ---
    M = 100 # number of lattice sites 
    # N = 5 # number of MTs
    tubulin_budget = 10
    num_rounds = 3 # number of stable positions with decreasing cost before we kill the simulation

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

    file_prefix = f'2D-single-mt-update-{M}-points-{tubulin_budget}-mts-{num_rounds}-positions'

    spindle_state = np.array(spindle_state)

    initial_mtoc_pos = np.array([0., 0, 0])

    test_spindle = ss.Spindle(
        initial_mtoc_pos, 
        spindle_state, 
        lattice, 
        tubulin_budget=10, 
        timestep_size=0.001, 
        rigidity=0.05, 
        nucleation_rate=20, 
        catastrophe_rate=0.5, 
        dir_prefix=file_prefix, 
        save=True, 
        readout=True,
        seed=1,
        )

    test_spindle.single_mt_update_simulation_to_equilibrium(num_positions=num_rounds, resolution=3)

    # dir_path = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/2D-single-mt-update-100-points-10-mts-3-positions_2026-02-20_17-09-05'

    # print('restarting simulation')
    # test_spindle = ss.restart_experiment_from_directory(dir_path, 10)

    # test_spindle.single_mt_update_simulation_to_equilibrium(10)

    test_spindle.save_trajectory('trajectory')

    ani = test_spindle.animate(interval=100, save=True, file_prefix=test_spindle)
    plt.show()
