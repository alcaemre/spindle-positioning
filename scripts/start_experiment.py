#
# Emre Alca
# University of Pennsylvania
# Created on Fri Feb 13 2026
# Last Modified: 2026/04/17 11:43:15
#

import numpy as np
import matplotlib.pyplot as plt
import trimesh

from src import spindle_state as ss

import argparse

from rich.console import Console
from rich.live import Live
from rich.table import Table
console = Console()

def find_initial_spindle_state(lattice, num_initial_mts):
    # choose N // 2 random sites in lattice
    first_half_indices = np.random.randint(0, lattice.shape[0], num_initial_mts//2)
    minus_first_half = -lattice[first_half_indices]
    
    # find nearest neighbours to min_first_half on lattice
    diffs = minus_first_half[:, np.newaxis, :] - lattice[np.newaxis, :, :]  # (M, N, 3)
    dists = np.linalg.norm(diffs, axis=2)               # (M, N)
    indices = np.argmin(dists, axis=1)                  # (M,) — one match per B vector

    second_half = lattice[indices]  # (M, 3)
    return np.append(first_half_indices, indices)


if __name__ == "__main__":
    # --- Set basic numbers ---
    num_attempts = 1000 # number of stable positions with decreasing cost before we kill the simulation
    N = 100 # tubulin_budget, expected number of MTs when in cost basin
    relax_time = 10
    R = 1 # radius of sphere

    # --- set basic numbers from argparse ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--relax_time", type=float, default=np.inf)
    parser.add_argument("--mean_nuc", type=int, default=1)
    parser.add_argument("--mean_cat", type=int, default=1)

    args = parser.parse_args()
    N = int(args.budget)
    relax_time = args.relax_time
    mean_nuc = args.mean_nuc
    mean_cat = args.mean_cat
    print(N, relax_time, mean_nuc, mean_cat)
    
    # --- creating circular discrete lattice ---
    # M = 1000 # number of lattice sites 
    # theta = np.arange(0, 2* np.pi, 2* np.pi/M)
    # x = np.cos(theta)
    # y = np.sin(theta)
    # # theta
    # # (x,y)

    # lattice = []
    # spindle_state = []
    # for i in range(len(theta)):
    #     lattice.append([x[i], y[i], 0])
    #     spindle_state.append(1) # 1 for pushing 3 for pulling

    # lattice = np.array(lattice)

    # --- 3D lattice ---
    # -- from thompson (4-130 pts) --
    # M = 130 # number of points
    # sphere_lattice = np.load(f"/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/sloane_cache/sloane_{M}.npy")

    # -- thousands of MTs -- 
    boundary = trimesh.creation.icosphere(subdivisions=4, radius=1) # initializes discretized boundary boundary condition
    boundary_sites = np.array(boundary.vertices) 
    sphere_lattice = boundary_sites
    M = boundary_sites.shape[0]
    print(f'{M} sites, {N} MTs')
    spindle_state = []
    for i in range(M):
        spindle_state.append(1) # 1 for pushing 3 for pulling

    lattice = sphere_lattice

    import pickle
    with open('lattice.pkl', "wb") as f:
            pickle.dump(lattice, f)



    # --- initializing spindle ---

    file_prefix = f'3D_pushing_{M}_points_{N}_MTs_{relax_time}_relax_time_{mean_nuc}_mean_nuc_{mean_cat}_mean_cat_{num_attempts}_attempts'
    # dir_prefix='testing_reconstruction'
    dir_prefix = file_prefix
    # dir = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/testing_max_relax_time_pushing_2D'

    spindle_state = np.array(spindle_state)

    initial_mtoc_pos = np.array([0.0, 0, 0])

    test_spindle = ss.Spindle(
        initial_mtoc_pos, 
        spindle_state, 
        lattice, 
        tubulin_budget=N, 
        timestep_size=0.001, 
        rigidity=0.05, 
        # nucleation_rate=20, 
        # catastrophe_rate=0.5, 
        dir_prefix=dir_prefix, 
        save=True, 
        readout=True,
        # dir_path=dir
        max_relax_time=relax_time
        )

    # initial_indices = np.random.randint(0, M, N)
    initial_indices = find_initial_spindle_state(lattice, N)
    # initial_indices = np.arange(M)

    test_spindle.add_microtubules(initial_indices)

    # with Live(console=console, refresh_per_second=4) as live:
    #     test_spindle.fast_relax(resolution=100, readout=True, live=live)

    # --- simulate spindle with 1 MT updates ---

    test_spindle.spindle_optimization_uniform(num_attempts=num_attempts, resolution=5)

    print(np.round(test_spindle.mtoc_pos, 3))
    print(np.round(test_spindle.current_time, 3))
    print(np.round(test_spindle.calc_cost(), 3))

    # test_spindle.save_trajectory(file_prefix)

    ani = test_spindle.animate(interval=1000, save=False, file_prefix=file_prefix)
    plt.show()