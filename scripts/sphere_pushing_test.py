#
# Emre Alca
# University of Pennsylvania
# Created on Fri Mar 06 2026
# Last Modified: 2026/03/26 10:05:53
#
import src.spindle_state as ss
import src.plotting_tools as pt
import src.lattice as lat

import numpy as np
# np.set_printoptions(formatter={'float': '{:.3f}'.format})
import matplotlib.pyplot as plt
import trimesh

from rich.console import Console
from rich.live import Live
from rich.table import Table
console = Console()


console = Console()

if __name__ == "__main__":


    # -- sphere lattice
    # --- from thompson (4-130 pts) ---
    N = 30 # number of points
    R = 1 # radius of sphere
    sphere_lattice = np.load(f"/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/sloane_cache/sloane_{N}.npy")

    # sphere_lattice = np.array([
    #     [1, 0, 0],
    #     [-1, 0, 0],
    #     [0, 1, 0],
    #     [0, -1, 0],
    #     [0, 0, 1],
    #     [0, 0, -1],
    # ])

    # --- from trimesh (based on the number of recursive subdivisions)
    # boundary = trimesh.creation.icosphere(subdivisions=6, radius=1) # initializes discretized boundary boundary condition

    # sphere_lattice = np.array(boundary.vertices) # gets coordinates of the vertices of boundary as a numpy array--these are our valid impingement sites

    # N = sphere_lattice.shape[0] # number of total impingement sites

    # print(f'num sites: {N}')
    folder = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/sphere-pushing-phase-portraits'

    # -- 3d scatterplot -- 
    xs = sphere_lattice[:,0]
    ys = sphere_lattice[:,1]
    zs = sphere_lattice[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs,ys,zs, c='tab:orange')
    ax.set_aspect('equal', 'box')
    plt.savefig(f'{folder}/{N}-points-distribution')
    plt.show()

    sphere_spindle_state = np.ones(sphere_lattice.shape[0])
    initial_mtoc_pos_3d = np.array([0.1,0.1,0.1])

    # print(sphere_spindle_state.shape)

    # setting critical length and consequently rigidity
    critical_length = 0.1 #* np.sqrt(3)
    stall_force = 1
    rigidity = stall_force * np.square(critical_length) / np.square(np.pi) 

    sphere_spindle = ss.Spindle(
        initial_mtoc_pos_3d, 
        sphere_spindle_state, 
        sphere_lattice, 
        timestep_size=0.001,
        stall_force=stall_force,
        rigidity=rigidity,
        save=True,
        # dir_prefix='cube'
        dir_path='/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/cube_2026-02-27_15-50-22'
        )
    
    sphere_spindle.add_microtubules(np.arange(N))

    Z = -0.0
    x_lim = (-1,1)
    y_lim = (-1,1)
    num_points = 250
    fig, (ax_xy, ax_z) = pt.phase_portrait_2D_with_z(sphere_spindle, z=Z, xlim=x_lim, ylim=y_lim, n=num_points)
    # fig, ax = pt.radial_velocity_heatmap(sphere_spindle, Z, x_lim, y_lim, n=num_points,)
    plt.savefig(f'{folder}/{N}-points-phase-portrait')
    plt.show()

    # sphere_spindle.set_mtoc_pos(np.array([0.1,0,0]))

    # with Live(console=console, refresh_per_second=4) as live:
    #     sphere_spindle.fast_relax(readout=True, live=live)