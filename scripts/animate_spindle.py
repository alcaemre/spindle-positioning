#
# Emre Alca
# University of Pennsylvania
# Created on Sun Jan 25 2026
# Last Modified: 2026/01/26 13:56:17
#

# --- import box ---
import numpy as np
# import trimesh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
# import tqdm
from datetime import datetime

from IPython.display import HTML

from src import spindle_state as ss
import pickle
import os

def file_path_to_data(file_path):
    """
    given the path to a spindle trajectory file, extracts the data

    Args:
        file_path (str): string of path to .pkl file containing a spindle trajectory dictionary

    Returns:
        dict: spindle trajectory dictionary
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def find_sim_dims(data):
    """find dimensionality of a spindle simulation

    Args:
        data (dict): spindle trajectory data dict

    Returns:
        numpy.array: dimensions which are nonzero at any point in the simulation (e.g [0] or [0, 2] or [0, 1, 2])
    """

    # dimensionality of spindle
    spindle_dims = np.array([])
    for dim in range(3):
        if (data['spindle']['lattice_sites'][:,dim] != 0).any():
            spindle_dims = np.append(spindle_dims, dim).astype(int)

    # dimensionality of trajectory
    traj_dims = np.array([])
    for t in data['trajectory'].keys():
        dims = np.where(data['trajectory'][t]['mtoc_pos'] != 0)[0]
        for dim in dims:
            if dim not in traj_dims:
                traj_dims = np.append(traj_dims, dim).astype(int)

    sim_dims = np.unique(np.concatenate((spindle_dims, traj_dims)))

    return sim_dims


def animate_1d_spindle(data, dim, interval=100, save=False, file_prefix=None):
    """
    given a spindle trajectory dictionary, animate the 1-D behaviour in the specified dimension

    Args:
        data (dict): spindle trajectory data
        dim (int): dimension of interest
        save (bool, optional): saves fig as mp4 if true. Defaults to False.
        file_prefix (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # sim_dims = find_sim_dims(data)
    # -- static components --
    # static domain: the things which are set and not moved. In our case the MT lattice sites
    dim_labels = ['x', 'y', 'z']

    #initializing figure
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_box_aspect(3 / 1)
    ax.set_ylabel(f'{dim_labels[0]} position (dimensionless)')
    ax.xaxis.set_visible(False)

    # plot lattice sites (orange -> pushing, purlple -> pulling)
    site_color_list = np.where((data['spindle']['spindle_state'] == 1) + (data['spindle']['spindle_state'] == 2), 'tab:orange', 'tab:purple') # works from initial empty spindle state
    plotted_lattice_sites = ax.scatter(np.zeros(len(data['spindle']['lattice_sites'][:,dim])), data['spindle']['lattice_sites'][:,dim], c=site_color_list)

    # legend

    legend_handles = [
        Line2D([0],[0], marker='o', linestyle='None',
            color='tab:orange', alpha=0.8, markersize=8, label='Pushing'),
        Line2D([0],[0], marker='o', linestyle='None',
            color='tab:purple', alpha=0.8, markersize=8, label='Pulling'),
        Line2D([0],[0], marker='o', linestyle='None',
            color='tab:red', alpha=0.8, markersize=8, label='MTOC'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(2, 1.0),)


    # -- make dynamic artists -- 
    time_text = ax.text(0.02, 1.12, '', transform=ax.transAxes)
    mtoc_pos_text = ax.text(0.02, 1.07, '', transform=ax.transAxes)
    cost_text = ax.text(0.02, 1.02, '', transform=ax.transAxes)
    mtoc_pos = ax.scatter([], [], color='tab:red', label='mtoc_pos', zorder=3)
    present_mt_lines = LineCollection([], zorder=1)
    ax.add_collection(present_mt_lines)

    def init():
        mtoc_pos.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        time_text.set_text('')
        mtoc_pos_text.set_text('')
        cost_text.set_text('')
        present_mt_lines.set_segments([])
        present_mt_lines.set_colors([])
        return mtoc_pos, time_text, mtoc_pos_text, cost_text, present_mt_lines

    def update(frame):
        # time
        times = list(data['trajectory'].keys())[::interval]
        t = times[frame]
        time_text.set_text(f't = {np.round(t,3)}')
        # MTOC position
        mtoc_pos_num = data['trajectory'][t]['mtoc_pos'][dim]
        mtoc_pos_text.set_text(f'mtoc_pos = {np.round(mtoc_pos_num,3)}')
        mtoc_pos.set_offsets((0, data['trajectory'][t]['mtoc_pos'][dim]))
        # cost
        cost = data['trajectory'][t]['cost']
        cost_text.set_text(f'cost = {np.round(cost, 5)}')

        # MTs
        present_mt_indices = np.where(np.isin(data['trajectory'][t]['spindle_state'], [2, 4]))
        present_mt_colors = site_color_list[present_mt_indices]
        # lattices_with_mts_present = data['spindle']['lattice_sites'][present_mt_indices][dim] 
        lattices_with_mts_present = data['spindle']['lattice_sites'][present_mt_indices, dim][0]
        lines_for_present_mts = []
        for present_mt_site in lattices_with_mts_present.flatten():
            line = [(0, present_mt_site), (0, data['trajectory'][t]['mtoc_pos'][dim])]
            lines_for_present_mts.append(line)
        present_mt_lines.set_segments(lines_for_present_mts)
        present_mt_lines.set_colors(present_mt_colors)

        return mtoc_pos, time_text, mtoc_pos_text, cost_text, present_mt_lines

    #int(len(data['trajectory'].keys())/100)
    ani = FuncAnimation(fig, update, frames= int(len(data['trajectory'].keys())/interval), init_func=init,
                        interval=30, blit=False, repeat=True)
        
    if save:
        # finding data directory
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        target_child_dir = os.path.join(parent_dir, "data")
        os.makedirs(target_child_dir, exist_ok=True)

        # writing path to save file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if file_prefix is None:
            file_prefix = str(data['spindle']['spindle_state'].astype(int))
        filename = f"{file_prefix}_{timestamp}.mp4"

        file_path = os.path.join(target_child_dir, filename)
        ani.save(file_path, fps=60, dpi=150)
        print(f'animation saved to {file_path}')
        # plt.close()
    # else:
        # plt.show()

    return ani


def animate_2d_spindle(data, xdim, ydim, interval=100, save=False, file_prefix=None):
    # -- static components --
    dim_labels = ['x', 'y', 'z']

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_box_aspect(1)
    ax.set_ylim(((-1* data['spindle']['boundary_radius'])-0.2), (data['spindle']['boundary_radius'] + 0.2))
    ax.set_xlim(((-1* data['spindle']['boundary_radius'])-0.2), (data['spindle']['boundary_radius'] + 0.2))
    ax.set_ylabel(f'{dim_labels[ydim]} position (dimensionless)')
    ax.set_xlabel(f'{dim_labels[xdim]} position (dimensionless)')

    site_color_list = np.where((data['spindle']['spindle_state'] == 1) + (data['spindle']['spindle_state'] == 2), 'tab:orange', 'tab:purple') # works from initial empty spindle state
    plotted_lattice_sites = ax.scatter(data['spindle']['lattice_sites'][:,xdim], data['spindle']['lattice_sites'][:,ydim], c=site_color_list)

    # legend

    legend_handles = [
        Line2D([0],[0], marker='o', linestyle='None',
            color='tab:orange', alpha=0.8, markersize=8, label='Pushing'),
        Line2D([0],[0], marker='o', linestyle='None',
            color='tab:purple', alpha=0.8, markersize=8, label='Pulling'),
        Line2D([0],[0], marker='o', linestyle='None',
            color='tab:red', alpha=0.8, markersize=8, label='MTOC'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.33, 1.0),)

    # -- make dynamic artists -- 
    time_text = ax.text(0.02, 1.12, '', transform=ax.transAxes)
    mtoc_pos_text = ax.text(0.02, 1.07, '', transform=ax.transAxes)
    cost_text = ax.text(0.02, 1.02, '', transform=ax.transAxes)
    mtoc_pos = ax.scatter([], [], color='tab:red', label='mtoc_pos', zorder=3)
    present_mt_lines = LineCollection([], zorder=1)
    ax.add_collection(present_mt_lines)

    def init():
        mtoc_pos.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        mtoc_pos_text.set_text('')
        cost_text.set_text('')
        present_mt_lines.set_segments([])
        # present_mt_lines.set_colors([])
        return mtoc_pos, time_text, mtoc_pos_text, cost_text, present_mt_lines

    def update(frame):
        # time
        times = list(data['trajectory'].keys())[::interval]
        t = times[frame]
        time_text.set_text(f't = {np.round(t,3)}')

        # MTOC position
        current_mtoc_pos = (data['trajectory'][t]['mtoc_pos'][xdim], data['trajectory'][t]['mtoc_pos'][ydim])
        mtoc_pos_text.set_text(f'mtoc_pos = {np.round(current_mtoc_pos,3)}')
        mtoc_pos.set_offsets(current_mtoc_pos)

        # cost
        cost = data['trajectory'][t]['cost']
        cost_text.set_text(f'cost = {np.round(cost, 5)}')

        # present MTs
        present_mt_indices = np.where(np.isin(data['trajectory'][t]['spindle_state'], [2, 4]))
        present_mt_colors = site_color_list[present_mt_indices]
        # lattices_with_mts_present = data['spindle']['lattice_sites'][present_mt_indices][dim] 
        lattices_with_mts_present = data['spindle']['lattice_sites'][present_mt_indices][:, [xdim, ydim]]
        lines_for_present_mts = []
        for present_mt_site in lattices_with_mts_present:
            line = [present_mt_site, current_mtoc_pos]
            lines_for_present_mts.append(line)
        present_mt_lines.set_segments(lines_for_present_mts)
        present_mt_lines.set_colors(present_mt_colors)
        
        return mtoc_pos, time_text, mtoc_pos_text, cost_text, present_mt_lines

    ani = FuncAnimation(fig, update, frames= int(len(data['trajectory'].keys())/interval), init_func=init,
                            interval=30, blit=False, repeat=True)
    if save:
        # finding data directory
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        target_child_dir = os.path.join(parent_dir, "data")
        os.makedirs(target_child_dir, exist_ok=True)

        # writing path to save file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if file_prefix is None:
            file_prefix = str(data['spindle']['spindle_state'].astype(int))
        filename = f"{file_prefix}_{timestamp}.mp4"

        file_path = os.path.join(target_child_dir, filename)
        ani.save(file_path, fps=60, dpi=150)
        print(f'animation saved to {file_path}')
        # plt.close()
    # else:
        # plt.show()

    return ani

def animate_3d_spindle(data, interval=100, save=False, file_prefix=None):
    # -- static components --
    # figure setup
    dim_labels = ['x', 'y', 'z']

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(((-1* data['spindle']['boundary_radius'])-0.2), (data['spindle']['boundary_radius'] + 0.2))
    ax.set_ylim(((-1* data['spindle']['boundary_radius'])-0.2), (data['spindle']['boundary_radius'] + 0.2))
    ax.set_zlim(((-1* data['spindle']['boundary_radius'])-0.2), (data['spindle']['boundary_radius'] + 0.2))
    ax.set_xlabel('x position (dimensionless)')
    ax.set_ylabel('y position (dimensionless)')
    ax.set_zlabel('z position (dimensionless)')

    # lattice sites 
    site_color_list = np.where((data['spindle']['spindle_state'] == 1) + (data['spindle']['spindle_state'] == 2), 'tab:orange', 'tab:purple') # works from initial empty spindle state
    plotted_lattice_sites = ax.scatter(data['spindle']['lattice_sites'][:,0], data['spindle']['lattice_sites'][:,1], data['spindle']['lattice_sites'][:,2], c=site_color_list)

    # legend

    legend_handles = [
        Line2D([0],[0], marker='o', linestyle='None',
            color='tab:orange', alpha=0.8, markersize=8, label='Pushing'),
        Line2D([0],[0], marker='o', linestyle='None',
            color='tab:purple', alpha=0.8, markersize=8, label='Pulling'),
        Line2D([0],[0], marker='o', linestyle='None',
            color='tab:red', alpha=0.8, markersize=8, label='MTOC'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.3, 1.1),)

    # -- make dynamic artists -- 
    time_text = ax.text2D(0.02, 1.12, '', transform=ax.transAxes)
    mtoc_pos_text = ax.text2D(0.02, 1.07, '', transform=ax.transAxes)
    cost_text = ax.text2D(0.02, 1.02, '', transform=ax.transAxes)
    mtoc_pos = ax.scatter([], [], [], color='tab:red', label='mtoc_pos', zorder=3)
    present_mt_lines = Line3DCollection([], zorder=1)
    ax.add_collection(present_mt_lines)

    def init():
        mtoc_pos._offsets3d = ([], [], [])
        time_text.set_text('')
        mtoc_pos_text.set_text('')
        cost_text.set_text('')
        present_mt_lines.set_segments([])
        # present_mt_lines.set_colors([])
        return time_text, mtoc_pos_text, cost_text, mtoc_pos, present_mt_lines

    def update(frame):
        # time
        times = list(data['trajectory'].keys())[::interval]
        t = times[frame]
        time_text.set_text(f't = {np.round(t,3)}')

        # MTOC position
        current_mtoc_pos = (data['trajectory'][t]['mtoc_pos'])
        mtoc_pos_text.set_text(f'mtoc_pos = {np.round(current_mtoc_pos,3)}')
        mtoc_pos._offsets3d = (np.array([current_mtoc_pos[0]]), np.array([current_mtoc_pos[1]]), np.array([current_mtoc_pos[2]]))

        # cost
        cost = data['trajectory'][t]['cost']
        cost_text.set_text(f'cost = {np.round(cost, 5)}')

        # MTs
        present_mt_indices = np.where(np.isin(data['trajectory'][t]['spindle_state'], [2, 4]))
        present_mt_colors = site_color_list[present_mt_indices]
        # lattices_with_mts_present = data['spindle']['lattice_sites'][present_mt_indices][dim] 
        lattices_with_mts_present = data['spindle']['lattice_sites'][present_mt_indices]
        lines_for_present_mts = []
        for present_mt_site in lattices_with_mts_present:
            line = [present_mt_site, current_mtoc_pos]
            lines_for_present_mts.append(line)
        present_mt_lines.set_segments(lines_for_present_mts)
        present_mt_lines.set_colors(present_mt_colors)

        return time_text, mtoc_pos_text, cost_text, mtoc_pos, #present_mt_lines

    ani = FuncAnimation(fig, update, frames=int(len(data['trajectory'].keys())/interval), init_func=init,
                            interval=30, blit=False, repeat=True)

    if save:
        # finding data directory
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        target_child_dir = os.path.join(parent_dir, "data")
        os.makedirs(target_child_dir, exist_ok=True)

        # writing path to save file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if file_prefix is None:
            file_prefix = str(data['spindle']['spindle_state'].astype(int))
        filename = f"{file_prefix}_{timestamp}.mp4"

        file_path = os.path.join(target_child_dir, filename)
        ani.save(file_path, fps=60, dpi=150)
        print(f'animation saved to {file_path}')
        # plt.close()
    # else:
        # plt.show()

    return ani


def animate_spindle(data, interval=100, save=False, file_prefix=None):

    sim_dims = find_sim_dims(data)

    if len(sim_dims) == 3:
        ani = animate_3d_spindle(data, interval=interval, save=save, file_prefix=file_prefix)
    
    elif len(sim_dims) == 2:
        ani = animate_2d_spindle(data, int(sim_dims[0]), int(sim_dims[1]), interval=interval, save=save, file_prefix=file_prefix)

    elif len(sim_dims) == 1:
        ani = animate_1d_spindle(data, int(sim_dims[0]), interval=interval, save=save, file_prefix=file_prefix)
    
    return ani


if __name__ == "__main__":

    # -- 1-D test --
    # file_path = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/1d-validation/[1 3]_2026-01-22_10-11-06.pkl' # expect sim_dims = [0]
    # file_path = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/1d-validation/[3 1]_2026-01-22_10-12-39.pkl'
    # data = file_path_to_data(file_path)

    # ani = animate_1d_spindle(data, 0, save=False)

    # -- 2-D test -- 
    # file_path = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/1d-validation/[3 3]_2026-01-23_13-58-49.pkl' # sim_dims = [0, 1]
    # data = file_path_to_data(file_path)

    # dims = find_sim_dims(data)
    # xdim = dims[0] 
    # ydim = dims[1]

    # ani = animate_2d_spindle(data, xdim, ydim, interval=500, save=False)

    # plt.show()

    # -- 3-D test --
    file_path = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/octahedron/[1 1 3 3 1 1]_2026-01-16_14-04-53.pkl' # sim dims = [0,1,2]

    data = file_path_to_data(file_path)
    ani = animate_3d_spindle(data, interval=100, save=True)
    # plt.show()

    # -- adaptive test -- 

    # use data from any of the tests above

    # ani = animate_spindle(data)
    # plt.show()