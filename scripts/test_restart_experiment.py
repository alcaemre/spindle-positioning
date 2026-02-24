#
# Emre Alca
# University of Pennsylvania
# Created on Fri Feb 20 2026
# Last Modified: 2026/02/24 01:30:00
#

import numpy as np
np.set_printoptions(formatter={'float': '{:.3f}'.format})
import tqdm
import matplotlib.pyplot as plt

from datetime import datetime

import pickle
import os

import src.animate_spindle as anisp
import src.spindle_state as ss

from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()



def restart_experiment_from_directory(dir_path, readout=False):
    """restarts an experiment from a particular directory and rebuild the trajectory outlined in its spindle trace

    Args:
        dir_path (str): path to experiment directory

    Returns:
        Spindle: spindle upon finishing simulation with that number of rounds
    """

    # load spindle initialization data
    spindle_path = os.path.join(dir_path, 'spindle.pkl')
    with open(spindle_path, 'rb') as f:
        spindle_dict = pickle.load(f)

    # load spindle trace
    trace_path = os.path.join(dir_path, 'spindle_trace.pkl')
    with open(trace_path, "rb") as f:
        spindle_trace = pickle.load(f)
    # initialize spindle
    spindle_from_dict = ss.init_spindle_from_dict(spindle_dict)

    # print(spindle_from_dict.tubulin_budget)

    # print(spindle_trace)
    # print('num states: ', len(spindle_trace))

    # iterate through spindle updates in trace until we reconstruct trajectory
    # i = 0
    with Live(console=console, refresh_per_second=4) as live:
        for i in range(spindle_trace.shape[0]):

            if readout:
                # initialize table
                outer_table = Table(title="Restarting Spindle Simulation")
                outer_table.add_column("Parameter", justify="left")
                outer_table.add_column("Value", justify="right")
                
                # set table values
                outer_table.add_row(f'State', f'{i} / {spindle_trace.shape[0]}') # state number
                outer_table.add_row('Current Position', str(spindle_from_dict.mtoc_pos)) # last stable time
                outer_table.add_row('Current Time', str(spindle_from_dict.current_time)) # last stable time
                outer_table.add_row('current_cost', str(spindle_from_dict.calc_cost())) # last accepted cost
                live.update(outer_table)

            # update spindle state
            state = spindle_trace[i, :]
            spindle_from_dict.update_spindle_state(new_spindle_state=state)
            # relax
            meta_traj, meta_boundary_violated = spindle_from_dict.relax(resolution=3, readout=False, live=live)
            
    
            # save
            spindle_from_dict.trajectory = spindle_from_dict.trajectory | meta_traj
            spindle_from_dict.current_time = list(meta_traj.keys())[-1]
            
    spindle_from_dict.spindle_trace = spindle_trace

    return spindle_from_dict





if __name__ == "__main__":

    dir_path = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/testing_reconstruction_2026-02-24_01-09-05'

    spindle_from_dir = restart_experiment_from_directory(dir_path, readout=True)

    print(np.round(spindle_from_dir.mtoc_pos, 3))
    print(np.round(spindle_from_dir.current_time, 3))
    print(np.round(spindle_from_dir.calc_cost(), 3))

    assert (np.round(spindle_from_dir.mtoc_pos, 3) == np.array([-0.026, 0.001, 0.000])).all()
    assert np.round(spindle_from_dir.current_time, 3) == 351.588
    assert np.round(spindle_from_dir.calc_cost(), 3) == 0.040