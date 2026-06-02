#
# Emre Alca
# University of Pennsylvania
# Created on Sat Nov 22 2025
# Last Modified: 2026/04/17 11:37:20
#


import numpy as np
np.set_printoptions(formatter={'float': '{:.3f}'.format})
import tqdm
import matplotlib.pyplot as plt

from datetime import datetime

import pickle
import os

import src.animate_spindle as anisp

from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()

def normalize_vecs(vecs):
    """
    Normalizes an array of vectors

    Args:
        vecs (numpy.array): array of vectors to normalize

    Returns:
        numpy.array: normalized vecs
    """

    if vecs.shape == (3,):
        norm = np.linalg.norm(vecs)
        saved_norm = norm.copy()
        if norm == 0:
            norm = 1
        return vecs / norm, saved_norm
    else:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)

        saved_norms = norms.copy().flatten()

        norms[norms == 0] = 1  # Avoid division by zero
        return vecs / norms, saved_norms
    

def init_spindle_from_dict(spindle_dict):
    """initializes a spindle from the dictionary form of a spindle which we save.

    Args:
        spindle_dict (dict): dict containing initialization values of a spindle

    Returns:
        Spindle: spindle object with the parameters in spindle_dict
    """

    spindle_from_dict = Spindle(
        initial_mtoc_pos=spindle_dict['mtoc_pos'],
        initial_spindle_state=spindle_dict['spindle_state'],
        lattice_sites=spindle_dict['lattice_sites'],
        f_pull_0=spindle_dict['f_pull_0'],
        rigidity=spindle_dict['rigidity'],
        friction_coefficient=spindle_dict['friction_coefficient'],
        growth_rate=spindle_dict['growth_rate'],
        stall_force=spindle_dict['stall_force'],
        drag_factor=spindle_dict['drag_factor'],
        boundary_radius=spindle_dict['boundary_radius'],
        timestep_size=spindle_dict['timestep_size'],
        max_relax_time=spindle_dict['max_relax_time'],
        tubulin_budget=spindle_dict['tubulin_budget'],
        dir_path=spindle_dict['dir_path'],
        seed=spindle_dict['seed'],
        save=True,
        readout=True,
        
    )
    return spindle_from_dict


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

    # load total attemtps
    total_attempts_path = os.path.join(dir_path, 'total_attempts.pkl')
    with open(total_attempts_path, "rb") as f:
        total_attempts = pickle.load(f)

    # initialize spindle
    spindle_from_dict = init_spindle_from_dict(spindle_dict)

    # iterate through spindle updates in trace until we reconstruct trajectory
    # i = 0
    with Live(console=console, refresh_per_second=4) as live:
        for i in range(spindle_trace.shape[0]):

            # update spindle state
            state = spindle_trace[i, :]
            spindle_from_dict.update_spindle_state(new_spindle_state=state)
            spindle_from_dict.total_attempts = total_attempts
            # relax
            # meta_traj, meta_boundary_violated = spindle_from_dict.relax(resolution=3, readout=False, live=live)
            meta_traj, meta_boundary_violated = spindle_from_dict.fast_relax(resolution=3, readout=False, live=live)
            
    
            # save
            spindle_from_dict.trajectory = spindle_from_dict.trajectory | meta_traj
            spindle_from_dict.current_time = list(meta_traj.keys())[-1]

            if readout:
                # initialize table
                outer_table = Table(title="Restarting Spindle Simulation")
                outer_table.add_column("Parameter", justify="left")
                outer_table.add_column("Value", justify="right")
                
                # set table values
                outer_table.add_row(f'State', f'{i} / {spindle_trace.shape[0]}') # state number
                outer_table.add_row('Current Time', str(spindle_from_dict.current_time)) # last stable time
                outer_table.add_row('current_cost', str(spindle_from_dict.calc_cost())) # last accepted cost
                outer_table.add_row('Current Position', str(spindle_from_dict.mtoc_pos)) # last stable time
                outer_table.add_row('number of MTs', str(np.sum(spindle_from_dict.spindle_state - spindle_from_dict.empty_spindle_state))) # number of MTs
                outer_table.add_row('tubulin use / tubulin budget', f'{np.sum(spindle_from_dict.mt_norms[spindle_from_dict.spindle_state==2]) + np.sum(spindle_from_dict.mt_norms[spindle_from_dict.spindle_state==4])} / {spindle_from_dict.tubulin_budget}') # tubulin use / tubulin budget
                live.update(outer_table)
            
    spindle_from_dict.spindle_trace = spindle_trace

    return spindle_from_dict


class Spindle:
    """ 
    Class representing a spindle containing a single microtubule organizing centrosome (MTOC).
    Spindle is responsible manages the manipulation of the spindle state (addition and removal of microtubules) 
    and the calculation of forces exerted by microtubules with a given spindle state and MTOC position

    Attributes:
        mtoc_pos (numpy.array): mtoc position in (x,y,z).
        spindle_state: vector of the states of lattice sites: 1 if empty, 2 if pushing, 3 if empty motor, 4 if pulling
        lattice: positions of the lattice sites in (x,y,z) coordinates.
    """


    def __init__(
            self, 
            initial_mtoc_pos, 
            initial_spindle_state,  
            lattice_sites,
            initial_time=0,
            # -- hyperparameters --
            acceptable_radius=0.05,
            timestep_size=0.001,
            max_relax_time=np.inf,
            # -- biophysical constants
            f_pull_0=1,
            rigidity=1, 
            friction_coefficient=1, 
            growth_rate=1, 
            stall_force=1,
            drag_factor=100,
            boundary_radius=1,
            tubulin_budget=4,
            # -- saving info --
            dir_prefix = '',
            seed=None,
            dir_path = None,
            save=False,
            readout=False,
            ):
        """
        initializes a Spindle with a single centrosome

        Args:
            initial_mtoc_pos (nunpy.array): position of the centrosome in (x,y,z) coordinates.
            initial_spindle_state (numpy.array): initial spindle state shape, contains N elements for N sites.
            lattice_sites (numpy.array): array of the coordinates of the sites, shape is N x 3 for N sites.
        hyperparameters:
            rigidity (float, optional): rigidity coefficient of a single MT. Defaults to 1.
            friction_coefficient (float, optional): friction coefficient between the cortex and the MT. Defaults to 1.
            growth_rate (float, optional): velocity of growth of the MT. Defaults to 1.
            stall_force (float, optional): stall force of the MT. Defaults to 1.
            drag_factor (float, optional): the stokes drag factor of the MTOC, this is the force of drag divided by velocity. Defaults to 100.
            boundary_radius (float, optional): the radius of the boundary sphere. Defaults to 1.
            timestep_size (float, optional): the size of a timestep.
            tubulin_budget (float or None, optional): amount of tubulin in the system
        """

        # -- setting parameters --
        
        self.spindle_state = initial_spindle_state
        self.empty_spindle_state = self.extract_empty_spindle_state()
        
        self.lattice_sites = lattice_sites

        self.boundary_unit_normals = normalize_vecs(lattice_sites)[0]

        self.set_mtoc_pos(initial_mtoc_pos)

        self.num_sites = len(self.spindle_state)
        # -- setting hyperparameters --
        self.acceptable_radius = acceptable_radius
        self.max_relax_time = max_relax_time
        self.f_pull_0 = f_pull_0
        self.rigidity = rigidity 
        self.friction_coefficient = friction_coefficient
        self.growth_rate = growth_rate
        self.stall_force = stall_force
        self.drag_factor = drag_factor
        self.boundary_radius = boundary_radius
        self.timestep_size = timestep_size

        self.total_attempts = 0

        if tubulin_budget is None:
            tubulin_budget = boundary_radius * len(self.spindle_state)

        self.tubulin_budget = tubulin_budget

        # -- setting rng seed --
        self.seed = seed
        self.rng =  np.random.default_rng(seed)

        # -- saving basic info --
        self.readout = readout
        self.save = save

        self.dir_prefix = dir_prefix

        # making dir for Spindle

        if self.save:
            if dir_path is not None:
                self.dir_path = dir_path
            else:
                self.dir_path = self.make_spindle_dir()

            # saving spindle initialization
            spindle_path = os.path.join(self.dir_path, 'spindle.pkl')
            with open(spindle_path, "wb") as f:
                pickle.dump(self.as_dict(), f)

        # -- trajectory initialization -- 
        self.current_time = initial_time
        self.trajectory = {}

        self.spindle_trace = np.array([self.spindle_state])
        # self.update_spindle_trace()


    def make_spindle_dir(self):
        """
        Makes a directory to store the data and animations from experiments with a Spindle object.
        
        Returns:
            str: string for path to the directory for this spindle
        """
        # finding data directory
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), "..")) # find directory which is parent to 'here'
        target_child_dir = os.path.join(parent_dir, "data")
        os.makedirs(target_child_dir, exist_ok=True) # raises error if data directory does not exist

        # make new directory for this experiment
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dirname = f'{self.dir_prefix}_{timestamp}'
        dir_path = os.path.join(target_child_dir, dirname)
        os.makedirs(dir_path, exist_ok=True)
        # print(f'dir made at {dir_path}')
        return dir_path
        

    def as_dict(self):
        """
        creates a dictionary of the parameters and hyperparameters of a spindle and returns the dictionary

        Returns:
            dict: dictionary of all parameters and hyperparameters in the spindle.
        """

        spindle_dict = {}
        # -- parameters --
        spindle_dict['mtoc_pos'] = self.mtoc_pos
        spindle_dict['spindle_state'] = self.empty_spindle_state
        spindle_dict['lattice_sites'] = self.lattice_sites

        # -- hyperparameters --
        spindle_dict['timestep_size'] = self.timestep_size
        spindle_dict['max_relax_time'] = self.max_relax_time
        spindle_dict['acceptable_radius'] = self.acceptable_radius
        spindle_dict['f_pull_0'] = self.f_pull_0
        spindle_dict['rigidity'] = self.rigidity
        spindle_dict['friction_coefficient'] = self.friction_coefficient
        spindle_dict['growth_rate'] = self.growth_rate
        spindle_dict['stall_force'] = self.stall_force
        spindle_dict['drag_factor'] = self.drag_factor
        spindle_dict['boundary_radius'] = self.boundary_radius
        spindle_dict['tubulin_budget'] = self.tubulin_budget
        spindle_dict['seed'] = self.seed

        if self.save:
            spindle_dict['dir_path'] = self.dir_path

        return spindle_dict
    
    def extract_empty_spindle_state(self):
        empty_spindle_state = self.spindle_state.copy()

        empty_spindle_state[empty_spindle_state==2] = 1
        empty_spindle_state[empty_spindle_state==4] = 3

        return empty_spindle_state
    
    
    def update_spindle_trace(self):

        # print(self.spindle_state)

        self.spindle_trace = np.vstack((self.spindle_trace, self.spindle_state.astype(int)))

        if self.save:
            spindle_trace_path = os.path.join(self.dir_path, 'spindle_trace.pkl')
            with open(spindle_trace_path, "wb") as f:
                pickle.dump(self.spindle_trace, f)



    def set_mtoc_pos(self, new_mtoc_pos):
        """ 
        Sets the mtoc to be at a particular position.
        Also updates the mt vectors and directions to consider this new mtoc position.

        Args:
            new_mtoc_pos (numpy.array): new mtoc position in form numpy.array([x, y, z])
        """

        self.mtoc_pos = new_mtoc_pos # set new mtoc position 

        self.mt_vecs = (self.lattice_sites - self.mtoc_pos) # update mt vectors and directions
        self.mt_dirs, self.mt_norms = normalize_vecs(self.mt_vecs) 


    def add_microtubules(self, mt_indices_to_add):
        """
        Adds a microtubule to each site at the i-th position for each index i in mt_indices_to_add.

        Args:
            mt_indices_to_add (numpy.array): list containing the indices of microtubules to add

        Raises:
            ValueError: cannot add a microtubule to a site already containing a microtubule
        """

        if len(mt_indices_to_add) == 0:
            return None

        if len(np.where(self.spindle_state[mt_indices_to_add] == 2)[0]) != 0 or len(np.where(self.spindle_state[mt_indices_to_add] == 4)[0]) != 0:
            raise ValueError("cannot add a microtubule to a site already containing a microtubule")
        
        update = np.zeros(self.num_sites)

        update[mt_indices_to_add] = 1
        
        self.spindle_state = self.spindle_state + update


    def remove_microtubules(self, mt_indices_to_remove):
        """
        Removes a microtubule to each site at the i-th position for each index i in mt_indices_to_remove.

        Args:
            mt_indices_to_remove (numpy.array): list containing the indices of microtubules to remove

        Raises:
            ValueError: cannot remove a microtubule from a site which does not contain a microtubule
        """
        if len(mt_indices_to_remove) == 0:
            return None

        if len(np.where(self.spindle_state[mt_indices_to_remove] == 1)[0]) != 0 or len(np.where(self.spindle_state[mt_indices_to_remove] == 3)[0]) != 0:
            raise ValueError("cannot remove a microtubule from a site which does not contain a microtubule")
        
        update = np.zeros(self.num_sites)

        update[mt_indices_to_remove] = -1
        
        self.spindle_state = self.spindle_state + update

    def update_spindle_state(self, mts_to_remove=None, mts_to_add=None, new_spindle_state=None):
        if mts_to_remove is None and mts_to_add is None and new_spindle_state is None:
            raise ValueError("mts_to_remove, mts_to_add, new_spindle_state cannot all be None")
        if new_spindle_state is not None and (mts_to_add is not None or mts_to_remove is not None):
            raise ValueError('specify the new spindle state OR MT-wise changes, not both ')
        
        if new_spindle_state is not None: # update spindle state directly
            self.spindle_state = new_spindle_state
        elif mts_to_remove is not None: # remove MTs
            self.remove_microtubules(mts_to_remove)
        elif mts_to_add is not None: # add MTs
            self.add_microtubules(mts_to_add)

        
    def calculate_pulling_forces(self):
        """ 
        Calculates the pulling force experienced by the mtoc.
        For the moment, we assume the pulling force to be constant

        Args:
            f_pull_0 (int, optional): magnitude of pulling force.. Defaults to 1.

        Returns:
            numpy.array: vector with the direction and magnitude of the sum of all pulling MTs.
        """
        return self.f_pull_0 * np.sum(self.mt_dirs[self.spindle_state == 4], axis=0) # f_minus_0 times the sum of the pulling mhat vectors
    
    def calculate_pushing_forces(self):
        """
        Calculates the total pushing force exerted all the pushing MTs on the MTOC.
        The force is bounded above the by the buckling force.
        The force is modulated effective force coefficient: 
        The more perpendicular the MT relative to the cortex, the harder it pushes.
        The total pushing force is the sum over each pushing MT's pushing force magnitude and direction.

        Returns:
            numpy.array: sum of the forces of all pushing MTs as in the total pushing force vector
        """
        
        # -- calculating buckling forces -- 

        buckling_forces = (np.pi**2) * self.rigidity / (self.mt_norms[self.spindle_state == 2]**2)
        # print(self.spindle_state==2)
        # print(self.mt_norms[self.spindle_state == 2])
        # print(buckling_forces)

        # -- calculating unbuckled pushing forces --

        # calculating the effective force coefficients (mt_dir . boundary_norm)
        pushing_mt_dirs = self.mt_dirs[self.spindle_state == 2]
        # print(pushing_mt_dirs)
        pushing_boundary_normals = self.boundary_unit_normals[self.spindle_state == 2]
        effective_force_coefficients = np.sum(pushing_mt_dirs * pushing_boundary_normals, axis=1)
        # effective_force_coefficients = np.dot(pushing_boundary_normals, pushing_mt_dirs)

        # calculating the denominator of the pushing force magnitude
        pushing_force_denominators = (self.stall_force / (self.growth_rate * self.friction_coefficient)) * (1 - effective_force_coefficients) + 1

        # putting the pieces together
        pushing_force_magnitudes = self.stall_force / pushing_force_denominators

        # -- calculating pushing force vectors --

        # pushing forces are bounded above by the buckling force
        # print(f'is buckling: \n {pushing_force_magnitudes > buckling_forces}')
        pushing_force_magnitudes[pushing_force_magnitudes > buckling_forces] = buckling_forces[pushing_force_magnitudes > buckling_forces]

        # total pushing force is the component-wise sum of the pushing vectors
        pushing_vectors = pushing_force_magnitudes[:, np.newaxis] * pushing_mt_dirs
        # pushing_vectors = buckling_forces[:, np.newaxis] * pushing_mt_dirs
        # print(-pushing_vectors)

        # -- summing over pushing force vectors to find total pushing force --

        # pushing_mt_dirs point outwards from the mtoc, we want pushing forces to point inwards towards the mtoc
        total_pushing_force = -np.sum(pushing_vectors, axis=0)
        # print(f'pushing vectors: \n {pushing_vectors}')

        return total_pushing_force
    
    
    def calc_mtoc_velocity(self):
        """Calculates the velocity of the mtoc based on the mtoc position and the set of pushing and pulling mts
        This is the implementation of the mtoc's equation of motion.

        Returns:
            numpy.array: velocity vector in the form of numpy.array([x,y,z])
        """
        return (self.calculate_pulling_forces() + self.calculate_pushing_forces()) / self.drag_factor
    

    def mtoc_time_evolution(self):
        """
        Evolves the MTOC position by the MTOC velocity times the size of a timestep.
        This function returns both the new MTOC position, and a bool representing whether the boundary has been violated.
        If the boundary is violated, the unit direction of the current position is taken, and multiplied by the boundary's radius.
        Finally, the MTOC's position is moved according to the equations of motion.

        Returns:
            numpy.array: the new MTOC position after one timestep in the form of numpy.array([x,y,z]),
            bool:  whether the boundary has been violated
        """
        # calculate the new position of the MTOC
        velocity = self.calc_mtoc_velocity()
        new_mtoc_pos = self.mtoc_pos + (velocity * self.timestep_size)

        # check that the new mtoc position is not outside of the radius
        normalized_new_mtoc_pos, new_mtoc_pos_norm = normalize_vecs(new_mtoc_pos)

        boundary_violated = False
        if new_mtoc_pos_norm > self.boundary_radius:
            # if the new MTOC position is outside of the radius, place it on the radius, pointing in the same direction
            new_mtoc_pos = normalized_new_mtoc_pos * self.boundary_radius
            boundary_violated = True

        self.set_mtoc_pos(new_mtoc_pos)
        self.current_time += self.timestep_size
        return new_mtoc_pos, boundary_violated
    

    def calc_cost(self):
        """
        Calculates the cost of the current position and spindle state.
        C = |r|^2 + (L_tot - L_max)^2

        Returns:
            float: cost = displacement cost + length cost.
        """
        # -- displacement cost (|r|/R)^2 --
        # calculate displacement
        mtoc_pos_norm = normalize_vecs(self.mtoc_pos)[1]
        # calculate positioning cost
        displacement_cost = np.square(mtoc_pos_norm/self.boundary_radius)

        # allowing small deviations from the positional set point
        if displacement_cost < np.square(self.acceptable_radius):
            displacement_cost = 0

        # -- material cost (1 - (L_tot / L_max)^2 --
        # calculate total length of all MTs
        total_mt_length = np.sum(self.mt_norms[self.spindle_state==2]) + np.sum(self.mt_norms[self.spindle_state==4])

        # calculate material cost
        material_cost = np.square(1 - (total_mt_length / self.tubulin_budget))

        # Allowing small deviations around the target tubulin use
        if material_cost < np.square(self.acceptable_radius):
            material_cost = 0
        
        return displacement_cost + material_cost
    
    def spindle_update_single_mt(self):
        """updates the spindle by removing a single MT and placing it somewhere else.
        Nucleation and catastrophe follow a uniform spacial distribution

        Returns:
            int, int: index of the mt to remove, index of the mt to add
        """
        # select empty and filled indices 
        empty_indices = np.append(np.where(self.spindle_state == 1), np.where(self.spindle_state==3))
        filled_indices = np.append(np.where(self.spindle_state == 2), np.where(self.spindle_state==4))

        # choose at random which single MT to remove and which MT to place
        mt_to_remove = filled_indices[np.random.randint(0, len(filled_indices))]
        mt_to_add = empty_indices[np.random.randint(0, len(empty_indices))]

        self.add_microtubules(np.array([mt_to_add]))
        self.remove_microtubules(np.array([mt_to_remove]))

        return mt_to_remove, mt_to_add
    

    def spindle_update_uniform(self):

        # -- catastrophe --
        # select filled sites
        filled_indices = np.append(np.where(self.spindle_state == 2), np.where(self.spindle_state==4))
        # find the number of filled sites (MTs present)
        num_filled_sites = len(filled_indices)
        # print(f'num_filled_sites: {num_filled_sites}')
        # uniformly choose a number of filled sites to empty
        num_to_remove = self.rng.integers(0, num_filled_sites+1)
        # print(f'num_to_remove: {num_to_remove}')
        # uniformly choose which MTs to remove
        filled_mts_to_remove = self.rng.permutation(num_filled_sites)[:num_to_remove]
        sites_to_empty = filled_indices[filled_mts_to_remove]

        # -- nucleation --
        # select empty sites
        empty_indices = np.append(np.where(self.spindle_state == 1), np.where(self.spindle_state==3))
        # find the number of empty sites
        num_empty_sites = len(empty_indices)
        # print(f'num_empty_sites: {num_empty_sites}')
        # uniformly choose a number of empty sites to fill
        num_to_add = self.rng.integers(0, num_empty_sites+1)
        # print(f'num_to_add: {num_to_add}')
        # uniformly choose which sites to fill
        empty_sites_to_fill = self.rng.permutation(num_empty_sites)[:num_to_add]
        sites_to_fill = empty_indices[empty_sites_to_fill]

        # -- execute update --
        self.remove_microtubules(sites_to_empty)
        # print(f'sites_to_empty: {sites_to_empty}')
        self.add_microtubules(sites_to_fill)
        # print(f'sites_to_fill: {sites_to_fill}')

        return sites_to_empty, sites_to_fill
    
    def spindle_update_uniform_poisson(self, mean_nuc=1, mean_cat=1):
        # -- catastrophe --
        # select filled sites
        filled_indices = np.append(np.where(self.spindle_state == 2), np.where(self.spindle_state==4))
        # find the number of filled sites (MTs present)
        num_filled_sites = len(filled_indices)

        # sample the number of filled sites to empty from poisson
        num_to_remove = np.random.poisson(mean_cat)
        # uniformly choose which MTs to remove
        filled_mts_to_remove = self.rng.permutation(num_filled_sites)[:num_to_remove]
        sites_to_empty = filled_indices[filled_mts_to_remove]

        # -- nucleation --
        # select empty sites
        empty_indices = np.append(np.where(self.spindle_state == 1), np.where(self.spindle_state==3))
        # find the number of empty sites
        num_empty_sites = len(empty_indices)
        # print(f'num_empty_sites: {num_empty_sites}')
        # uniformly choose a number of empty sites to fill
        num_to_add = np.random.poisson(mean_nuc)
        # print(f'num_to_add: {num_to_add}')
        # uniformly choose which sites to fill
        empty_sites_to_fill = self.rng.permutation(num_empty_sites)[:num_to_add]
        sites_to_fill = empty_indices[empty_sites_to_fill]

        # -- execute update --
        self.remove_microtubules(sites_to_empty)
        # print(f'sites_to_empty: {sites_to_empty}')
        self.add_microtubules(sites_to_fill)
        # print(f'sites_to_fill: {sites_to_fill}')

        return sites_to_empty, sites_to_fill

    
    def relax(self, resolution=5, readout=False, live=None, attempt_number=None):
        meta_t = float(self.current_time)
        # evolve time until either the boundary is violated or the system is stable
        dr_dt_norm = 1
        boundary_violated = False
        trajectory = {}
        timepoint_data = {
            'spindle_state': self.spindle_state.astype(int),
            'mtoc_pos': self.mtoc_pos,
            'boundary_violated': boundary_violated,
            'cost': self.calc_cost(),
            'tubulin_use': np.sum(self.mt_norms[self.spindle_state==2]) + np.sum(self.mt_norms[self.spindle_state==4]),
            'num_mts': len(np.append(np.where(self.spindle_state == 2), np.where(self.spindle_state==4))),
        }
        trajectory[meta_t] = timepoint_data

        # loop until the MTOC is pseudo-stationary or has violated the boundary
        while np.round(dr_dt_norm, resolution) > 0 and not boundary_violated:
            # calculate velocity
            dr_dt = self.calc_mtoc_velocity() 
            # calculate speed for comparison and pseudo-stationary condition
            dr_dt_norm = normalize_vecs(dr_dt)[1]
            
            # calculate the new position of the MTOC
            new_mtoc_pos = self.mtoc_pos + (dr_dt * self.timestep_size)

            # check that the new mtoc position is not outside of the radius
            normalized_new_mtoc_pos, new_mtoc_pos_norm = normalize_vecs(new_mtoc_pos)

            boundary_violated = False
            if new_mtoc_pos_norm > self.boundary_radius:
                # if the new MTOC position is outside of the radius, place it on the radius, pointing in the same direction
                new_mtoc_pos = normalized_new_mtoc_pos * self.boundary_radius
                boundary_violated = True

            self.set_mtoc_pos(new_mtoc_pos)
            meta_t += float(self.timestep_size)

            timepoint_data = {
                    'spindle_state': self.spindle_state.astype(int),
                    'mtoc_pos': self.mtoc_pos,
                    'boundary_violated': boundary_violated,
                    'cost': self.calc_cost(),
                    'tubulin_use': np.sum(self.mt_norms[self.spindle_state==2]) + np.sum(self.mt_norms[self.spindle_state==4]),
                    'num_mts': np.sum(self.spindle_state - self.empty_spindle_state),
                }
            trajectory[meta_t] = timepoint_data

            if readout:
                # initialize table 
                metastate_table = Table(title="Metastate Relaxation")
                metastate_table.add_column("Parameter", justify="left")
                metastate_table.add_column("Value", justify="right")
                
                # set values in table
                if attempt_number is not None:
                    metastate_table.add_row("Attempt Number", str(attempt_number)) # attempt number
                metastate_table.add_row("Time", f"{meta_t:.2f}")                          # time
                metastate_table.add_row("Position", str(self.mtoc_pos))      # position
                metastate_table.add_row("Direction of Motion", str(dr_dt/self.timestep_size))      # velocity
                metastate_table.add_row("Boundary Violated", str(boundary_violated)) # boundary violated
                live.update(metastate_table)

        return trajectory, boundary_violated

    def fast_relax(self, resolution=5, readout=False, live=None, attempt_number=None):
        # -- precompute once; spindle state is frozen during relax --
        push_mask = self.spindle_state == 2
        pull_mask = self.spindle_state == 4
        active_lattice_push = self.lattice_sites[push_mask]        # (M_push, 3)
        active_lattice_pull = self.lattice_sites[pull_mask]        # (M_pull, 3)
        push_boundary_normals = self.boundary_unit_normals[push_mask]  # (M_push, 3)
        has_push = len(active_lattice_push) > 0
        has_pull = len(active_lattice_pull) > 0

        spindle_state_int = self.spindle_state.astype(int)   # copy once, reuse every step
        num_mts = int(np.sum(spindle_state_int - self.empty_spindle_state))
        acceptable_radius_sq = self.acceptable_radius ** 2

        max_relax_time = self.max_relax_time

        # local references to avoid repeated attribute lookups in the hot loop
        f_pull_0            = self.f_pull_0
        rigidity            = self.rigidity
        stall_force         = self.stall_force
        growth_rate         = self.growth_rate
        friction_coefficient = self.friction_coefficient
        drag_factor         = self.drag_factor
        boundary_radius     = self.boundary_radius
        timestep_size       = self.timestep_size
        tubulin_budget      = self.tubulin_budget
        sf_over_gf          = stall_force / (growth_rate * friction_coefficient)  # constant scalar

        # -- working position (local, avoid touching self until the end) --
        mtoc_pos = self.mtoc_pos.copy()
        initial_time = float(self.current_time)
        boundary_violated = False
        trajectory = {}

        # -- initialize active-MT vecs/norms at current position --
        if has_push:
            push_vecs  = active_lattice_push - mtoc_pos
            push_norms = np.linalg.norm(push_vecs, axis=1)   # (M_push,)
        if has_pull:
            pull_vecs  = active_lattice_pull - mtoc_pos
            pull_norms = np.linalg.norm(pull_vecs, axis=1)   # (M_pull,)

        def _cost_and_tubulin(pos_norm):
            total_mt_length = (np.sum(push_norms) if has_push else 0.0) + \
                              (np.sum(pull_norms) if has_pull else 0.0)
            disp = (pos_norm / boundary_radius) ** 2
            if disp < acceptable_radius_sq:
                disp = 0.0
            mat = (1.0 - total_mt_length / tubulin_budget) ** 2
            if mat < acceptable_radius_sq:
                mat = 0.0
            return disp + mat, total_mt_length

        mtoc_norm = np.linalg.norm(mtoc_pos)
        cost, tubulin_use = _cost_and_tubulin(mtoc_norm)
        trajectory[initial_time] = {
            'spindle_state':    spindle_state_int,
            'mtoc_pos':         mtoc_pos,
            'boundary_violated': boundary_violated,
            'cost':             cost,
            'tubulin_use':      tubulin_use,
            'num_mts':          num_mts,
        }

        dr_dt_norm = 1.0
        step = 0

        while (np.round(dr_dt_norm, resolution) > 0 and not boundary_violated) and (step * timestep_size < max_relax_time):
            # -- pulling force (O(M_pull)) --
            if has_pull:
                pull_dirs = pull_vecs / pull_norms[:, np.newaxis]
                pull_force = f_pull_0 * np.sum(pull_dirs, axis=0)
            else:
                pull_force = np.zeros(3)

            # -- pushing force (O(M_push)) --
            if has_push:
                push_dirs       = push_vecs / push_norms[:, np.newaxis]
                buckling        = (np.pi ** 2) * rigidity / (push_norms ** 2)
                eff_coeff       = np.sum(push_dirs * push_boundary_normals, axis=1)
                denom           = sf_over_gf * (1.0 - eff_coeff) + 1.0
                magnitudes      = np.minimum(stall_force / denom, buckling)
                push_force      = -np.sum(magnitudes[:, np.newaxis] * push_dirs, axis=0)
            else:
                push_force = np.zeros(3)

            dr_dt      = (pull_force + push_force) / drag_factor
            dr_dt_norm = np.linalg.norm(dr_dt)

            # -- update position --
            new_pos      = mtoc_pos + dr_dt * timestep_size
            new_pos_norm = np.linalg.norm(new_pos)

            boundary_violated = False
            if new_pos_norm > boundary_radius:
                new_pos      = (new_pos / new_pos_norm) * boundary_radius
                new_pos_norm = boundary_radius
                boundary_violated = True

            mtoc_pos = new_pos
            step += 1
            meta_t = initial_time + step * timestep_size

            # -- update active-MT vecs/norms for next iteration (and cost below) --
            if has_push:
                push_vecs  = active_lattice_push - mtoc_pos
                push_norms = np.linalg.norm(push_vecs, axis=1)
            if has_pull:
                pull_vecs  = active_lattice_pull - mtoc_pos
                pull_norms = np.linalg.norm(pull_vecs, axis=1)

            cost, tubulin_use = _cost_and_tubulin(new_pos_norm)
            trajectory[meta_t] = {
                'spindle_state':    spindle_state_int,
                'mtoc_pos':         mtoc_pos,
                'boundary_violated': boundary_violated,
                'cost':             cost,
                'tubulin_use':      tubulin_use,
                'num_mts':          num_mts,
            }

            if readout and live is not None:
                metastate_table = Table(title="Metastate Relaxation")
                metastate_table.add_column("Parameter", justify="left")
                metastate_table.add_column("Value", justify="right")
                if attempt_number is not None:
                    metastate_table.add_row("Attempt Number", str(attempt_number))
                metastate_table.add_row("Time", f"{meta_t:.2f}")
                metastate_table.add_row("Position", str(mtoc_pos))
                metastate_table.add_row("Direction of Motion", str(dr_dt / timestep_size))
                metastate_table.add_row("Boundary Violated", str(boundary_violated))
                live.update(metastate_table)

        # -- commit final position to self (one set_mtoc_pos call total) --
        self.set_mtoc_pos(mtoc_pos)
        self.current_time = initial_time + step * timestep_size
        return trajectory, boundary_violated

    def spindle_optimization_uniform(self, num_total_attempts, mean_nuc=1, mean_cat=1,  initial_cost=None, resolution=5, readout=True):
        """simulates spindle evolution where the spindle state changes by a random number of nucleations
        and catastrophes each time it is updated.
        The mechanics of pushing and pulling MTs come to equilibrium before the spindle is updated.

        Args:
            num__positions (int): number of stable positions before ending simulations
        """
        if initial_cost is None: 
            initial_cost = len(self.spindle_state) + 1 # setting initial cost to be very high

        old_mtoc_pos = np.copy(self.mtoc_pos) # initial original state is the current state
        old_cost = initial_cost # any stable position is an improvement
        old_current_time = self.current_time
        total_attempts = 0
        

        with Live(console=console, refresh_per_second=4) as live:

            # --- update spindle -> relax loop ---

            # for i in range(num_positions):
            while total_attempts <= num_total_attempts:

                # -- middle loop (update spindle, try to relax, if cost improves, accept, if not, reject) -- 
                improvement = False
                attempt_counter = 0
                
                while not improvement:

                    # readout
                    if readout:
                        # initialize table
                        outer_table = Table(title="Spindle Simulation")
                        outer_table.add_column("Parameter", justify="left")
                        outer_table.add_column("Value", justify="right")
                        
                        # set table values
                        outer_table.add_row('Last Accepted Time', str(old_current_time)) # last stable time
                        outer_table.add_row('Last Accepted Position', str(old_mtoc_pos)) # last stable position
                        outer_table.add_row('Last Accepted Cost', str(old_cost)) # last accepted cost
                        outer_table.add_row('number of MTs', str(np.sum(self.spindle_state - self.empty_spindle_state))) # number of MTs
                        outer_table.add_row('tubulin use / tubulin budget', f'{np.round(np.sum(self.mt_norms[self.spindle_state==2]) + np.sum(self.mt_norms[self.spindle_state==4]), 3)} / {self.tubulin_budget}') # tubulin use / tubulin budget
                        outer_table.add_row('Attempt Counter', str(attempt_counter)) # attempt counter
                        outer_table.add_row('Number of Positions Accepted', str(self.spindle_trace.shape[0])) # number of accepted positions
                        outer_table.add_row('Number of total attempts', str(self.total_attempts))
                        live.update(outer_table)

                    # update metastate spindle
                    # --- edit here to switch to single MT update ---
                    # sites_to_empty, sites_to_fill = self.spindle_update_single_mt() 
                    # sites_to_empty, sites_to_fill = self.spindle_update_uniform()
                    sites_to_empty, sites_to_fill = self.spindle_update_uniform_poisson(mean_nuc=mean_nuc, mean_cat=mean_cat)
                    
                    attempt_counter += 1
                    self.total_attempts +=1
                    # print(f'updated spindle state: {self.spindle_state}')
                    # print(f'attempt counter: {attempt_counter}')

                    # -- inner loop relax metastate --
                    # meta_traj, meta_boundary_violated = self.relax(resolution=resolution, readout=False,live=live, attempt_number=attempt_counter)
                    meta_traj, meta_boundary_violated = self.fast_relax(resolution=resolution, readout=False,live=live, attempt_number=attempt_counter)

                    # compare meta_cost to old_cost
                    meta_cost = self.calc_cost()
                    meta_current_time = self.current_time

                    if meta_cost > old_cost or meta_boundary_violated:
                        # the cost of the metastate is greater than the cost of the old state or the boundary has been violated,reject this change and try again
                        self.add_microtubules(np.array([sites_to_empty]))
                        self.remove_microtubules(np.array([sites_to_fill]))
                        self.set_mtoc_pos(old_mtoc_pos)
                        self.current_time = old_current_time
                    else:
                        # implies meta_cost <= old_cost and meta_boundary_violated == False, so we accept the change
                        # fold in trajectory data 
                        self.trajectory = self.trajectory | meta_traj
                        self.current_time = list(meta_traj.keys())[-1]
                        improvement = True
                        self.update_spindle_trace()

                # save current position and spindle state
                old_mtoc_pos = np.copy(self.mtoc_pos)
                old_cost = np.copy(self.calc_cost())
                old_current_time = np.copy(self.current_time)
                # self.total_attempts += attempt_counter

                if self.save:
                    total_attempts_path = os.path.join(self.dir_path, 'total_attempts.pkl')
                    with open(total_attempts_path, "wb") as f:
                        pickle.dump(self.total_attempts, f)
        
        return total_attempts
    
    def animate(self, interval=100, save=False, file_prefix=''):
        """makes an animation of the spindle trajectory.

        Args:
            interval (int, optional): number of frames skipped. Defaults to 100.
            save (bool, optional): True to save. False not to. Defaults to False.
            file_prefix (str, optional): file prefix. Defaults to ''.

        Returns:
            matplotlib.animation: matplotlib animation object of trajectory
        """
        data = {}
        data['spindle'] = self.as_dict()
        data['trajectory'] = self.trajectory

        

        ani = anisp.animate_spindle(data, interval=interval,)

        if save:
            os.makedirs(self.dir_path, exist_ok=True)

            # writing path to save file
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if file_prefix == '':
                file_prefix = 'spindle_animation'
            filename = f"{file_prefix}_{timestamp}.mp4"

            file_path = os.path.join(self.dir_path, filename)
            ani.save(file_path, fps=60, dpi=150)
            print(f'animation saved to {file_path}')

        return ani
    
    def save_trajectory(self, file_prefix=''):
        """saves the spindle.as_dict() information and the trajectory data

        Args:
            file_prefix (str, optional): prefix for file name. Defaults to ''.
        """

        data = {}
        data['spindle'] = self.as_dict()
        data['trajectory'] = self.trajectory

        target_child_dir = self.dir_path

        # writing path to save file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{file_prefix}_{timestamp}.pkl"
        file_path = os.path.join(target_child_dir, filename)

        # saving file
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

        print(f'trajectory data saved to {file_path}')
        
