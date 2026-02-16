#
# Emre Alca
# University of Pennsylvania
# Created on Sat Nov 22 2025
# Last Modified: 2026/02/16 11:53:35
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
            # -- hyperparameters --
            timestep_size=0.1,
            # -- biophysical constants
            nucleation_rate=10,
            catastrophe_rate=0.1,
            f_pull_0=1,
            rigidity=1, 
            friction_coefficient=1, 
            growth_rate=1, 
            stall_force=1,
            drag_factor=100,
            cytoplasmic_catastrophe_rate=1,
            boundary_radius=1,
            tubulin_budget=4,
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

        # setting parameters
        if not np.isin(initial_spindle_state, [1,3]).all():
            raise ValueError("initial spindle state must be empty!")
        
        self.empty_spindle_state = np.copy(initial_spindle_state)
        self.spindle_state = initial_spindle_state
        
        self.lattice_sites = lattice_sites

        self.boundary_unit_normals = normalize_vecs(lattice_sites)[0]

        self.set_mtoc_pos(initial_mtoc_pos)

        self.num_sites = len(self.spindle_state)

        # trajectory saving parameters
        self.current_time = 0
        self.trajectory = {}

        # setting hyperparameters
        self.nucleation_rate = nucleation_rate
        self.catastrophe_rate = catastrophe_rate
        self.f_pull_0 = f_pull_0
        self.rigidity = rigidity 
        self.friction_coefficient = friction_coefficient
        self.growth_rate = growth_rate
        self.stall_force = stall_force
        self.drag_factor = drag_factor
        self.boundary_radius = boundary_radius
        self.timestep_size = timestep_size

        if tubulin_budget is None:
            tubulin_budget = boundary_radius * len(self.spindle_state)

        self.tubulin_budget = tubulin_budget
        self.cytoplasmic_catastrophe_rate = cytoplasmic_catastrophe_rate

        self.max_total_mt_length = len(self.spindle_state) * boundary_radius
        

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
        spindle_dict['nucleation_rate'] = self.nucleation_rate
        spindle_dict['catastrophe_rate'] = self.catastrophe_rate
        spindle_dict['f_pull_0'] = self.f_pull_0
        spindle_dict['rigidity'] = self.rigidity
        spindle_dict['friction_coefficient'] = self.friction_coefficient
        spindle_dict['growth_rate'] = self.growth_rate
        spindle_dict['stall_force'] = self.stall_force
        spindle_dict['drag_factor'] = self.drag_factor
        spindle_dict['boundary_radius'] = self.boundary_radius
        spindle_dict['timestep_size'] = self.timestep_size
        spindle_dict['tubulin_budget'] = self.tubulin_budget
        spindle_dict['cytoplasmic_catastrophe_rate'] = self.cytoplasmic_catastrophe_rate

        return spindle_dict


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
        # print(buckling_forces)

        # -- calculating unbuckled pushing forces --

        # calculating the effective force coefficients (mt_dir . boundary_norm)
        pushing_mt_dirs = self.mt_dirs[self.spindle_state == 2]
        # print(pushing_mt_dirs)
        pushing_boundary_normals = self.boundary_unit_normals[self.spindle_state == 2]
        effective_force_coefficients = np.sum(pushing_mt_dirs * pushing_boundary_normals, axis=1)

        # calculating the denominator of the pushing force magnitude
        pushing_force_denominators = (self.stall_force / (self.growth_rate * self.friction_coefficient)) * (1 - effective_force_coefficients) + 1

        # putting the pieces together
        pushing_force_magnitudes = self.stall_force / pushing_force_denominators

        # -- calculating pushing force vectors --

        # pushing forces are bounded above by the buckling force
        pushing_force_magnitudes[pushing_force_magnitudes > buckling_forces] = buckling_forces[pushing_force_magnitudes > buckling_forces]

        # total pushing force is the component-wise sum of the pushing vectors
        pushing_vectors = pushing_force_magnitudes[:, np.newaxis] * pushing_mt_dirs
        # pushing_vectors = buckling_forces[:, np.newaxis] * pushing_mt_dirs
        # print(-pushing_vectors)

        # -- summing over pushing force vectors to find total pushing force --

        # pushing_mt_dirs point outwards from the mtoc, we want pushing forces to point inwards towards the mtoc
        total_pushing_force = -np.sum(pushing_vectors, axis=0)
        # print(np.sum(pushing_vectors, axis=0))

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
        # -- displacement cost |r|^2/R --
        displacement_cost = np.square(normalize_vecs(self.mtoc_pos)[1]/self.boundary_radius)

        # -- material cost (L_tot - L_max)^2 --
        total_mt_length = np.sum(self.mt_norms[self.spindle_state==2]) + np.sum(self.mt_norms[self.spindle_state==4])
        material_cost = np.square((total_mt_length - self.tubulin_budget)/(self.max_total_mt_length - self.tubulin_budget))
        
        return displacement_cost + material_cost
    
    def spindle_update_single_mt(self):

        # select empty and filled indices 
        empty_indices = np.append(np.where(self.spindle_state == 1), np.where(self.spindle_state==3))
        filled_indices = np.append(np.where(self.spindle_state == 2), np.where(self.spindle_state==4))
        # print(f'empty sites: {empty_indices}')
        # print(f'filled sites {filled_indices}') 

        # choose at random which single MT to remove and which MT to place
        mt_to_remove = filled_indices[np.random.randint(0, len(filled_indices))]
        mt_to_add = empty_indices[np.random.randint(0, len(empty_indices))]

        # print(f'mt to remove: {mt_to_remove}')
        # print(f'mt to add: {mt_to_add}')

        self.add_microtubules(np.array([mt_to_add]))
        self.remove_microtubules(np.array([mt_to_remove]))

        return mt_to_remove, mt_to_add

    
    def relax(self, resolution=5, readout=True, live=None, attempt_number=None):
        meta_t = float(self.current_time)
        # print(type(meta_t))
        # evolve time until either the boundary is violated or the system is stable
        dr_dt_norm = 1
        boundary_violated = False
        trajectory = {}
        timepoint_data = {
            'spindle_state': self.spindle_state.astype(int),
            'mtoc_pos': self.mtoc_pos,
            'boundary_violated': boundary_violated,
            'cost': self.calc_cost(),
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
    
    def single_mt_update_simulation_to_equilibrium(self, num_positions, resolution=5, readout=True, save=False):
        """simulates spindle evolution where the spindle state changes by a single MT each time it is updated.
        The mechanics of pushing and pulling MTs come to equilibrium before the spindle is updated.

        Args:
            num__positions (int): number of stable positions before ending simulations
        """

        old_mtoc_pos = np.copy(self.mtoc_pos) # initial original state is the current state
        old_cost = 10 # any stable position is an improvement
        old_current_time = self.current_time
        

        with Live(console=console, refresh_per_second=4) as live:

            # --- update spindle -> relax loop ---

            for i in range(num_positions):

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
                        outer_table.add_row('Attempt Counter', str(attempt_counter)) # attempt counter
                        outer_table.add_row('Number of Positions Accepted', str(i)) # attempt counter
                        live.update(outer_table)

                    # update metastate spindle
                    mt_removed, mt_added = self.spindle_update_single_mt()
                    attempt_counter += 1
                    # print(f'updated spindle state: {self.spindle_state}')
                    # print(f'attempt counter: {attempt_counter}')

                    # -- inner loop relax metastate --
                    meta_traj, meta_boundary_violated = self.relax(resolution=resolution, readout=False,live=live, attempt_number=attempt_counter)

                    # compare meta_cost to old_cost
                    meta_cost = self.calc_cost()
                    meta_current_time = self.current_time

                    if meta_cost > old_cost or meta_boundary_violated:
                        # the cost of the metastate is greater than the cost of the old state or the boundary has been violated,reject this change and try again
                        self.add_microtubules(np.array([mt_removed]))
                        self.remove_microtubules(np.array([mt_added]))
                        self.set_mtoc_pos(old_mtoc_pos)
                        self.current_time = old_current_time
                    else:
                        # implies meta_cost <= old_cost and meta_boundary_violated == False, so we accept the change
                        # fold in trajectory data 
                        self.trajectory = self.trajectory | meta_traj
                        self.current_time = list(meta_traj.keys())[-1]
                        improvement = True

                # save current position and spindle state
                old_mtoc_pos = np.copy(self.mtoc_pos)
                old_cost = np.copy(self.calc_cost())
                old_current_time = np.copy(self.current_time)
    
    def animate(self, interval=100, save=False, file_prefix=None):
        data = {}
        data['spindle'] = self.as_dict()
        data['trajectory'] = self.trajectory

        return anisp.animate_spindle(data, interval=interval, save=save, file_prefix=file_prefix)
    
    def save_trajectory(self, file_prefix=''):
        """saves the spindle.as_dict() information and the trajectory data

        Args:
            file_prefix (str, optional): prefix for file name. Defaults to ''.
        """

        data = {}
        data['spindle'] = self.as_dict()
        data['trajectory'] = self.trajectory
    
        # finding data directory
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        target_child_dir = os.path.join(parent_dir, "data")
        os.makedirs(target_child_dir, exist_ok=True)

        # writing path to save file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{file_prefix}_{timestamp}.pkl"
        file_path = os.path.join(target_child_dir, filename)

        # saving file
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

        print(f'data saved to {file_path}')
        


    # --- -------------------------------------------------------------------------------------------------------------
    

    def gradient_descent_spindle_update(self):

        # save old spindle state, cost, and MTOC position
        old_spindle_state = np.copy(self.spindle_state)
        old_mtoc_pos = np.copy(self.mtoc_pos)
        old_cost = self.calc_cost()


        # --- only accept modifications which reduce the cost ---

        new_cost = np.copy(old_cost) + 1
        new_spindle_state = np.copy(old_spindle_state)

        attempt_counter = 1
        cost_acceptance_resolution = 6

        while (np.round(new_cost, cost_acceptance_resolution) >= np.round(old_cost, cost_acceptance_resolution)): # or (new_spindle_state == old_spindle_state).all():
            # print(f'attempt number: {attempt_counter}')
            # sample random numbers to compare to distributions
            nucleation_random_numbers = np.random.rand(len(self.spindle_state))
            catastrophe_random_numbers = np.random.rand(len(self.spindle_state))

            # -- new uniform catastrophe and nucleation distributions --
            # important_numbers
            num_mts_present = np.sum(self.spindle_state) - np.sum(self.empty_spindle_state)
            num_empty_sites = len(self.spindle_state) - num_mts_present

            # choose which MTs nucleate
            nucleation_probability = (self.nucleation_rate * self.timestep_size) / num_empty_sites
            select_empty_sites_only = np.zeros(len(self.spindle_state)) + (self.spindle_state==3).astype(int) + (self.spindle_state == 1).astype(int)
            nucleation_distribution = nucleation_probability * select_empty_sites_only
            # print(f'nucleation_distribution: {nucleation_distribution}')

            # mt_nucleations = np.where(spindle_update_random_numbers < nucleation_distribution) # indices of MT nucleations
            mt_nucleations = np.where(nucleation_random_numbers < nucleation_distribution) # indices of MT nucleations
            # execute MT nucleations
            self.add_microtubules(mt_nucleations) 
            # print(f'nucleations: {mt_nucleations}')

            # choose which MTs experience catastrophes
            select_filled_sites_only = np.zeros(len(self.spindle_state)) + (self.spindle_state==4).astype(int) + (self.spindle_state == 2).astype(int)
            catastrophe_distribution = (self.catastrophe_rate * self.timestep_size) * select_filled_sites_only
            # print(f'catastrophe_distribution: {catastrophe_distribution}')

            # mt_catastrophes = np.where(spindle_update_random_numbers < catastrophe_distribution) # indices of MT catastrophes
            mt_catastrophes = np.where(catastrophe_random_numbers < catastrophe_distribution) # indices of MT catastrophes
            # execute MT catastrophes
            self.remove_microtubules(mt_catastrophes)

            # print(f'catastrophes: {mt_catastrophes}')

            # evolve time with new spindle state and calculate cost
            self.mtoc_time_evolution()
            new_cost = self.calc_cost()
            new_spindle_state = self.spindle_state

            # print(f'new spindle state {new_spindle_state}')

            # if there is no improvement in cost, reset the changes
            if np.round(new_cost, cost_acceptance_resolution) >= np.round(old_cost, cost_acceptance_resolution):
                # print('not accepted')
            # if new_cost >= old_cost:
            # if new_cost - old_cost >= -1e6:
                attempt_counter += 1

                # reset MTOC position
                self.set_mtoc_pos(old_mtoc_pos)

                # nucleate MTs which had catastrophes
                self.add_microtubules(mt_catastrophes)

                # MTs which nucleated are removed
                self.remove_microtubules(mt_nucleations)

                # print(f'spindle state returned to {self.spindle_state}')

        # print(f'accepted spindle state {self.spindle_state}')
        # print((new_spindle_state == old_spindle_state).all())

        return attempt_counter
    

    def simulate(self, max_time, readout=False, save=False, file_prefix='spindle-simulation', update_spindle=True):
    
        # initializing 
        data = {}
        data['spindle'] = self.as_dict()
        
        t = 0
        last_spindle_update_time = 0
        number_of_spindle_updates = 0
        most_recent_number_of_attempts = 0

        boundary_violated = False

        # saving data for later
        trajectory = {}
            
        with Live(console=console, refresh_per_second=4) as live:
            while t < (max_time - self.timestep_size) and not boundary_violated:
                # save data at this time
                timepoint_data = {
                    'spindle_state': self.spindle_state.astype(int),
                    'mtoc_pos': self.mtoc_pos,
                    'boundary_violated': boundary_violated,
                    'cost': self.calc_cost(),
                    'num_update_attempts': most_recent_number_of_attempts,
                }
                trajectory[t] = timepoint_data

                # MTOC position and cost before time evolution
                old_mtoc_pos = self.mtoc_pos
                old_cost = self.calc_cost()

                # time evolution and saving MTOC position and cost after time evolution
                new_mtoc_pos, boundary_violated = self.mtoc_time_evolution()
                new_cost = self.calc_cost()

                # if new_cost >= old_cost, change the spindle state

                most_recent_number_of_attempts = 0
                if (update_spindle): # and (new_cost - old_cost >= 0): #-1e-7: # forces turnover by disallowing stasis
                    # undo most recent time evolution step
                    # self.set_mtoc_pos(old_mtoc_pos)

                    # change spindle state
                    most_recent_number_of_attempts = self.gradient_descent_spindle_update()
                    new_cost = self.calc_cost()
                    last_spindle_update_time = np.round(np.copy(t), 3)
                    number_of_spindle_updates += 1
                    # most_recent_number_of_attempts = np.copy(attempts)

                t = t + self.timestep_size

                # readout table
                if readout:
                    table = Table(title="Spindle Simulation")
                    table.add_column("Parameter", justify="left")
                    table.add_column("Value", justify="right")
                    table.add_row("Time", f"{t:.2f}")
                    table.add_row("Progress", f"{(100 * t/max_time):.2f}%")
                    table.add_row("Boundary Violated", str(boundary_violated))
                    table.add_row("Current Position", str(self.mtoc_pos))
                    table.add_row("Current cost", str(self.calc_cost()))
                    table.add_row("Last Cost Delta", str(new_cost - old_cost))
                    table.add_row("Spindle State", str(self.spindle_state.astype(int)))
                    table.add_row("Direction of Motion", f"{normalize_vecs(new_mtoc_pos - old_mtoc_pos)[0]}")
                    table.add_row("Last Spindle Update Time", str(np.round(last_spindle_update_time, 3)))
                    table.add_row("Spindle Update Attempts", str(most_recent_number_of_attempts))
                    table.add_row("Number of Spindle Updates", str(number_of_spindle_updates))
                    live.update(table)

        data['trajectory'] = trajectory
        
        if save:

            # finding data directory
            parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
            target_child_dir = os.path.join(parent_dir, "data")
            os.makedirs(target_child_dir, exist_ok=True)

            # writing path to save file
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{file_prefix}_{timestamp}.pkl"
            file_path = os.path.join(target_child_dir, filename)

            # saving file
            with open(file_path, "wb") as f:
                pickle.dump(data, f)

            print(f'data saved to {file_path}')

        return data