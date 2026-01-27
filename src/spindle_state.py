#
# Emre Alca
# University of Pennsylvania
# Created on Sat Nov 22 2025
# Last Modified: 2026/01/27 13:53:48
#


import numpy as np
np.set_printoptions(formatter={'float': '{:.3f}'.format})
import tqdm
import matplotlib.pyplot as plt

from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table

import pickle
import os

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
            mt_len_cost_punishment_degree=4,
            # -- biophysical constants
            f_pull_0=1,
            rigidity=1, 
            friction_coefficient=1, 
            growth_rate=1, 
            stall_force=1,
            drag_factor=100,
            cytoplasmic_catastrophe_rate=1,
            boundary_radius=1,
            max_total_mt_length=None,
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
            max_total_mt_length (float or None, optional): maximum total MT length before punishment. Defaults to the number of MT sites times the boundary radius.
            mt_len_cost_punishment_degree(int, optional): power to which the MT total length cost term is raised. Defaults to 4.
        """

        # setting parameters
        self.spindle_state = initial_spindle_state
        
        self.lattice_sites = lattice_sites

        self.boundary_unit_normals = normalize_vecs(lattice_sites)[0]

        self.set_mtoc_pos(initial_mtoc_pos)

        self.num_sites = len(self.spindle_state)

        # setting hyperparameters
        self.f_pull_0 = f_pull_0
        self.rigidity = rigidity 
        self.friction_coefficient = friction_coefficient
        self.growth_rate = growth_rate
        self.stall_force = stall_force
        self.drag_factor = drag_factor
        self.boundary_radius = boundary_radius
        self.timestep_size = timestep_size

        if max_total_mt_length is None:
            max_total_mt_length = boundary_radius * len(self.spindle_state)

        self.max_total_mt_length = max_total_mt_length
        self.mt_len_cost_punishment_degree = mt_len_cost_punishment_degree
        self.cytoplasmic_catastrophe_rate = cytoplasmic_catastrophe_rate
        

    def as_dict(self):
        """
        creates a dictionary of the parameters and hyperparameters of a spindle and returns the dictionary

        Returns:
            dict: dictionary of all parameters and hyperparameters in the spindle.
        """

        spindle_dict = {}
        # -- parameters --
        spindle_dict['mtoc_pos'] = self.mtoc_pos
        spindle_dict['spindle_state'] = self.spindle_state
        spindle_dict['lattice_sites'] = self.lattice_sites

        # -- hyperparameters --
        spindle_dict['f_pull_0'] = self.f_pull_0
        spindle_dict['rigidity'] = self.rigidity
        spindle_dict['friction_coefficient'] = self.friction_coefficient
        spindle_dict['growth_rate'] = self.growth_rate
        spindle_dict['stall_force'] = self.stall_force
        spindle_dict['drag_factor'] = self.drag_factor
        spindle_dict['boundary_radius'] = self.boundary_radius
        spindle_dict['timestep_size'] = self.timestep_size
        spindle_dict['max_total_mt_length'] = self.max_total_mt_length
        spindle_dict['mt_len_cost_punishment_degree'] = self.mt_len_cost_punishment_degree
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

        buckling_forces = np.pi * self.rigidity / (self.mt_norms[self.spindle_state == 2]**2)

        # -- calculating unbuckled pushing forces --

        # calculating the effective force coefficients (mt_dir . boundary_norm)
        pushing_mt_dirs = self.mt_dirs[self.spindle_state == 2]
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

        # -- summing over pushing force vectors to find total pushing force --

        # pushing_mt_dirs point outwards from the mtoc, we want pushing forces to point inwards towards the mtoc
        total_pushing_force = -np.sum(pushing_vectors, axis=0) 

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
        new_mtoc_pos = self.mtoc_pos + (self.calc_mtoc_velocity() * self.timestep_size)

        # check that the new mtoc position is not outside of the radius
        normalized_new_mtoc_pos, new_mtoc_pos_norm = normalize_vecs(new_mtoc_pos)

        boundary_violated = False
        if new_mtoc_pos_norm > 1:
            # if the new MTOC position is outside of the radius, place it on the radius, pointing in the same direction
            new_mtoc_pos = normalized_new_mtoc_pos * self.boundary_radius
            boundary_violated = True

        self.set_mtoc_pos(new_mtoc_pos)
        return new_mtoc_pos, boundary_violated
    

    def calc_cost(self):
        """
        Calculates the cost of the current position and spindle state.
        The cost has two terms: the first is the square of the displacement of the MTOC.
        The second is a punishment term for the spindle using too much tubulin, 
        which is zero when the total MT length is less than the max total MT length.
        Once there is more tubulin than that maximum, the square of the difference between 
        the current amount and the allowed maximum is added to the cost.

        Returns:
            float: cost = displacement cost + length cost.
        """
        displacement_cost = np.square(normalize_vecs(self.mtoc_pos)[1])

        total_mt_length = np.sum(self.mt_norms[self.spindle_state==2]) + np.sum(self.mt_norms[self.spindle_state==4])

        mt_length_cost = 0
        if total_mt_length > self.max_total_mt_length:
            mt_length_cost = np.power(total_mt_length - self.max_total_mt_length, self.mt_len_cost_punishment_degree)

        return displacement_cost + mt_length_cost
    

    def biased_spatial_nucleation_distribution(self):
        """ Calculates the biased spatial nucleation sampling  distribution

        These distributions are biased to nucleate MTs which exert force along the minus position vector
        so as to bias towards samples which reduce the cost.

        Returns:
            tuple[numpy.array]: biased_spatial_nucleation_distribution
        """

        # find the unoccupied sites and set pulling = 1, pushing = -1 
        f_vec = np.zeros(len(self.spindle_state)) + (self.spindle_state==3).astype(int) - (self.spindle_state == 1).astype(int)
        f_mhat_vec =  (self.mt_dirs.T * f_vec.T).T

        # minus the norm of the MTOC position
        minus_r_hat = -normalize_vecs(self.mtoc_pos)[0]

        # normalized distribution biased by the dot product between each mt direction vector and minus_r_hat
        f_mhat_dot_minus_r_hat = f_mhat_vec @ minus_r_hat

        # transforming dot product into biased nucleation distribution
        biased_spatial_nucleation_distribution = (f_mhat_dot_minus_r_hat + 1) / np.pi

        select_empty_sites_only = np.zeros(len(self.spindle_state)) + (self.spindle_state==3).astype(int) + (self.spindle_state == 1).astype(int)

        return biased_spatial_nucleation_distribution * select_empty_sites_only
    

    def biased_spatial_catastrophe_distribution(self):
        """
        Calculates the biased spatial catastrophe sampling distribution

        These distributions are biased to depolymerize MTs which exert force opposing the minus position vector
        so as to bias towards samples which reduce the cost.

        Returns:
            tuple[numpy.array]: biased_spatial_catastrophe_distribution
        """

        # find the unoccupied sites and set pulling = 1, pushing = -1 
        f_vec = np.zeros(len(self.spindle_state)) + (self.spindle_state==4).astype(int) - (self.spindle_state == 2).astype(int)
        f_mhat_vec =  (self.mt_dirs.T * f_vec.T).T

        # minus the norm of the MTOC position
        minus_r_hat = -normalize_vecs(self.mtoc_pos)[0]

        # normalized distribution biased by the dot product between each mt direction vector and minus_r_hat
        f_mhat_dot_minus_r_hat = f_mhat_vec @ minus_r_hat

        # transforming dot product into biased catastrophe distribution
        biased_spatial_catastrophe_distribution = (1 - f_mhat_dot_minus_r_hat) / np.pi

        select_filled_sites_only = np.zeros(len(self.spindle_state)) + (self.spindle_state==4).astype(int) + (self.spindle_state == 2).astype(int)

        return biased_spatial_catastrophe_distribution * select_filled_sites_only
    
    def calculate_mt_length_distribution(self):
        """
        Calculates the the probabilities of an MT growing to be long enough to reach the cortex at each lattice site
        based on the exponential distribution of MT lengths using the growth_rate and cytoplasmic_catastrophe_rate hyperparameters.

        Returns:
            np.array: array of the probabilities of an MT growing to be long enough to reach the cortex at each lattice site.
        """
        average_mt_length = self.growth_rate / self.cytoplasmic_catastrophe_rate
        return average_mt_length * np.exp(- average_mt_length * self.mt_norms)


    def biased_length_nucleation_distribution(self):
        """
        A biased distribution for nucleating MTs where shorter MTs more likely to be nucleated than longer MTs.
        Weights are given by one minus the exponential length distribution of MTs.

        Returns:
            np.array: array of biased probabilities preferring to nucleate shorter MTs
        """

        mt_length_probabilities = self.calculate_mt_length_distribution()

        select_empty_sites_only = np.zeros(len(self.spindle_state)) + (self.spindle_state==3).astype(int) + (self.spindle_state == 1).astype(int)

        return mt_length_probabilities * select_empty_sites_only
    

    def biased_length_catastrophe_distribution(self):
        """
        A biased distribution for depolymerizing MTs where longer MTs more likely to undergo catastrophe than shorter MTs.
        Weights are given by one minus the exponential length distribution of MTs.

        Returns:
            np.array: array of biased probabilities preferring to depolymerize longer MTs
        """

        mt_length_probabilities = self.calculate_mt_length_distribution()

        select_filled_sites_only = np.zeros(len(self.spindle_state)) + (self.spindle_state==4).astype(int) + (self.spindle_state == 2).astype(int)

        return (1 - mt_length_probabilities) * select_filled_sites_only

    def gradient_descent_spindle_update(self):

        # save old spindle state, cost, and MTOC position
        old_spindle_state = np.copy(self.spindle_state)
        old_mtoc_pos = np.copy(self.mtoc_pos)
        old_cost = self.calc_cost()

        # catastrophe distribution
        catastrophe_distribution = self.biased_spatial_catastrophe_distribution() * self.biased_length_catastrophe_distribution() # spatial and length biasing

        # nucleation_distribution
        nucleation_distribution = self.biased_spatial_nucleation_distribution() * self.biased_length_nucleation_distribution() # spatial and length biasing

        # --- only accept modifications which reduce the cost ---

        new_cost = np.copy(old_cost)
        new_spindle_state = np.copy(old_spindle_state)

        attempt_counter = 1
        cost_acceptance_resolution = 6

        while (np.round(new_cost, cost_acceptance_resolution) > np.round(old_cost, cost_acceptance_resolution)) or (new_spindle_state == old_spindle_state).all():
            # print(f'attempt number: {attempt_counter}')

            # sample random numbers to compare to distributions
            spindle_update_random_numbers = np.random.rand(len(self.spindle_state))

            # choose which MTs experience catastrophes
            mt_catastrophes = np.where(spindle_update_random_numbers < catastrophe_distribution) # indices of MT catastrophes
            # execute MT catastrophes
            self.remove_microtubules(mt_catastrophes)

            # print(f'catastrophes: {mt_catastrophes}')

            # choose which MTs nucleate
            mt_nucleations = np.where(spindle_update_random_numbers < nucleation_distribution) # indices of MT nucleations
            # execute MT nucleations
            self.add_microtubules(mt_nucleations) 

            # print(f'nucleations: {mt_nucleations}')

            # evolve time with new spindle state and calculate cost
            self.mtoc_time_evolution()
            new_cost = self.calc_cost()
            new_spindle_state = self.spindle_state

            # print(f'new spindle state {new_spindle_state}')

            # if there is no improvement in cost, reset the changes
            if np.round(new_cost, cost_acceptance_resolution) > np.round(old_cost, cost_acceptance_resolution):
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

                # MTOC position and cost before time evolution
                old_mtoc_pos = self.mtoc_pos
                old_cost = self.calc_cost()

                # time evolution and saving MTOC position and cost after time evolution
                new_mtoc_pos, boundary_violated = self.mtoc_time_evolution()
                new_cost = self.calc_cost()

                # if new_cost >= old_cost, change the spindle state

                attempts = 0
                if (new_cost - old_cost >= 0) and (update_spindle): #-1e-7: # forces turnover by disallowing stasis
                    # undo most recent time evolution step
                    self.set_mtoc_pos(old_mtoc_pos)

                    # change spindle state
                    attempts = self.gradient_descent_spindle_update()
                    new_cost = self.calc_cost()
                    last_spindle_update_time = np.round(np.copy(t), 3)
                    number_of_spindle_updates += 1
                    most_recent_number_of_attempts = np.copy(attempts)

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

                timepoint_data = {
                    'spindle_state': self.spindle_state.astype(int),
                    'mtoc_pos': self.mtoc_pos,
                    'boundary_violated': boundary_violated,
                    'cost': self.calc_cost(),
                    'num_update_attempts': attempts,
                }
                trajectory[t] = timepoint_data

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