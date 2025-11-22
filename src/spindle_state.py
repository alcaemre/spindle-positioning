#
# Emre Alca
# University of Pennsylvania
# Created on Sat Nov 22 2025
# Last Modified: 2025/11/22 17:55:50
#


import numpy as np
import tqdm
import matplotlib.pyplot as plt

from datetime import datetime

def normalize_vecs(vecs):
    """normalizes an array of vectors

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
            f_pull_0=1,
            rigidity=1, 
            friction_coefficient=1, 
            growth_rate=1, 
            stall_force=1,
            drag_factor=100,
            boundary_radius=1,
            timestep_size=0.1
            ):
        """initializes a Spindle with a single centrosome

        Args:
            initial_mtoc_pos (nunpy.array): position of the centrosome in (x,y,z) coordinates.
            initial_spindle_state (numpy.array): initial spindle state shape, contains N elements for N sites.
            lattice_sites (numpy.array): array of the coordinates of the sites, shape is N x 3 for N sites.
        hyperparameters:
            rigidity (float, optional): rigidity coefficient of a single MT. Defaults to 1.
            friction_coefficient (float, optional): friction coefficient between the cortex and the MT. Defaults to 1.
            growth_rate (float, optional): velocity of growth of the MT. Defaults to 1.
            stall_force (float, optional): stall force of the MT. Defaults to 1.
            drag_factor (float, optional): the stokes drag factor of the mtoc, this is the force of drag divided by velocity. Defaults to 100.
            boundary_radius (float, optional): the radius of the boundary sphere. Defaults to 1
            timestep_size (float, optional): the size of a timestep
        """

        self.spindle_state = initial_spindle_state
        
        self.lattice_sites = lattice_sites

        self.boundary_norms = normalize_vecs(lattice_sites)[0]

        self.set_mtoc_pos(initial_mtoc_pos)

        self.num_sites = len(self.spindle_state)

        # -- setting hyperparameters --
        self.f_pull_0 = f_pull_0
        self.rigidity = rigidity 
        self.friction_coefficient = friction_coefficient
        self.growth_rate = growth_rate
        self.stall_force = stall_force
        self.drag_factor = drag_factor
        self.boundary_radius = boundary_radius
        self.timestep_size = timestep_size



    def set_mtoc_pos(self, new_mtoc_pos):
        """ Sets the mtoc to be at a particular position.
        Also updates the mt vectors and directions to consider this new mtoc position.

        Args:
            new_mtoc_pos (numpy.array): new mtoc position in form numpy.array([x, y, z])
        """

        self.mtoc_pos = new_mtoc_pos # set new mtoc position 

        self.mt_vecs = (self.lattice_sites - self.mtoc_pos) # update mt vectors and directions
        self.mt_dirs, self.mt_norms = normalize_vecs(self.mt_vecs) 


    def add_microtubules(self, mt_indices_to_add):
        """Adds a microtubule to each site at the i-th position for each index i in mt_indices_to_add.

        Args:
            mt_indices_to_add (numpy.array): list containing the indices of microtubules to add

        Raises:
            ValueError: cannot add a microtubule to a site already containing a microtubule
        """

        if len(np.where(self.spindle_state[mt_indices_to_add] == 2)[0]) != 0 or len(np.where(self.spindle_state[mt_indices_to_add] != 4)[0]) == 0:
            raise ValueError("cannot add a microtubule to a site already containing a microtubule")
        
        update = np.zeros(self.num_sites)

        update[mt_indices_to_add] = 1
        
        self.spindle_state = self.spindle_state + update


    def remove_microtubules(self, mt_indices_to_remove):
        """removes a microtubule to each site at the i-th position for each index i in mt_indices_to_remove.

        Args:
            mt_indices_to_remove (numpy.array): list containing the indices of microtubules to remove

        Raises:
            ValueError: cannot remove a microtubule from a site which does not contain a microtubule
        """

        if len(np.where(self.spindle_state[mt_indices_to_remove] == 1)[0]) != 0 or len(np.where(self.spindle_state[mt_indices_to_remove] == 3)[0]) != 0:
            raise ValueError("cannot remove a microtubule from a site which does not contain a microtubule")
        
        update = np.zeros(self.num_sites)

        update[mt_indices_to_remove] = -1
        
        self.spindle_state = self.spindle_state + update


    def calculate_pulling_forces(self):
        """ Calculates the pulling force experienced by the mtoc.
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
        pushing_boundary_norms = self.boundary_norms[self.spindle_state == 2]
        effective_force_coefficients = np.sum(pushing_mt_dirs * pushing_boundary_norms, axis=1)

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
        """Evolves the MTOC position by the MTOC velocity times the size of a timestep.
        This function returns both the new MTOC position, and a bool representing whether the boundary has been violated.
        If the boundary is violated, the unit direction of the current position is taken, and multiplied by the boundary's radius.
        NOTE: is that this function does not automatically update the MTOC's position.

        Returns:
            np.array: the new MTOC position after one timestep in the form of numpy.array([x,y,z]),
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

        return new_mtoc_pos, boundary_violated