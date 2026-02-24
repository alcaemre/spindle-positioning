#
# Emre Alca
# University of Pennsylvania
# Created on Mon Feb 23 2026
# Last Modified: 2026/02/23 17:08:43
#

import numpy as np

def circle_lattice(num_sites):
    """Makes a lattice in the shape of a circle in x,y in z=0

    Args:
        num_sites (int): number of points on the circle

    Returns:
        numpy.ndarray: lattice of sites in a circle in x,y with z=0
    """
    theta = np.arange(0, 2* np.pi, 2* np.pi/num_sites)
    x = np.cos(theta)
    y = np.sin(theta)
    lattice = []
    spindle_state = []
    for i in range(len(theta)):
        lattice.append([x[i], y[i], 0])
        spindle_state.append(3)

    return np.array(lattice)