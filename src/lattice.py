#
# Emre Alca
# University of Pennsylvania
# Created on Mon Feb 23 2026
# Last Modified: 2026/03/06 13:27:13
#

import numpy as np
import os

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sloane_cache')


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


def sphere_lattice(N, R=1.0):
    """Loads a Sloane best-known sphere packing for N points (N=4..130).
    Run scripts/build_sloane_cache.py once to populate the cache.

    Args:
        N (int): number of points in lattice (must be 4–130)
        R (float, optional): radius of the sphere. Defaults to 1.0.

    Returns:
        numpy.ndarray: (N, 3) array of points on a sphere of radius R
    """
    cache_path = os.path.join(_CACHE_DIR, f"sloane_{N}.npy")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No cached packing for N={N}. "
            f"Run scripts/build_sloane_cache.py to download it."
        )
    pts = np.load(cache_path)
    return pts * R