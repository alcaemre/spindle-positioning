#
# Emre Alca
# University of Pennsylvania
# Created on Tue Feb 24 2026
# Last Modified: 2026/03/04 15:01:32
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

import pickle
import os

from src import spindle_state as ss


if __name__ == "__main__":
    # -- reconstructing a trajectory and animating it

    # dir_path = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/2D-single-mt-update-200-points-5-mts-50-positions_2026-02-24_01-41-04' # 5 MTs
    # dir_path = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/2D-1000-points-10-mts-50-positions_2026-02-24_22-23-36' # 10 MTs
    # dir_path = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/2D-10000-points-100-mts-50-positions_2026-02-26_23-58-27' #100 MTs
    dir_path = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/data/pushing-2D-10000-points-100-mts-50-positions_2026-03-03_12-50-01' # 100 MTs pushing

    spindle_from_dict = ss.restart_experiment_from_directory(dir_path, readout=True)

    print('animating')

    ani = spindle_from_dict.animate(interval=100, save=True, file_prefix='')

    plt.show()