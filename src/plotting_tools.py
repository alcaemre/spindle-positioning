#
# Emre Alca
# University of Pennsylvania
# Created on Fri Feb 27 2026
# Last Modified: 2026/03/13 12:56:50
#

import src.spindle_state as ss
import src.lattice as lat

import numpy as np
# np.set_printoptions(formatter={'float': '{:.3f}'.format})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# %matplotlib widget

from rich.console import Console
from rich.live import Live
from rich.table import Table
from tqdm import tqdm

console = Console()


# starting with only three points

def phase_portrait_2D(spindle, z, xlim, ylim, n=25, *, quiver=True, stream=True,
                            cmap='viridis', title=None, xlabel='x', ylabel='y',
                            quiver_scale=1.5, density=1.0, arrowsize=1.2, linewidth=0.8):
    """
    Plot a 2D phase portrait for a velocity field v(x, y) = (u, v).

    Parameters
    ----------
    z : float
        value of z we are looking at the slice of.
    xlim, ylim : tuple
        (min, max) ranges for x and y.
    n : int
        Grid density for sampling.
    quiver : bool
        Draw a quiver plot (arrows).
    stream : bool
        Draw streamlines (with arrows).
    cmap : str
        Colormap used to color arrows by speed (quiver).
    quiver_scale : float
        Scale factor for quiver arrow lengths.
    density : float
        Streamline density; higher = more lines.
    arrowsize : float
        Streamline arrow size.
    linewidth : float
        Streamline line width.
    """

    # Build grid
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y, indexing='xy')

    dx_dt = np.zeros_like(X, dtype=float)
    dy_dt = np.zeros_like(Y, dtype=float)

    for i in tqdm(range(len(x))):
        for j in range(len(y)):
            pos = np.array([x[i],y[j],z])
            spindle.set_mtoc_pos(pos)
            velocity = spindle.calc_mtoc_velocity()
            # velocity = spindle_circle_3.calculate_pushing_forces()
            dx_dt[i,j] = velocity[0]
            dy_dt[i,j] = velocity[1]

    dx_dt, dy_dt = dx_dt.T, dy_dt.T

    speed = np.hypot(dx_dt, dy_dt)

    # Mask invalids if any
    mask = ~np.isfinite(dx_dt) | ~np.isfinite(dy_dt)
    if np.any(mask):
        dx_dt = np.where(mask, 0.0, dx_dt)
        dy_dt = np.where(mask, 0.0, dy_dt)
        speed = np.where(mask, np.nan, speed)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    if quiver:
        q = ax.quiver(X, Y, dx_dt, dy_dt, speed, cmap=cmap,
                      angles='xy', scale_units='xy', scale=quiver_scale, width=0.003)
        cbar = fig.colorbar(q, ax=ax, label='Speed')

    if stream:
        # For streamplot we pass 1D x, y and 2D U, V.
        ax.streamplot(x, y, dx_dt, dy_dt, color='k', density=density, arrowsize=arrowsize, linewidth=linewidth)

    if title is None:
        title = f'Phase Portrait at z={z}'
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim)
    ax.grid(alpha=0.25)
    ax.set_aspect('equal', 'box')

    # # -- plotting lattice sites -- 
    # site_color_list = np.where((spindle_circle_3.empty_spindle_state == 1) + (spindle_circle_3.empty_spindle_state == 2), 'tab:orange', 'tab:purple')
    # ax.scatter(spindle_circle_3.lattice_sites[:,0], spindle_circle_3.lattice_sites[:,1], c=site_color_list)
    # legend_handles = [
    #     Line2D([0],[0], marker='o', linestyle='None',
    #         color='tab:orange', alpha=0.8, markersize=8, label='Pushing'),
    #     Line2D([0],[0], marker='o', linestyle='None',
    #         color='tab:purple', alpha=0.8, markersize=8, label='Pulling'),
    # ]
    # ax.legend(handles=legend_handles, loc='upper right',bbox_to_anchor=(1.3, 1.0),)

    plt.tight_layout()
    return fig, ax


def phase_portrait_2D_with_z(spindle, z, xlim, ylim, n=25, *,
                              quiver=True, stream=True,
                              cmap='viridis', quiver_scale=1.5,
                              density=1.0, arrowsize=1.2, linewidth=0.8):
    """
    Two-panel plot at a fixed z-slice:
      Left  — x-y phase portrait (quiver + streamlines, coloured by in-plane speed)
      Right — dz/dt heatmap (red = moving up, blue = moving down)

    Shows the full 3D vector field at the slice without discarding the z component.
    """

    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y, indexing='xy')

    dx_dt = np.zeros_like(X, dtype=float)
    dy_dt = np.zeros_like(Y, dtype=float)
    dz_dt = np.zeros_like(X, dtype=float)

    for i in tqdm(range(len(x))):
        for j in range(len(y)):
            spindle.set_mtoc_pos(np.array([x[i], y[j], z]))
            # if np.linalg.norm(np.array([x[i],y[j], z])) > 1:
            #     v = np.zeros(3)
            # else:
            v = spindle.calc_mtoc_velocity()
            dx_dt[i, j] = v[0]
            dy_dt[i, j] = v[1]
            dz_dt[i, j] = v[2]

    dx_dt, dy_dt, dz_dt = dx_dt.T, dy_dt.T, dz_dt.T

    speed = np.hypot(dx_dt, dy_dt)
    mask  = ~np.isfinite(dx_dt) | ~np.isfinite(dy_dt)
    if np.any(mask):
        dx_dt = np.where(mask, 0.0, dx_dt)
        dy_dt = np.where(mask, 0.0, dy_dt)
        speed = np.where(mask, np.nan, speed)
        dz_dt = np.where(mask, np.nan, dz_dt)

    fig, (ax_xy, ax_z) = plt.subplots(1, 2, figsize=(13, 5.2))

    # -- left: x-y phase portrait --
    if quiver:
        q = ax_xy.quiver(X, Y, dx_dt, dy_dt, speed, cmap=cmap,
                         angles='xy', scale_units='xy', scale=quiver_scale, width=0.003)
        fig.colorbar(q, ax=ax_xy, label='In-plane speed')
    if stream:
        ax_xy.streamplot(x, y, dx_dt, dy_dt, color='k',
                         density=density, arrowsize=arrowsize, linewidth=linewidth)
    ax_xy.set(xlabel='x', ylabel='y', title=f'x-y phase portrait  (z={z})',
              xlim=xlim, ylim=ylim)
    ax_xy.set_aspect('equal', 'box')
    ax_xy.grid(alpha=0.25)

    # -- right: dz/dt heatmap --
    vmax = np.nanmax(np.abs(dz_dt))
    pcm  = ax_z.pcolormesh(X, Y, dz_dt, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, shading='auto')
    fig.colorbar(pcm, ax=ax_z, label=r'$\dot{z}$')
    ax_z.set(xlabel='x', ylabel='y', title=f'z-velocity heatmap  (z={z})',
             xlim=xlim, ylim=ylim)
    ax_z.set_aspect('equal', 'box')
    ax_z.grid(alpha=0.25)

    plt.tight_layout()
    return fig, (ax_xy, ax_z)


def radial_velocity_heatmap(
        spindle, z, xlim, ylim, n=25, 
        *,
        cmap='viridis', 
        title=None, 
        xlabel='x', ylabel='y',
        linewidth=0.8
    ):
    
    # Build grid
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y, indexing='xy')

    dx_dt = np.zeros_like(X, dtype=float)
    dy_dt = np.zeros_like(Y, dtype=float)
    dz_dt = np.zeros_like(Y, dtype=float)

    for i in tqdm(range(len(x))):
        for j in range(len(y)):
            pos = np.array([x[i],y[j],z])
            spindle.set_mtoc_pos(pos)
            # velocity = spindle.calc_mtoc_velocity()
            if np.linalg.norm(np.array([x[i],y[j], z])) > 1:
                velocity = np.zeros(3)
            else:
                velocity = spindle.calc_mtoc_velocity()
            # velocity = spindle_circle_3.calculate_pushing_forces()
            dx_dt[i,j] = velocity[0]
            dy_dt[i,j] = velocity[1]
            dz_dt[i,j] = velocity[2]

    dx_dt, dy_dt, dz_dt = dx_dt.T, dy_dt.T, dz_dt.T  # my grids are row, column indexed, but matplotlib expects column,row indexing

    dr_dt = ((X * dx_dt) + (Y * dy_dt) + z*(dz_dt)) / ((X*X) + (Y*Y) + (z*z))
    # dr_dt[dr_dt > 0.0] = 0
    # dr_dt = dz_dt

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    vmax = np.nanmax(np.abs(dr_dt))
    pcm = ax.pcolormesh(X, Y, dr_dt, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    fig.colorbar(pcm, ax=ax, label='Radial velocity $\\dot{r}$')

    if title is None:
        title = f'Radial Velocity at z={z}'
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim)
    ax.set_aspect('equal', 'box')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig, ax