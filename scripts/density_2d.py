import os

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from .helpers import load_particles, identify_target_halo, loadHalos, unit_mass, unit_distance

def plot_2d_hist(
    run_num,
    snapnum,
    parttype,
    target: int,
    nbins: int = 512,
    output_path: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str | None = None,
    cmap: str = "viridis",
    particle_data: dict | None = None,
    ax=None,
) -> str | None:
    """
    Produce a 2D histogram (density map) for arbitrary x/y data.

    Parameters
    ----------
    run_num: int
        Simulation run number; files are read from ``snapshot_base/run_<run_num>``.
    snapnum: int
        Snapshot number for the simulation. Can alternatively be set to 'test'
    parttype: int
        Particle type to make image of
    target: int
        Target halo index.
    nbins : int, optional
        Number of bins along each axis (default: 512).
    output_path : str, optional
        Full path to save the PNG. If None, saves to 'Plots/density_2d.png'.
    xlabel, ylabel : str, optional
        Axis labels for the plot.
    title : str, optional
        Plot title.
    cmap : str, optional
        Matplotlib colormap name.

    Returns
    -------
    str
        Path to the saved image file.
    """

    if particle_data is not None:
        Particle_positions = particle_data["particles"][parttype]["Coordinates"]
        Particle_Masses = particle_data["particles"][parttype]["Masses"]

    # Identifying target particles:
    elif snapnum != 'test':
        
        Particle_positions = load_particles(run_num, parttype, ['Coordinates'], snapnum=snapnum)
        Particle_Masses = load_particles(run_num, parttype, ['Masses'], snapnum=snapnum)
    
        halo_length = loadHalos(run_num, snapnum, 'GroupLenType')
    
        # Using halo lengths to index DM particles
        Particle_positions = Particle_positions['Coordinates'][np.sum(halo_length[:target,parttype]):np.sum(halo_length[:target,parttype]) + halo_length[target,parttype]]
        Particle_Masses = Particle_Masses['Masses'][np.sum(halo_length[:target,parttype]):np.sum(halo_length[:target,parttype]) + halo_length[target,parttype]]

    else:

        Particle_positions = load_particles(run_num, parttype, ['Coordinates'], snapnum='test')['Coordinates']
        Particle_Masses = load_particles(run_num, parttype, ['Masses'], snapnum='test')['Masses']
        

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    positions = np.asarray(Particle_positions)
    masses = np.asarray(Particle_Masses)
    valid_data = (
        positions.ndim == 2
        and positions.shape[1] >= 2
        and masses.ndim == 1
        and len(positions) == len(masses)
    )
    if valid_data:
        valid = np.isfinite(positions[:, 0]) & np.isfinite(positions[:, 1]) & np.isfinite(masses)
        positions, masses = positions[valid], masses[valid]

    if valid_data and len(positions):
        positions = positions - np.median(positions, axis=0)
        H, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1], bins=(nbins, nbins), weights=masses
        )
        H = H * unit_mass / (unit_distance ** 2)
        positive = H[np.isfinite(H) & (H > 0)]
        if positive.size:
            vmin, vmax = positive.min(), positive.max()
            if vmin == vmax:
                vmax = vmin * 1.01
            extent = [
                xedges[0] * unit_distance, xedges[-1] * unit_distance,
                yedges[0] * unit_distance, yedges[-1] * unit_distance,
            ]
            im = ax.imshow(
                H.T, origin="lower", extent=extent, aspect="equal", cmap=cmap,
                norm=LogNorm(vmin=vmin, vmax=vmax),
            )
            cbar = fig.colorbar(im, ax=ax, shrink=0.6)
            cbar.set_label(r"$\rm M_\odot/kpc^2$", fontsize=15)

    ax.set_xlabel(xlabel + " (kpc)",fontsize=15)
    ax.set_ylabel(ylabel + " (kpc)",fontsize=15)
    if title is not None:
        ax.set_title(title)

    if owns_figure:
        fig.tight_layout()
        if output_path is None:
            os.makedirs("Plots", exist_ok=True)
            output_path = os.path.join("Plots", f"PartType{parttype}_density_2d.png")
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    else:
        output_path = None

    return output_path


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_2d_hist()."
    )
