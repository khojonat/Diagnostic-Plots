import os

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from .helpers import load_particles, identify_target_halo, loadHalos, unit_mass, unit_distance

def plot_2d_hist(
    data_dir,
    snapnum,
    parttype,
    target: int,
    nbins: int = 512,
    output_path: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str | None = None,
    cmap: str = "viridis",
) -> str:
    """
    Produce a 2D histogram (density map) for arbitrary x/y data.

    Parameters
    ----------
    data_dir: str
        Simulation data directory.
    snapnum: int
        Snapshot number for the simulation.
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
    
    # Identifying target particles: 
    Positions = load_particles(data_dir, parttype, ['Coordinates'], snapnum=snapnum)
    Masses = load_particles(data_dir, parttype, ['Masses'], snapnum=snapnum)

    halo_length = loadHalos(data_dir,snapnum,'GroupLenType')

    # Using halo lengths to index DM particles
    Particle_positions = Positions['Coordinates'][np.sum(halo_length[:target,parttype]):np.sum(halo_length[:target,parttype]) + halo_length[target,parttype]]
    Particle_Masses = Masses['Masses'][np.sum(halo_length[:target,parttype]):np.sum(halo_length[:target,parttype]) + halo_length[target,parttype]]

    H, xedges, yedges = np.histogram2d(Particle_positions[:,0], Particle_positions[:,1], bins=(nbins,nbins), weights=Particle_Masses)

    # Convert from code units to physical units (Msun/kpc^2)
    # H is in code mass per code distance^2, so multiply by unit_mass / unit_distance^2
    H = H * unit_mass / (unit_distance ** 2)

    fig, ax = plt.subplots(figsize=(6, 6))

    extent = [xedges[0] * unit_distance, xedges[-1] * unit_distance, yedges[0] * unit_distance, yedges[-1] * unit_distance]
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        norm = 'log'
    )

    ax.set_xlabel(xlabel + " (kpc)")
    ax.set_ylabel(ylabel + " (kpc)")
    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax,shrink=0.7)
    cbar.set_label("Surface Density (Msun/kpc$^2$)")

    fig.tight_layout()

    if output_path is None:
        os.makedirs("Plots", exist_ok=True)
        output_path = os.path.join("Plots", f"PartType{parttype}_density_2d.png")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_2d_hist()."
    )

