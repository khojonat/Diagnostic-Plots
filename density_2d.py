import os

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from load_sim_data import load_particles, identify_target_halo, loadHalos

def plot_2d_hist(
    box_num,
    snapnum,
    parttype,
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
    box_num: int
        Box number for the simulation.
    snapnum: int
        Snapshot number for the simulation.
    parttype: int
        Particle type to make image of
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
    target = identify_target_halo(box_num, snapnum)
    Positions = load_particles(box_num, parttype, ['Coordinates'], snapnum=snapnum)
    Masses = load_particles(box_num, parttype, ['Masses'], snapnum=snapnum)

    halo_length = loadHalos(box_num,snapnum,'GroupLenType')

    # Using halo lengths to index DM particles
    Particle_positions = Positions[np.sum(halo_length[:target,1]):np.sum(halo_length[:target,1]) + halo_length[target,1]]
    Particle_Masses = Masses[np.sum(halo_length[:target,1]):np.sum(halo_length[:target,1]) + halo_length[target,1]]

    H, xedges, yedges = np.histogram2d(Particle_positions[:,0], Particle_positions[:,1], bins=(nbins,nbins), weights=Particle_Masses)

    fig, ax = plt.subplots(figsize=(6, 6))

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        norm = 'log'
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Counts")

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

