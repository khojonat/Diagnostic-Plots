import os
import h5py
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_rotation_curve(
    rot_curve_file: str,
    run_num: int,
    output_dir: str = "Plots",
    filename_suffix: str = "",
    ax=None,
) -> str | None:
    """
    Make a rotation-curve plot from precomputed data stored in an HDF5 file.
    Code originally from Alex Garcia. 

    Parameters
    ----------
    rot_curve_file : str
        Path to the HDF5 file created by helpers.compute_rotation_curve_and_save.
    run_num : int
        Simulation run number for labeling and output filename.
    output_dir : str, optional
        Directory where the rotation-curve plot PNG will be saved.

    Returns
    -------
    str
        Path to the saved rotation-curve plot.
    """
    with h5py.File(rot_curve_file, "r") as f:
        rs = f["rs"][:]  # kpc
        vrot = f["vrot"][:]  # km/s
        vrot_dm_only = f["vrot_dm_only"][:]
        vrot_gas_only = f["vrot_gas_only"][:]
        vrot_stars_only = f["vrot_stars_only"][:]

    owns_figure = ax is None
    if owns_figure:
        fig, axs = plt.subplots(figsize=(8, 6))
    else:
        fig, axs = ax.figure, ax

    axs.plot(rs, vrot, color="k", lw=2)
    axs.plot(rs, vrot_dm_only, color="r", ls=":")
    axs.plot(rs, vrot_gas_only, color="orange", ls=":")
    axs.plot(rs, vrot_stars_only, color="b", ls=":")

    axs.text(
        0.95,
        0.90,
        r"${\rm DM}$",
        color="r",
        transform=axs.transAxes,
        ha="right",
    )
    axs.text(
        0.95,
        0.825,
        r"${\rm Gas}$",
        color="orange",
        transform=axs.transAxes,
        ha="right",
    )
    axs.text(
        0.95,
        0.75,
        r"${\rm Stars}$",
        color="b",
        transform=axs.transAxes,
        ha="right",
    )

    axs.set_xlabel(r"${\rm Radius~[kpc]}$", fontsize=15)
    axs.set_ylabel(r"$V_{\rm rot}~[{\rm km/s}]$", fontsize=15)

    if owns_figure:
        fig.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        outname = os.path.join(output_dir, f"rotation_curve2_run_{run_num}{filename_suffix}.png")
        fig.savefig(outname, bbox_inches="tight")
        plt.close(fig)
    else:
        outname = None

    return outname


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_rotation_curve()."
    )
