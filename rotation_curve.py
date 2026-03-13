import os

import h5py
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_rotation_curve(
    rot_curve_file: str,
    box_num: int,
    output_dir: str = "Plots",
) -> str:
    """
    Make a rotation-curve plot from precomputed data stored in an HDF5 file.

    Parameters
    ----------
    rot_curve_file : str
        Path to the HDF5 file created by load_sim_data.compute_rotation_curve_and_save.
    box_num : int
        Box identifier for labeling and output filename.
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

    fig, axs = plt.subplots(figsize=(8, 6))

    axs.plot(rs, vrot, color="k", lw=2)
    axs.plot(rs, vrot_dm_only, color="r", ls=":")
    axs.plot(rs, vrot_gas_only, color="orange", ls=":")
    axs.plot(rs, vrot_stars_only, color="b", ls=":")

    axs.text(
        0.95,
        0.90,
        r"${\rm DM}$",
        color="r",
        transform=plt.gca().transAxes,
        ha="right",
    )
    axs.text(
        0.95,
        0.825,
        r"${\rm Gas}$",
        color="orange",
        transform=plt.gca().transAxes,
        ha="right",
    )
    axs.text(
        0.95,
        0.75,
        r"${\rm Stars}$",
        color="b",
        transform=plt.gca().transAxes,
        ha="right",
    )

    axs.set_xlabel(r"${\rm Radius~[kpc]}$")
    axs.set_ylabel(r"$V_{\rm rot}~[{\rm km/s}]$")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)

    os.makedirs(output_dir, exist_ok=True)
    outname = os.path.join(output_dir, f"FIRE2_MW_box{box_num}_rotation_curve2.png")
    fig.savefig(outname, bbox_inches="tight")
    plt.close(fig)

    return outname


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_rotation_curve()."
    )

