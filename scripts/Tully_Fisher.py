import os

import h5py
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from .helpers import split_paired_array


# Literature data (kept as in original script)
SAMI = np.array([
    8.173529411764706, 14.735583812450457,
    8.341176470588236, 17.992936232915525,
    8.48235294117647, 21.460145050354196,
    8.614705882352942, 26.826957952797247,
    8.738235294117647, 32.75715098304728,
    8.897058823529413, 39.06939937054617,
    9.038235294117648, 43.42652736257701,
    9.23235294117647, 48.83998368582261,
    9.435294117647059, 53.65273145287729,
    9.602941176470587, 62.505519252739695,
    9.735294117647058, 73.6795455966163,
    9.894117647058824, 83.84349775221372,
    10.079411764705881, 94.29524558091273,
    10.238235294117647, 106.04988553128285,
    10.600000000000001, 143.9339570739062,
    10.75, 161.87645069182696,
    10.908823529411766, 195.35130938771198,
    11.05, 227.58459260747887,
])

FIRE_boxes = np.array([
    6.259703981655201, 4.928138751061449,
    6.2704607046070455, 8.93757115105424,
    6.723118615801543, 8.169493319132865,
    6.927746508234312, 7.139347751502866,
    6.835855743172816, 12.66015504518223,
    6.990202209714404, 14.486904919154643,
    7.075422138836773, 13.542764208288904,
    7.13545966228893, 14.325096656092072,
    6.963101938711695, 9.777861974998928,
    7.039983322910151, 9.668650319958088,
    7.563769022305607, 18.547811900706147,
    8.042276422764226, 17.933230746828492,
    8.07467166979362, 11.06375217222079,
    8.572149259954138, 17.732929513788157,
    8.992120075046904, 24.28655154971349,
    8.924286012090889, 28.104799268795613,
    9.148092557848654, 43.06717067640518,
    9.59328747133625, 52.717064564508966,
    9.63443819053575, 34.402132630397624,
    10.399791536376902, 121.03941869353282,
    10.67371273712737, 135.42764208288904,
    10.820512820512821, 202.91636467918352,
    10.940504481967896, 221.99411593691968,
    11.085720241817802, 217.06278248931105,
])


def plot_tully_fisher(
    rot_curve_file: str,
    run_num: int,
    output_dir: str = "Plots",
    filename_suffix: str = "",
    ax=None,
) -> str | None:
    """
    Load precomputed rotation-curve data from an HDF5 file and
    produce the Tully–Fisher plot.

    Parameters
    ----------
    rot_curve_file : str
        Path to the HDF5 file containing rotation curve and cumulative mass data.
    run_num : int
        Simulation run number for labeling and output filename.
    output_dir : str, optional
        Directory where the Tully–Fisher plot PNG will be saved.

    Returns
    -------
    str
        Path to the saved Tully–Fisher plot.
    """
    with h5py.File(rot_curve_file, "r") as f:
        cum_mass_stars_only = f["cum_mass_stars_only"][:]
        if "vrot_gas" in f:
            vrot_gas = float(f["vrot_gas"][()])
        else:
            vrot = f["vrot"][:]
            finite_vrot = vrot[np.isfinite(vrot)]
            vrot_gas = float(np.median(finite_vrot)) if finite_vrot.size else np.nan

    # Convert to stellar mass in Msun and characteristic rotation velocity
    M_stars_total = cum_mass_stars_only[-1] if cum_mass_stars_only.size else np.nan
    V_rot_med = vrot_gas

    # Unpack literature relations
    SAMI_x, SAMI_y = split_paired_array(SAMI, first_is_x=True)
    FIRE_boxes_x, FIRE_boxes_y = split_paired_array(FIRE_boxes, first_is_x=True)

    owns_figure = ax is None
    if owns_figure:
        fig2, axs2 = plt.subplots(figsize=(6, 6))
    else:
        fig2, axs2 = ax.figure, ax

    if np.isfinite(M_stars_total) and M_stars_total > 0 and np.isfinite(V_rot_med) and V_rot_med > 0:
        axs2.scatter(np.log10(M_stars_total), V_rot_med, label=f"run_{run_num}")
    axs2.scatter(FIRE_boxes_x, FIRE_boxes_y, label="El-Badry et al. 2018")
    axs2.plot(SAMI_x, SAMI_y, label="SAMI survey")
    axs2.set_yscale("log")
    axs2.set_xlabel(r"$\rm \log M_*~[M_\odot]$", size=15)
    axs2.set_ylabel(r"$\rm v_{rot,gas} [km/s] $", size=15)
    axs2.legend()
    axs2.set_title("Tully-Fisher", size=18)
    if owns_figure:
        fig2.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        outname = os.path.join(output_dir, f"run_{run_num}_TullyFisher{filename_suffix}.png")
        fig2.savefig(outname, bbox_inches="tight")
        plt.close(fig2)
    else:
        outname = None

    return outname


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_tully_fisher()."
    )
