import os

import h5py
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from load_sim_data import split_paired_array


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
    11.067647058823528, 222.29964825261956,
    10.908823529411766, 224.92659888140892,
    10.802941176470588, 204.7502731435764,
    10.652941176470588, 137.32673733736806,
    10.370588235294118, 122.1053502998154,
    9.58529411764706, 53.02611335911987,
    9.629411764705882, 34.73892112083116,
    9.126470588235293, 43.42652736257701,
    8.91470588235294, 28.11768697974231,
    8.967647058823529, 24.42053094548651,
    8.561764705882354, 17.57510624854791,
    8.08529411764706, 10.857111194022039,
    8.041176470588235, 17.78279410038923,
    7.555882352941177, 18.63838005858641,
])


def plot_tully_fisher(
    rot_curve_file: str,
    box_num: int,
    output_dir: str = "Plots",
) -> str:
    """
    Load precomputed rotation-curve data from an HDF5 file and
    produce the Tully–Fisher plot.

    Parameters
    ----------
    rot_curve_file : str
        Path to the HDF5 file containing rotation curve and cumulative mass data.
    box_num : int
        Box identifier for labeling and output filename.
    output_dir : str, optional
        Directory where the Tully–Fisher plot PNG will be saved.

    Returns
    -------
    str
        Path to the saved Tully–Fisher plot.
    """
    with h5py.File(rot_curve_file, "r") as f:
        cum_mass_stars_only = f["cum_mass_stars_only"][:]
        vrot = f["vrot"][:]

    # Convert to stellar mass in Msun and characteristic rotation velocity
    M_stars_total = cum_mass_stars_only[-1]  # already in Msun in compiled file
    V_rot_med = float(np.median(vrot))

    # Unpack literature relations
    SAMI_x, SAMI_y = split_paired_array(SAMI, first_is_x=True)
    FIRE_boxes_x, FIRE_boxes_y = split_paired_array(FIRE_boxes, first_is_x=True)

    fig2, axs2 = plt.subplots(figsize=(6, 6))

    axs2.scatter(np.log10(M_stars_total), V_rot_med, label=f"FIRE box {box_num}")
    axs2.scatter(FIRE_boxes_x, FIRE_boxes_y, label="El-Badry et al. 2018")
    axs2.plot(SAMI_x, SAMI_y, label="SAMI survey")
    axs2.set_yscale("log")
    axs2.set_xlabel(r"$\rm \log M_*~[M_\odot]$", size=15)
    axs2.set_ylabel(r"$\rm v_{rot,gas} [km/s] $", size=15)
    axs2.legend()
    axs2.set_title("Tully-Fisher", size=18)
    fig2.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outname = os.path.join(output_dir, f"FIRE2_MW_run{box_num}_TF.png")
    fig2.savefig(outname, bbox_inches="tight")
    plt.close(fig2)

    return outname


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_tully_fisher()."
    )

