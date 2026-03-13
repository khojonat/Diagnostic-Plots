import os

import h5py
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


def _find_snapshot_file(sim_path: str, snapnum: int) -> str | None:
    """
    Locate a snapshot file for a given snapshot number, trying both
    snapdir_XXX/snap_XXX.hdf5 and snap_XXX.hdf5 in sim_path.
    Returns None if no file is found.
    """
    snapdir = os.path.join(sim_path, f"snapdir_{snapnum:03d}")
    if os.path.isdir(snapdir):
        snapfile = os.path.join(snapdir, f"snap_{snapnum:03d}.hdf5")
        if os.path.exists(snapfile):
            return snapfile

    snapfile = os.path.join(sim_path, f"snap_{snapnum:03d}.hdf5")
    if os.path.exists(snapfile):
        return snapfile

    return None


def plot_sfr_history(
    sim_path: str,
    max_snapnum: int,
    box_num: int | None = None,
    output_dir: str = "Plots",
) -> str:
    """
    Load the star formation history of the (global) system across snapshots
    and plot total SFR as a function of redshift.

    This function loops over all snapshot numbers from 0..max_snapnum,
    uses any snapshot files that exist, and for each:
      - reads Header/Redshift
      - sums PartType0/StarFormationRate to estimate the SFR

    Parameters
    ----------
    sim_path : str
        Path to the simulation snapshot directory.
    max_snapnum : int
        Highest snapshot number to consider (0..max_snapnum will be scanned).
    box_num : int, optional
        Box identifier for labeling and filenames.
    output_dir : str, optional
        Directory where the SFR history plot PNG will be saved.

    Returns
    -------
    str
        Path to the saved SFR history plot.
    """
    redshifts = []
    sfr_values = []

    for snap in range(max_snapnum + 1):
        snapfile = _find_snapshot_file(sim_path, snap)
        if snapfile is None:
            continue

        with h5py.File(snapfile, "r") as f:
            header = f["Header"]
            z = float(header.attrs.get("Redshift", 0.0))
            if "PartType0" not in f or "StarFormationRate" not in f["PartType0"]:
                continue

            sfr = np.sum(f["PartType0"]["StarFormationRate"][:])

        redshifts.append(z)
        sfr_values.append(sfr)

    if not redshifts:
        raise RuntimeError("No snapshots with gas StarFormationRate found for SFR history.")

    redshifts = np.array(redshifts)
    sfr_values = np.array(sfr_values)

    # Sort by redshift
    order = np.argsort(redshifts)
    redshifts = redshifts[order]
    sfr_values = sfr_values[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(redshifts, sfr_values, marker="o")

    ax.set_xlabel("Redshift")
    ax.set_ylabel("Total SFR")
    title = "Star Formation History"
    if box_num is not None:
        title += f" (box {box_num})"
    ax.set_title(title)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    tag = f"_box{box_num}" if box_num is not None else ""
    outname = os.path.join(output_dir, f"SFR_history{tag}.png")
    fig.savefig(outname, bbox_inches="tight")
    plt.close(fig)

    return outname


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_sfr_history()."
    )

