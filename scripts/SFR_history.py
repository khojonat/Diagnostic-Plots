import os

import h5py
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from .helpers import code, halo_particle_bounds, loadHalos, snapshot_base, _normalize_run_dir


def _find_snapshot_file(data_dir: str, snapnum: int) -> str | None:
    """
    Locate a snapshot file for a given snapshot number, trying both
    snapdir_XXX/snap_XXX.hdf5 and snap_XXX.hdf5 in sim_path.
    Returns None if no file is found.
    """
    path = os.path.join(snapshot_base, _normalize_run_dir(data_dir))
    snapdir = os.path.join(path, f"snapdir_{snapnum:03d}")
    if os.path.isdir(snapdir):
        snapfile = os.path.join(snapdir, f"snap_{snapnum:03d}.hdf5")
        if os.path.exists(snapfile):
            return snapfile

    snapfile = os.path.join(path, f"snap_{snapnum:03d}.hdf5")
    if os.path.exists(snapfile):
        return snapfile

    return None


def plot_sfr_history(
    data_dir: str,
    max_snapnum: int,
    target: int,
    output_dir: str = "Plots",
    particle_data: dict | None = None,
    filename_suffix: str = "",
    ax=None,
) -> str | None:
    """
    Load the star formation history of the (global) system across snapshots
    and plot total SFR as a function of redshift.

    This function loops over all snapshot numbers from 0..max_snapnum,
    uses any snapshot files that exist, and for each:
      - reads Header/Redshift
      - sums PartType0/StarFormationRate to estimate the SFR

    Parameters
    ----------
    data_dir : str
        Simulation data directory for labeling and filenames.
    max_snapnum : int
        Highest snapshot number to consider (0..max_snapnum will be scanned).
    target : int
        Target halo index.
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
        snapfile = _find_snapshot_file(data_dir, snap)
        if snapfile is None:
            continue

        with h5py.File(snapfile, "r") as f:
            header = f["Header"]
            z = float(header.attrs.get("Redshift", 0.0))
            if code == "arepo":
                if snap == max_snapnum and particle_data is not None:
                    gas_data = particle_data["particles"].get(0, {})
                    if "StarFormationRate" not in gas_data:
                        continue
                    sfr = np.sum(gas_data["StarFormationRate"])
                elif "PartType0" in f and "StarFormationRate" in f["PartType0"]:
                    halo_length = loadHalos(data_dir, snap, 'GroupLenType')
                    start, end = halo_particle_bounds(halo_length, target, 0)
                    sfr_all = f["PartType0"]["StarFormationRate"][:]
                    sfr = np.sum(sfr_all[start:end])
                else:
                    continue

            elif code == "gizmo":
                if "Time" in header.attrs:
                    a_snap = float(header.attrs["Time"])
                else:
                    a_snap = 1.0 / (1.0 + float(header.attrs.get("Redshift", 0.0)))

                h = float(header.attrs.get("HubbleParam", 1.0))
                H0 = 100.0 * h * u.km / u.s / u.Mpc
                Om0 = float(header.attrs.get("Omega0", 0.3))
                cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
                z_snap = 1.0 / a_snap - 1.0
                t_snap = cosmo.age(z_snap).to(u.Myr).value

                if snap == max_snapnum and particle_data is not None:
                    star_data = particle_data["particles"].get(4, {})
                    if "StellarFormationTime" not in star_data or "Masses" not in star_data:
                        continue
                    star_ages = star_data["StellarFormationTime"]
                    star_masses = star_data["Masses"]
                elif "PartType4" in f and "StellarFormationTime" in f["PartType4"]:
                    halo_length = loadHalos(data_dir, snap, 'GroupLenType')
                    start, end = halo_particle_bounds(halo_length, target, 4)
                    star_ages = f["PartType4"]["StellarFormationTime"][start:end]
                    star_masses = f["PartType4"]["Masses"][start:end]
                else:
                    continue

                valid = (star_ages > 0) & (star_ages <= 1.0)
                t_form = np.full_like(star_ages, np.inf, dtype=float)
                if np.any(valid):
                    z_form = np.maximum(0.0, 1.0 / star_ages[valid] - 1.0)
                    t_form[valid] = cosmo.age(z_form).to(u.Myr).value

                dt = t_snap - t_form
                recent = (dt >= 0.0) & (dt <= 500.0)
                star_mass_recent = star_masses[recent]
                sfr = np.sum(star_mass_recent) * 1e10 / h / (500.0e6)
            else:
                raise ValueError(f"Unsupported code type: {code}")

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

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    ax.plot(redshifts, sfr_values, marker="o")

    ax.set_xlabel("Redshift",fontsize=15)
    ax.set_ylabel(r"Total SFR [$\rm M_\odot/yr$]",fontsize=15)
    ax.set_yscale("log")
    title = "Star Formation History"
    if data_dir is not None:
        title += f" (box {data_dir})"
    ax.set_title(title,fontsize=15)

    if owns_figure:
        fig.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        tag = f"_{data_dir}" if data_dir is not None else ""
        outname = os.path.join(output_dir, f"SFR_history{tag}{filename_suffix}.png")
        fig.savefig(outname, bbox_inches="tight")
        plt.close(fig)
    else:
        outname = None

    return outname


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_sfr_history()."
    )
