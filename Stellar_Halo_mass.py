import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

from load_sim_data import load_particles, split_paired_array


# Literature comparison data (kept as provided)
Zacharegkas_etal = np.array([
    9.078366290018831, 11.157768361581923,
    9.541596045197739, 11.370409604519775,
    9.827071563088511, 11.503283898305085,
    10.004802259887004, 11.596151129943504,
    10.225612052730696, 11.71550141242938,
    10.47328154425612, 11.887429378531074,
    10.710122410546138, 12.085487288135594,
    10.871539548022598, 12.256850282485876,
    10.812358757062146, 12.190960451977402,
    10.957603578154425, 12.362217514124294,
    11.070550847457627, 12.507062146892656,
    11.194209039548022, 12.691278248587572,
    11.307085687382298, 12.87542372881356,
    11.398469868173256, 13.02012711864407,
    11.500564971751412, 13.20420197740113,
    11.62415254237288, 13.427718926553673,
    11.742349340866289, 13.651200564971752,
    11.882015065913372, 13.927224576271188,
    12.010946327683616, 14.176977401129944,
    12.070032956685498, 14.295268361581922,
])

Read_etal = np.array([
    669510782.5337304, 674836.3585488194,
    1687612475.7881384, 4074958.389011232,
    6135907273.413163, 7935828.924687643,
    5823063353.505637, 16166741.135822779,
    8399287059.458424, 16633501.890130416,
    11103363181.676321, 33949857.814488225,
    12115276586.28585, 54596109.14005392,
    20092330025.65046, 77139081.2851754,
    23507807256.003456, 71336807.72757018,
    20805675382.17163, 59273065.76269066,
    12766169490.574062, 8625118.959802005,
    333217094.1244796, 477099.1086006026,
    489095806.71506673, 134879.0452151551,
])


def _find_snapshot_file(path: str, snapnum: int) -> str:
    """
    Reproduce the snapshot path logic from load_sim_data to read header info.
    """
    snapdir = os.path.join(path, f"snapdir_{snapnum:03d}")
    if os.path.isdir(snapdir):
        snapfile = os.path.join(snapdir, f"snap_{snapnum:03d}.hdf5")
    else:
        snapfile = os.path.join(path, f"snap_{snapnum:03d}.hdf5")
    return snapfile


def _get_hubble_param(path: str, snapnum: int) -> float:
    """
    Read HubbleParam from the snapshot header.
    """
    snapfile = _find_snapshot_file(path, snapnum)
    with h5py.File(snapfile, "r") as f:
        return float(f["Header"].attrs["HubbleParam"])


def _compute_total_masses(sim_path: str,
                          snapnum: int = None,
                          redshift: float = None,
                          verbose: bool = True):
    """
    Use load_particles to get stellar and dark matter masses and
    return total masses in physical units [Msun].
    """
    if snapnum is None and redshift is None:
        raise ValueError("Must supply either snapnum or redshift")

    # Load particle masses in code units (1e10 Msun / h for AREPO/Illustris)
    star_data = load_particles(sim_path, "stars", ["Masses"],
                               redshift=redshift, snapnum=snapnum,
                               verbose=verbose)
    dm_data = load_particles(sim_path, "dm", ["Masses"],
                             redshift=redshift, snapnum=snapnum,
                             verbose=verbose)

    # Resolve the actual snapshot number (if we came in via redshift)
    if snapnum is None:
        # load_particles already printed which snapshot was loaded;
        # here we simply assume the target is uniquely determined by redshift
        # and use a naive search equivalent to load_sim_data.find_snapshot_from_redshift.
        raise ValueError("For now, please call with snapnum set explicitly.")

    h = _get_hubble_param(sim_path, snapnum)

    star_masses_code = star_data["Masses"]
    dm_masses_code = dm_data["Masses"]

    # Convert from 1e10 Msun / h to Msun
    factor = 1.0e10 / h
    total_stellar = np.sum(star_masses_code) * factor
    total_dm = np.sum(dm_masses_code) * factor

    return total_stellar, total_dm


def plot_stellar_halo_mass(sim_path: str,
                           snapnum: int,
                           box_num: int | None = None,
                           output_dir: str = "Plots",
                           verbose: bool = True):
    """
    Make the stellar–halo mass plot using simulation data loaded
    via load_sim_data.py plus the literature comparison arrays.

    Parameters
    ----------
    sim_path : str
        Path to the simulation snapshot directory (as used by load_sim_data.py).
    snapnum : int
        Snapshot number to load.
    box_num : int, optional
        Box identifier for labeling and output filename.
    output_dir : str, optional
        Directory where the plot PNG will be written.
    verbose : bool, optional
        If True, prints a short message with the computed masses.
    """
    total_stellar, total_dm = _compute_total_masses(
        sim_path, snapnum=snapnum, redshift=None, verbose=verbose
    )

    if verbose:
        print(f"Total stellar mass: {total_stellar:.3e} Msun")
        print(f"Total halo (DM) mass: {total_dm:.3e} Msun")

    # Literature curves unpacked using the generic helper in load_sim_data
    # Zacharegkas: [x0, y0, x1, y1, ...]
    Z25_x, Z25_y = split_paired_array(Zacharegkas_etal, first_is_x=True)
    # Read et al.: [y0, x0, y1, x1, ...]
    R17_x, R17_y = split_paired_array(Read_etal, first_is_x=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    label = f"Box {box_num}" if box_num is not None else "Simulation"
    ax.scatter(np.log10(total_dm), np.log10(total_stellar), label=label)

    ax.vlines(np.log10([5e11, 2.5e12]), 9, 12,
              color="red", ls="--", label="Min and Max")

    # Example: uncomment if you want Read et al. points
    # ax.scatter(np.log10(R17_x), np.log10(R17_y), label="Read et al. 2017")
    ax.plot(Z25_x, Z25_y, label="Zacharegkas et al. 2025")

    ax.set_ylabel(r"Stellar mass $[M_\odot]$", size=15)
    ax.set_xlabel(r"Halo mass $[M_\odot]$", size=15)
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    tag = f"Box_{box_num}_" if box_num is not None else ""
    outname = os.path.join(output_dir, f"{tag}Halo_Stellar_mass_alt.png")
    fig.savefig(outname, bbox_inches="tight")
    plt.close(fig)

    return outname


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_stellar_halo_mass()."
    )
