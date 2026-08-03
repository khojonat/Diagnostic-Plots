import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

from .helpers import load_particles, split_paired_array, identify_target_halo, loadHalos

# Literature comparison data 
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

Wang_etal = np.array([
    32917456.737695787, 30.406899326507,
    50087793.86869629, 74.65605128970535,
    75279305.4003894, 208.3941142745337,
    105062497.67666402, 431.1961977256241,
    114546344.25494413, 534.015889536362,
    143052673.447375, 784.7599703514622,
    183119496.03857014, 1203.6365190931167,
    261957334.68653345, 2386.211741394953,
    479694951.11405057, 7572.808349202965,
    1225947436.5447083, 41907.429363976524,
    2668538411.0996227, 171907.22018585782,
    5808648102.878435, 768155.0966474186,
    12800844026.583998, 3738999.5936589604,
    22037561580.36261, 11369138.422623834,
    42398046272.63869, 41020702.723806135,
    92288471480.4732, 191308270.76154342])


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


def _get_hubble_param(run_num: int, snapnum: int) -> float:
    """
    Read HubbleParam from the snapshot header.
    """
    from .helpers import snapshot_base, _normalize_run_dir
    snapfile = os.path.join(snapshot_base, _normalize_run_dir(run_num), f'snap_{snapnum:03d}.hdf5')
    with h5py.File(snapfile, "r") as f:
        return float(f["Header"].attrs["HubbleParam"])


def _compute_total_masses(run_num: int,
                          snapnum: int = None,
                          target: int = None,
                          verbose: bool = True,
                          particle_data: dict | None = None):
    """
    Use load_particles to get stellar and dark matter masses and
    return total masses in physical units [Msun].
    """
    if snapnum is None:
        raise ValueError("snapnum must be provided")
    if target is None:
        raise ValueError("target must be provided")

    if particle_data is not None:
        h = float(particle_data["header"].get("HubbleParam", 1.0))
        star_masses = particle_data["particles"][4]["Masses"] * 1e10 / h
        dm_masses = particle_data["particles"][1]["Masses"] * 1e10 / h
        return np.sum(star_masses), np.sum(dm_masses)

    # Identify target halo
    # target = identify_target_halo(run_num, snapnum)
    halo_length = loadHalos(run_num, snapnum, 'GroupLenType')
    # Load particle masses in code units (1e10 Msun / h for AREPO/Illustris)
    star_data = load_particles(run_num, "stars", ["Masses"],
                               snapnum=snapnum,
                               verbose=verbose)
    dm_data = load_particles(run_num, "dm", ["Masses"],
                             snapnum=snapnum,
                             verbose=verbose)

    h = _get_hubble_param(run_num, snapnum)

    # Slice to halo particles
    start_star = np.sum(halo_length[:target, 4])
    end_star = start_star + halo_length[target, 4]
    star_masses_code = star_data["Masses"][start_star:end_star]

    start_dm = np.sum(halo_length[:target, 1])
    end_dm = start_dm + halo_length[target, 1]
    dm_masses_code = dm_data["Masses"][start_dm:end_dm]

    # Convert to physical masses
    star_masses = star_masses_code * 1e10 / h
    dm_masses = dm_masses_code * 1e10 / h

    return np.sum(star_masses), np.sum(dm_masses)


def plot_stellar_halo_mass(run_num: int,
                           snapnum: int,
                           target: int,
                           min_mass: float,
                           max_mass: float,
                           output_dir: str = "Plots",
                           verbose: bool = True,
                           particle_data: dict | None = None,
                           filename_suffix: str = "",
                           ax=None) -> str | None:
    """
    Make the stellar–halo mass plot using simulation data loaded
    via helpers.py plus the literature comparison arrays.

    Parameters
    ----------
    run_num : int
        Simulation run number for labeling and output filename.
    snapnum : int
        Snapshot number to load.
    target : int
        Target halo index.
    output_dir : str, optional
        Directory where the plot PNG will be written.
    verbose : bool, optional
        If True, prints a short message with the computed masses.
    """
    total_stellar, total_dm = _compute_total_masses(
        run_num,
        snapnum=snapnum,
        target=target,
        verbose=verbose,
        particle_data=particle_data,
    )

    if verbose:
        print(f"Total stellar mass: {total_stellar:.3e} Msun")
        print(f"Total halo (DM) mass: {total_dm:.3e} Msun")

    # Literature curves unpacked using the generic helper in helpers.py
    # Zacharegkas: [x0, y0, x1, y1, ...]
    Z25_x, Z25_y = split_paired_array(Zacharegkas_etal, first_is_x=True)
    # Read et al.: [y0, x0, y1, x1, ...]
    R17_x, R17_y = split_paired_array(Read_etal, first_is_x=False)
    # Wang et al.: [x0, y0, x1, y1, ...]
    W21_x, W21_y = split_paired_array(Wang_etal, first_is_x=True)

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    if np.isfinite(total_dm) and np.isfinite(total_stellar) and total_dm > 0 and total_stellar > 0:
        ax.scatter(np.log10(total_dm), np.log10(total_stellar), label=f"run_{run_num}")

    ax.vlines([min_mass, max_mass], 2, 12,
              color="red", ls="--", label="Min and Max")

    # ax.scatter(np.log10(R17_x), np.log10(R17_y), label="Read et al. 2017")
    ax.plot(Z25_y, Z25_x, label="Zacharegkas et al. 2025")
    ax.plot(np.log10(W21_x), np.log10(W21_y), label="Wang et al. 2021")

    ax.set_ylabel(r"Stellar mass $[M_\odot]$", size=15)
    ax.set_xlabel(r"Halo mass $[M_\odot]$", size=15)
    ax.legend()

    if owns_figure:
        os.makedirs(output_dir, exist_ok=True)
        outname = os.path.join(output_dir, f"Halo_Stellar_mass_run_{run_num}{filename_suffix}.png")
        fig.savefig(outname, bbox_inches="tight")
        plt.close(fig)
    else:
        outname = None

    return outname


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_stellar_halo_mass()."
    )
