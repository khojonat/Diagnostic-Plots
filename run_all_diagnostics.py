import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.Stellar_Halo_mass import plot_stellar_halo_mass
from scripts.Tully_Fisher import plot_tully_fisher
from scripts.rotation_curve import plot_rotation_curve
from scripts.density_2d import plot_2d_hist
from scripts.SFR_history import plot_sfr_history
from scripts.Kennicutt_Schmidt import plot_kennicutt_schmidt
from scripts.helpers import compute_rotation_curve_and_save, plot_dir, unit_mass, unit_distance


def read_plot_flags(params_path: Path) -> dict:
    flags = {}
    if not params_path.exists():
        raise FileNotFoundError(f"Sim_params.txt not found at {params_path}")

    with params_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().split()[0]

            try:
                flags[key] = int(value)
            except ValueError:
                continue

    return flags

def _component_density(parttype_str: str, label: str, filename: str,output_dir,data_dir,snapnum,target):
    '''
    Helper to make 2d hist plots
    '''
    output_path = os.path.join(output_dir, filename)
    return plot_2d_hist(
        data_dir=data_dir,
        snapnum=snapnum,
        parttype={"dm": 1, "gas": 0, "stars": 4}[parttype_str],
        target=target,
        output_path=output_path,
        xlabel="x",
        ylabel="y",
        title=label,
    )

def run_all_diagnostics(data_dir: str, snapnum: int):
    """
    Run all diagnostic plots for a given simulation data directory.
    Plot production is controlled by flags in Sim_params.txt.
    """
    from scripts.helpers import identify_target_halo

    params_path = Path(__file__).resolve().parent / "Sim_params.txt"
    plot_flags = read_plot_flags(params_path)

    # Identify target halo once
    target, min_mass, max_mass = identify_target_halo(data_dir, snapnum)

    output_dir = os.path.join(plot_dir, data_dir)
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    if plot_flags.get("stellar_halo", 0):
        results["stellar_halo_mass_plot"] = plot_stellar_halo_mass(
            data_dir=data_dir,
            snapnum=snapnum,
            target=target,
            min_mass=min_mass,
            max_mass=max_mass,
            output_dir=output_dir,
        )

    rot_needed = plot_flags.get("rot_curve", 0) or plot_flags.get("Tully_Fisher", 0)
    rot_curve_file = None
    if rot_needed:
        rot_curve_file = compute_rotation_curve_and_save(
            data_dir=data_dir,
            snapnum=snapnum,
            target=target,
            output_dir=os.path.join(output_dir, "sim_data"),
        )
        results["rotation_curve_data"] = rot_curve_file

        if plot_flags.get("rot_curve", 0):
            results["rotation_curve_plot"] = plot_rotation_curve(
                rot_curve_file=rot_curve_file,
                data_dir=data_dir,
                output_dir=output_dir,
            )

        if plot_flags.get("Tully_Fisher", 0):
            results["tully_fisher_plot"] = plot_tully_fisher(
                rot_curve_file=rot_curve_file,
                data_dir=data_dir,
                output_dir=output_dir,
            )

    if plot_flags.get("DM_density", 0):
        results["dm_density_map"] = _component_density(
            "dm", f"DM density ({data_dir})", f"FIRE2_MW_{data_dir}_dm_density.png", output_dir, data_dir, snapnum, target
        )

    if plot_flags.get("gas_density", 0):
        results["gas_density_map"] = _component_density(
            "gas", f"Gas density ({data_dir})", f"FIRE2_MW_{data_dir}_gas_density.png", output_dir, data_dir, snapnum, target
        )

    if plot_flags.get("star_density", 0):
        results["stellar_density_map"] = _component_density(
            "stars",
            f"Stellar density ({data_dir})",
            f"FIRE2_MW_{data_dir}_stellar_density.png", output_dir, data_dir, snapnum, target
        )

    if plot_flags.get("SFR_history", 0):
        results["sfr_history_plot"] = plot_sfr_history(
            data_dir=data_dir,
            max_snapnum=snapnum,
            target=target,
            output_dir=output_dir,
        )

    if plot_flags.get("Ken_Schmidt", 0):
        results["kennicutt_schmidt_plot"] = plot_kennicutt_schmidt(
            data_dir=data_dir,
            snapnum=snapnum,
            target=target,
            output_dir=output_dir,
        )

    return results


def run_test_diagnostics():
    """
    Run diagnostic plots on the toy test galaxy.
    Uses test_galaxy.hdf5 and outputs to Plots/test/.
    """
    import h5py
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from tests.generate_test_galaxy import create_test_galaxy_snapshot

    params_path = Path(__file__).resolve().parent / "Sim_params.txt"
    plot_flags = read_plot_flags(params_path)

    # Generate test galaxy if needed
    data_dir = "test_data"
    snap_path = os.path.join(data_dir, "test_galaxy.hdf5")

    if not os.path.exists(snap_path):
        print(f"Generating test galaxy snapshot at {snap_path}...")
        create_test_galaxy_snapshot(
            output_dir=str(data_dir), filename="test_galaxy.hdf5"
        )
    else:
        print(f"Using existing test galaxy snapshot at {snap_path}")

    output_dir = os.path.join(plot_dir, "test")
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    snapnum = 'test'
    target = 0

    if plot_flags.get("DM_density", 0):
        results["dm_density_map"] = _component_density(
            "dm", f"DM density ({data_dir})", f"{data_dir}_dm_density.png", output_dir, data_dir, snapnum, target
        )

    if plot_flags.get("gas_density", 0):
        results["gas_density_map"] = _component_density(
            "gas", f"Gas density ({data_dir})", f"{data_dir}_gas_density.png", output_dir, data_dir, snapnum, target
        )

    if plot_flags.get("star_density", 0):
        results["stellar_density_map"] = _component_density(
            "stars",
            f"Stellar density ({data_dir})",
            f"{data_dir}_stellar_density.png", output_dir, data_dir, snapnum, target
        )

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run diagnostic plots for simulation or test data.\n"
        "Usage: python run_all_diagnostics.py test\n"
        "       python run_all_diagnostics.py DATA_DIR SNAP_NUM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "first_arg",
        help="Either 'test' for toy galaxy, or DATA_DIR for production data.",
    )
    parser.add_argument(
        "second_arg",
        nargs="?",
        help="SNAP_NUM for production data (omit for test mode).",
    )

    args = parser.parse_args()

    # Determine mode based on arguments
    if args.first_arg == "test":
        args.mode = "test"
        args.data_dir = None
        args.snapnum = None
        if args.second_arg is not None:
            parser.error("Test mode takes no additional arguments.")
    else:
        args.data_dir = args.first_arg
        if args.second_arg is None:
            parser.error(
                "Production mode requires both DATA_DIR and SNAP_NUM.\n"
                "Usage: python run_all_diagnostics.py DATA_DIR SNAP_NUM"
            )
        try:
            args.snapnum = int(args.second_arg)
        except ValueError:
            parser.error("SNAP_NUM must be an integer.")
        args.mode = "production"

    return args


def main() -> None:
    args = _parse_args()

    if args.mode == "test":
        results = run_test_diagnostics()
    elif args.mode == "production":
        results = run_all_diagnostics(
            data_dir=args.data_dir,
            snapnum=args.snapnum,
        )

    # Print a short summary of generated outputs
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

