"""
Test mass conservation in 2D histogram binning.

This script verifies that the 2D histogram function properly bins particle
data without losing or duplicating mass. It creates histograms with varying
numbers of bins and checks that the summed mass across all bins remains
constant and equals the total particle mass.

Usage:
    python test_mass_conservation.py test
    python test_mass_conservation.py DIRECTORY_NAME SNAP_NUM [PARTTYPE]
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from tests.generate_test_galaxy import create_test_galaxy_snapshot
from scripts.load_sim_data import load_particles


def test_mass_conservation(snap_path: str, parttype: int, nbins_list: list = None) -> dict:
    """
    Test mass conservation by creating 2D histograms with varying bin counts.

    Parameters
    ----------
    snap_path : str
        Path to the HDF5 snapshot file.
    parttype : int
        Particle type (0=gas, 1=dm, 4=stars).
    nbins_list : list, optional
        List of bin counts to test. Defaults to [32, 64, 128, 256, 512].

    Returns
    -------
    dict
        Dictionary with results including 'passed', 'total_mass', and 'histogram_sums'.
    """
    if nbins_list is None:
        nbins_list = [32, 64, 128, 256, 512]

    # Load particle data
    with h5py.File(snap_path, "r") as f:
        pos_key = f"PartType{parttype}/Coordinates"
        mass_key = f"PartType{parttype}/Masses"

        if pos_key not in f or mass_key not in f:
            return {
                "passed": False,
                "error": f"PartType{parttype} not found in snapshot",
                "total_mass": None,
                "histogram_sums": {},
                "histograms": {},
            }

        positions = f[pos_key][:]
        masses = f[mass_key][:]

    total_mass = np.sum(masses)

    # Test histograms with varying bin counts
    histogram_sums = {}
    histograms = {}
    tolerance = 1e-6  # Relative tolerance for floating point comparison

    for nbins in nbins_list:
        H, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1], bins=(nbins, nbins), weights=masses
        )
        hist_sum = np.sum(H)
        histogram_sums[nbins] = hist_sum
        histograms[nbins] = (H, xedges, yedges)

    # Check consistency
    all_sums = list(histogram_sums.values())
    ref_sum = all_sums[0]

    passed = True
    discrepancies = []

    # Check that all histogram sums match the total mass
    for nbins, h_sum in histogram_sums.items():
        # Check against reference (first histogram)
        if not np.isclose(h_sum, ref_sum, rtol=tolerance):
            passed = False
            discrepancies.append(
                f"  nbins={nbins}: sum={h_sum:.10e}, expected={ref_sum:.10e}, "
                f"diff={abs(h_sum - ref_sum):.2e}"
            )

        # Check against total mass
        if not np.isclose(h_sum, total_mass, rtol=tolerance):
            passed = False
            discrepancies.append(
                f"  nbins={nbins}: sum={h_sum:.10e}, total_mass={total_mass:.10e}, "
                f"diff={abs(h_sum - total_mass):.2e}"
            )

    return {
        "passed": passed,
        "total_mass": total_mass,
        "histogram_sums": histogram_sums,
        "histograms": histograms,
        "discrepancies": discrepancies,
        "parttype": parttype,
        "nbins_tested": nbins_list,
    }


def plot_histograms(result: dict, output_dir: str, prefix: str = "") -> list:
    """
    Create and save plots for each histogram result.

    Parameters
    ----------
    result : dict
        Result dictionary from test_mass_conservation().
    output_dir : str
        Directory to save plot files.
    prefix : str, optional
        Prefix for filename (e.g., "run_0_").

    Returns
    -------
    list
        List of saved plot file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    parttype_names = {0: "gas", 1: "dm", 4: "stars"}
    parttype_name = parttype_names.get(result["parttype"], f"type{result['parttype']}")

    saved_files = []
    parttype = result["parttype"]

    for nbins, (H, xedges, yedges) in result["histograms"].items():
        fig, ax = plt.subplots(figsize=(8, 7))

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(
            H.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            norm="log",
        )

        ax.set_xlabel("x (code units)")
        ax.set_ylabel("y (code units)")
        ax.set_title(
            f"Mass Distribution ({parttype_name}, {nbins} bins)\n"
            f"Total mass: {result['total_mass']:.6e}"
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Mass")

        fig.tight_layout()

        # Filename with bin count
        filename = f"mass_cons/{prefix}test_mass_conservation_{parttype_name}_{nbins}bins.png"
        output_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.join(output_dir,"mass_cons"),exist_ok = True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        saved_files.append(output_path)

    return saved_files


def run_mass_conservation_tests(
    snap_path: str, parttype: int = None, box_num: int = None, snapnum: int = None
) -> dict:
    """
    Run mass conservation tests for specified particle type(s).

    Parameters
    ----------
    snap_path : str
        Path to the HDF5 snapshot file.
    parttype : int, optional
        Specific particle type to test. If None, tests all types.
    box_num : int, optional
        Box number (for output directory naming).
    snapnum : int, optional
        Snapshot number (for output directory naming).

    Returns
    -------
    dict
        Dictionary with test results for each particle type.
    """
    parttype_names = {0: "Gas", 1: "DM", 4: "Stars"}
    parttypes_to_test = [parttype] if parttype is not None else [0, 1, 4]
    results = {}

    # Determine output directory
    if box_num is not None:
        output_dir = os.path.join("Plots", str(box_num))
        prefix = f"{box_num}_snap{snapnum:03d}_"
    else:
        output_dir = os.path.join("Plots", "test")
        prefix = ""

    os.makedirs(output_dir, exist_ok=True)

    for pt in parttypes_to_test:
        print(f"\nTesting {parttype_names[pt]} particles (PartType{pt})...")
        result = test_mass_conservation(snap_path, pt)
        results[pt] = result

        if result["passed"]:
            print(f"    PASSED: Mass conserved across all bin counts")
            print(f"    Total mass: {result['total_mass']:.6e}")
            for nbins, h_sum in result["histogram_sums"].items():
                print(f"      nbins={nbins:3d}: {h_sum:.6e}")

            # Generate plots
            plot_files = plot_histograms(result, output_dir, prefix=prefix)
            print(f"    Saved {len(plot_files)} plots to {output_dir}")
        else:
            print(f"    FAILED: Mass conservation check failed")
            print(f"    Total mass: {result['total_mass']:.6e}")
            for nbins, h_sum in result["histogram_sums"].items():
                print(f"      nbins={nbins:3d}: {h_sum:.6e}")
            if result.get("error"):
                print(f"    Error: {result['error']}")
            else:
                for disc in result.get("discrepancies", []):
                    print(disc)

    return results


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test mass conservation in 2D histogram binning.\n"
        "Usage: python test_mass_conservation.py test\n"
        "       python test_mass_conservation.py DIRECTORY_NAME SNAP_NUM [PARTTYPE]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "first_arg",
        help="Either 'test' for toy galaxy, or BOX_NUM for production data.",
    )
    parser.add_argument(
        "second_arg",
        nargs="?",
        help="SNAP_NUM for production data (omit for test mode).",
    )
    parser.add_argument(
        "parttype",
        nargs="?",
        type=int,
        help="Particle type to test (0=gas, 1=dm, 4=stars). If omitted, tests all types.",
    )

    args = parser.parse_args()

    # Determine mode based on arguments
    if args.first_arg == "test":
        args.mode = "test"
        args.directory_name = None
        args.snapnum = None
        args.parttype = args.parttype  # Can be None (all types) or a specific type
        if args.second_arg is not None:
            parser.error("Test mode takes no additional arguments.")
    else:
        args.directory_name = args.first_arg
        if args.second_arg is None:
            parser.error(
                "Production mode requires both DIRECTORY_NAME and SNAP_NUM.\n"
                "Usage: python test_mass_conservation.py DIRECTORY_NAME SNAP_NUM [PARTTYPE]"
            )
        try:
            args.snapnum = int(args.second_arg)
        except ValueError:
            parser.error("SNAP_NUM must be an integer.")
        args.mode = "production"

    return args


def main():
    args = _parse_args()

    print("\n" + "=" * 70)
    print("Mass Conservation Test Suite")
    print("=" * 70)

    if args.mode == "test":
        # Test mode
        script_dir = Path(__file__).resolve().parent
        test_data_dir = script_dir / "sim_data"
        snap_path = test_data_dir / "test_galaxy.hdf5"

        # Generate test snapshot if it doesn't exist
        if not snap_path.exists():
            print(f"\nGenerating test galaxy snapshot at {snap_path}...")
            create_test_galaxy_snapshot(
                output_dir=str(test_data_dir), filename="test_galaxy.hdf5"
            )
        else:
            print(f"\nUsing existing test galaxy snapshot at {snap_path}")

        results = run_mass_conservation_tests(str(snap_path), parttype=args.parttype)

    else:
        # Production mode
        print(f"\nLoading snapshot data for {args.first_arg}, snapshot {args.snapnum}...")
        
        # For production data, we load directly from the snapshot HDF5 files
        from scripts.load_sim_data import snapshot_base
        snap_path = os.path.join(snapshot_base, args.first_arg, f"snap_{args.snapnum:03d}.hdf5")
        
        if not os.path.exists(snap_path):
            print(f"Error: Snapshot file not found at {snap_path}")
            return 1

        results = run_mass_conservation_tests(
            snap_path,
            parttype=args.parttype,
            box_num=args.first_arg,
            snapnum=args.snapnum,
        )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_passed = all(r["passed"] for r in results.values())

    parttype_names = {0: "Gas", 1: "DM", 4: "Stars"}
    tested_types = list(results.keys())

    for pt in tested_types:
        status = " PASS" if results[pt]["passed"] else " FAIL"
        print(f"{status}: {parttype_names[pt]} (PartType{pt})")

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests PASSED! Mass conservation verified.")
        return 0
    else:
        print("Some tests FAILED. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
