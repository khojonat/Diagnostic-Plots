"""
Test particle ordering independence in 2D histogram binning.

This script verifies that particle indexing and relative position calculations
are done correctly by randomizing particle order and confirming that the resulting
histograms are identical. This tests that the code doesn't depend on particle
ordering but only on particle positions and masses.

Usage:
    python test_particle_ordering.py test [NBINS]
    python test_particle_ordering.py BOX_NUM SNAP_NUM [NBINS]
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

from generate_test_galaxy import create_test_galaxy_snapshot


def create_histogram(positions: np.ndarray, masses: np.ndarray, nbins: int = 256) -> tuple:
    """
    Create a 2D histogram from particle data.

    Parameters
    ----------
    positions : np.ndarray
        Particle positions (N, 3).
    masses : np.ndarray
        Particle masses (N,).
    nbins : int, optional
        Number of bins per axis.

    Returns
    -------
    tuple
        (histogram, xedges, yedges)
    """
    H, xedges, yedges = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=(nbins, nbins), weights=masses
    )
    return H, xedges, yedges


def test_particle_ordering(snap_path: str, parttype: int, nbins: int = 256) -> dict:
    """
    Test that particle ordering does not affect histogram results.

    Parameters
    ----------
    snap_path : str
        Path to the HDF5 snapshot file.
    parttype : int
        Particle type (0=gas, 1=dm, 4=stars).
    nbins : int, optional
        Number of bins per axis (default: 256).

    Returns
    -------
    dict
        Dictionary with test results including histograms and comparison metrics.
    """
    # Load particle data
    with h5py.File(snap_path, "r") as f:
        pos_key = f"PartType{parttype}/Coordinates"
        mass_key = f"PartType{parttype}/Masses"

        if pos_key not in f or mass_key not in f:
            return {
                "passed": False,
                "error": f"PartType{parttype} not found in snapshot",
                "num_particles": 0,
                "nbins": nbins,
            }

        positions_orig = f[pos_key][:].copy()
        masses_orig = f[mass_key][:].copy()

    num_particles = len(masses_orig)

    # Create histogram with original ordering
    H_orig, xedges, yedges = create_histogram(positions_orig, masses_orig, nbins)

    # Randomize particle order
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    random_indices = rng.permutation(num_particles)
    positions_shuffled = positions_orig[random_indices]
    masses_shuffled = masses_orig[random_indices]

    # Create histogram with shuffled ordering
    H_shuffled, _, _ = create_histogram(positions_shuffled, masses_shuffled, nbins)

    # Compare histograms
    tolerance = 1e-10  # Very strict tolerance for exact comparison
    histograms_match = np.allclose(H_orig, H_shuffled, rtol=tolerance, atol=tolerance)

    # Compute statistics for diagnostics
    max_diff = np.max(np.abs(H_orig - H_shuffled))
    mean_diff = np.mean(np.abs(H_orig - H_shuffled))
    rel_diff = max_diff / (np.max(H_orig) + 1e-15)  # Avoid division by zero

    return {
        "passed": histograms_match,
        "num_particles": num_particles,
        "nbins": nbins,
        "parttype": parttype,
        "histogram_original": H_orig,
        "histogram_shuffled": H_shuffled,
        "xedges": xedges,
        "yedges": yedges,
        "max_difference": max_diff,
        "mean_difference": mean_diff,
        "relative_difference": rel_diff,
    }


def plot_ordering_comparison(
    result: dict, output_dir: str, prefix: str = ""
) -> list:
    """
    Create and save comparison plots for original vs shuffled ordering.

    Parameters
    ----------
    result : dict
        Result dictionary from test_particle_ordering().
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

    # Create side-by-side comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    H_orig = result["histogram_original"]
    H_shuffled = result["histogram_shuffled"]
    xedges = result["xedges"]
    yedges = result["yedges"]
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Original histogram
    im0 = axes[0].imshow(
        H_orig.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="viridis",
        norm="log",
    )
    axes[0].set_xlabel("x (code units)")
    axes[0].set_ylabel("y (code units)")
    axes[0].set_title("Original Ordering")
    fig.colorbar(im0, ax=axes[0], label="Mass")

    # Shuffled histogram
    im1 = axes[1].imshow(
        H_shuffled.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="viridis",
        norm="log",
    )
    axes[1].set_xlabel("x (code units)")
    axes[1].set_ylabel("y (code units)")
    axes[1].set_title("Shuffled Ordering")
    fig.colorbar(im1, ax=axes[1], label="Mass")

    # Difference (absolute)
    diff = np.abs(H_orig - H_shuffled)
    # Avoid log(0) by adding small value
    diff_plot = np.where(diff > 0, diff, 1e-15)
    im2 = axes[2].imshow(
        diff_plot.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="hot",
        norm="log",
    )
    axes[2].set_xlabel("x (code units)")
    axes[2].set_ylabel("y (code units)")
    axes[2].set_title("Absolute Difference")
    fig.colorbar(im2, ax=axes[2], label="Difference")

    fig.suptitle(
        f"Particle Ordering Test - {parttype_name.upper()} ({result['nbins']} bins)\n"
        f"N={result['num_particles']:,} | Max diff: {result['max_difference']:.2e} | "
        f"Rel diff: {result['relative_difference']:.2e}",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    filename = f"{prefix}test_ordering_{parttype_name}_{result['nbins']}bins.png"
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    saved_files.append(output_path)

    return saved_files


def run_ordering_tests(
    snap_path: str, parttype: int = None, nbins: int = 256, 
    box_num: int = None, snapnum: int = None
) -> dict:
    """
    Run particle ordering tests for specified particle type(s).

    Parameters
    ----------
    snap_path : str
        Path to the HDF5 snapshot file.
    parttype : int, optional
        Specific particle type to test. If None, tests all types.
    nbins : int, optional
        Number of bins per axis (default: 256).
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
        output_dir = os.path.join("Plots", f"run_{box_num}")
        prefix = f"run_{box_num}_snap{snapnum:03d}_"
    else:
        output_dir = os.path.join("Plots", "test")
        prefix = ""

    os.makedirs(output_dir, exist_ok=True)

    for pt in parttypes_to_test:
        print(f"\nTesting {parttype_names[pt]} particles (PartType{pt})...")
        result = test_particle_ordering(snap_path, pt, nbins=nbins)
        results[pt] = result

        if result.get("error"):
            print(f"  ✗ ERROR: {result['error']}")
            continue

        if result["passed"]:
            print(f"  ✓ PASSED: Histograms are identical regardless of particle order")
            print(f"    Particles: {result['num_particles']:,}")
            print(f"    Bins: {nbins}×{nbins}")
            print(f"    Max difference: {result['max_difference']:.2e}")
            print(f"    Mean difference: {result['mean_difference']:.2e}")
            print(f"    Relative difference: {result['relative_difference']:.2e}")

            # Generate comparison plot
            plot_files = plot_ordering_comparison(result, output_dir, prefix=prefix)
            print(f"    Saved comparison plot to {output_dir}")
        else:
            print(f"  ✗ FAILED: Histograms differ when particle order changes")
            print(f"    Particles: {result['num_particles']:,}")
            print(f"    Bins: {nbins}×{nbins}")
            print(f"    Max difference: {result['max_difference']:.2e}")
            print(f"    Mean difference: {result['mean_difference']:.2e}")
            print(f"    Relative difference: {result['relative_difference']:.2e}")

    return results


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test particle ordering independence in 2D histogram binning.\n"
        "Usage: python test_particle_ordering.py test [NBINS]\n"
        "       python test_particle_ordering.py BOX_NUM SNAP_NUM [NBINS]",
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
        "nbins",
        nargs="?",
        type=int,
        default=256,
        help="Number of bins per axis (default: 256).",
    )

    args = parser.parse_args()

    # Determine mode based on arguments
    if args.first_arg == "test":
        args.mode = "test"
        args.box_num = None
        args.snapnum = None
        if args.second_arg is not None:
            try:
                args.nbins = int(args.second_arg)
            except ValueError:
                parser.error("For test mode, provide optional NBINS as integer.")
    else:
        try:
            args.box_num = int(args.first_arg)
            if args.second_arg is None:
                parser.error(
                    "Production mode requires both BOX_NUM and SNAP_NUM.\n"
                    "Usage: python test_particle_ordering.py BOX_NUM SNAP_NUM [NBINS]"
                )
            args.snapnum = int(args.second_arg)
            args.mode = "production"
        except ValueError:
            parser.error(
                f"Invalid argument '{args.first_arg}'. "
                "Use 'test' or provide BOX_NUM SNAP_NUM."
            )

    return args


def main():
    args = _parse_args()

    print("\n" + "=" * 70)
    print("Particle Ordering Independence Test Suite")
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

        results = run_ordering_tests(str(snap_path), nbins=args.nbins)

    else:
        # Production mode
        print(f"\nLoading snapshot data for run_{args.box_num}, snapshot {args.snapnum}...")

        from load_sim_data import snapshot_base
        snap_dir = os.path.join(snapshot_base, f"run_{args.box_num}", f"snapdir_{args.snapnum:03d}")
        snap_path = os.path.join(snap_dir, f"snap_{args.snapnum:03d}.hdf5")

        if not os.path.exists(snap_path):
            print(f"Error: Snapshot file not found at {snap_path}")
            return 1

        results = run_ordering_tests(
            snap_path,
            nbins=args.nbins,
            box_num=args.box_num,
            snapnum=args.snapnum,
        )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_passed = all(
        r.get("passed", False) for r in results.values() if "error" not in r
    )

    parttype_names = {0: "Gas", 1: "DM", 4: "Stars"}
    tested_types = list(results.keys())

    for pt in tested_types:
        if "error" in results[pt]:
            status = "✗ ERROR"
            print(f"{status}: {parttype_names[pt]} (PartType{pt}) - {results[pt]['error']}")
        else:
            status = "✓ PASS" if results[pt]["passed"] else "✗ FAIL"
            print(f"{status}: {parttype_names[pt]} (PartType{pt})")

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests PASSED! Particle ordering is correctly handled.")
        return 0
    else:
        print("Some tests FAILED. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
