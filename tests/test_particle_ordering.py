"""
Run diagnostics after randomizing the target halo's particle ordering.

Snapshot and FOF/Subfind files are only read. The target halo particles are
copied into memory, shuffled there, and passed into the normal diagnostic
orchestration path.

Usage:
    python tests/test_particle_ordering.py test
    python tests/test_particle_ordering.py DATA_DIR SNAP_NUM [TARGET]
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.dont_write_bytecode = True

cache_root = tempfile.gettempdir()
os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_root, "diagnostic-plots-mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(cache_root, "diagnostic-plots-cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from run_all_diagnostics import read_plot_flags, run_all_diagnostics
from scripts.density_2d import plot_2d_hist
from scripts.helpers import identify_target_halo, load_target_halo_particle_data
from scripts.helpers import plot_dir
from tests.generate_test_galaxy import create_test_galaxy_snapshot


def run_reordered_diagnostics(
    data_dir: str,
    snapnum: int,
    target: int | None = None,
    seed: int = 42,
) -> dict:
    """
    Identify a target halo, shuffle its particle ordering in memory, and run
    the standard diagnostics with reordered output filenames.
    """
    if target is None:
        target, _, _ = identify_target_halo(data_dir, snapnum)

    particle_data = load_target_halo_particle_data(
        data_dir=data_dir,
        snapnum=snapnum,
        target=target,
        shuffle=True,
        seed=seed,
    )

    return run_all_diagnostics(
        data_dir=data_dir,
        snapnum=snapnum,
        particle_data=particle_data,
        output_subdir="ordering_test",
        filename_suffix="_reordered",
    )


def _load_reordered_test_particle_data(snap_path: Path, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    particles = {}

    with h5py.File(snap_path, "r") as f:
        header = dict(f["Header"].attrs.items())

        for parttype in (0, 1, 4):
            group = f.get(f"PartType{parttype}")
            if group is None:
                continue

            count = group["Coordinates"].shape[0]
            order = rng.permutation(count)
            particles[parttype] = {}

            for field, dataset in group.items():
                if dataset.shape and dataset.shape[0] == count:
                    particles[parttype][field] = dataset[:][order]

    return {
        "data_dir": "test_data",
        "snapnum": "test",
        "target": 0,
        "header": header,
        "halo_pos": np.zeros(3),
        "particles": particles,
    }


def run_reordered_test_diagnostics(seed: int = 42) -> dict:
    script_dir = Path(__file__).resolve().parent
    test_data_dir = script_dir / "sim_data"
    snap_path = test_data_dir / "test_galaxy.hdf5"

    if not snap_path.exists():
        print(f"Generating test galaxy snapshot at {snap_path}...")
        create_test_galaxy_snapshot(
            output_dir=str(test_data_dir), filename="test_galaxy.hdf5"
        )
    else:
        print(f"Using existing test galaxy snapshot at {snap_path}")

    particle_data = _load_reordered_test_particle_data(snap_path, seed=seed)
    plot_flags = read_plot_flags(repo_root / "Sim_params.txt")
    output_dir = os.path.join(plot_dir, "test", "ordering_test")
    os.makedirs(output_dir, exist_ok=True)

    results = {"output_dir": output_dir}
    data_dir = "test_data"

    if plot_flags.get("DM_density", 0):
        results["dm_density_map"] = plot_2d_hist(
            data_dir=data_dir,
            snapnum="test",
            parttype=1,
            target=0,
            output_path=os.path.join(output_dir, f"{data_dir}_dm_density_reordered.png"),
            xlabel="x",
            ylabel="y",
            title=f"DM density ({data_dir})",
            particle_data=particle_data,
        )

    if plot_flags.get("gas_density", 0):
        results["gas_density_map"] = plot_2d_hist(
            data_dir=data_dir,
            snapnum="test",
            parttype=0,
            target=0,
            output_path=os.path.join(output_dir, f"{data_dir}_gas_density_reordered.png"),
            xlabel="x",
            ylabel="y",
            title=f"Gas density ({data_dir})",
            particle_data=particle_data,
        )

    if plot_flags.get("star_density", 0):
        results["stellar_density_map"] = plot_2d_hist(
            data_dir=data_dir,
            snapnum="test",
            parttype=4,
            target=0,
            output_path=os.path.join(output_dir, f"{data_dir}_stellar_density_reordered.png"),
            xlabel="x",
            ylabel="y",
            title=f"Stellar density ({data_dir})",
            particle_data=particle_data,
        )

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomize the target halo's particle order in memory and run the "
            "standard diagnostics with _reordered output filenames."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "first_arg",
        help="Either 'test' for the toy galaxy, or DATA_DIR for production data.",
    )
    parser.add_argument(
        "second_arg",
        nargs="?",
        help="SNAP_NUM for production data (omit for test mode).",
    )
    parser.add_argument(
        "target",
        nargs="?",
        type=int,
        default=None,
        help="Optional target halo index. If omitted, identify_target_halo is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the in-memory particle permutation.",
    )
    args = parser.parse_args()

    if args.first_arg == "test":
        args.mode = "test"
        args.data_dir = None
        args.snapnum = None
        if args.second_arg is not None:
            parser.error("Test mode takes no additional arguments.")
    else:
        args.mode = "production"
        args.data_dir = args.first_arg
        if args.second_arg is None:
            parser.error(
                "Production mode requires both DATA_DIR and SNAP_NUM.\n"
                "Usage: python tests/test_particle_ordering.py DATA_DIR SNAP_NUM [TARGET]"
            )
        try:
            args.snapnum = int(args.second_arg)
        except ValueError:
            parser.error("SNAP_NUM must be an integer.")

    return args


def main() -> int:
    args = _parse_args()
    if args.mode == "test":
        results = run_reordered_test_diagnostics(seed=args.seed)
    else:
        results = run_reordered_diagnostics(
            data_dir=args.data_dir,
            snapnum=args.snapnum,
            target=args.target,
            seed=args.seed,
        )

    for key, value in results.items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
