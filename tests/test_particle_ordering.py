"""
Run diagnostics after randomizing the target halo's particle ordering.

Snapshot and FOF/Subfind files are only read. The target halo particles are
copied into memory, shuffled there, and passed into the normal diagnostic
orchestration path.

Usage:
    python tests/test_particle_ordering.py test
    python tests/test_particle_ordering.py RUN_NUM [TARGET]
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

from run_all_diagnostics import run_all_diagnostics, run_test_diagnostics
from scripts.helpers import identify_target_halo, load_target_halo_particle_data
from tests.generate_test_galaxy import create_test_galaxy_snapshot


def run_reordered_diagnostics(
    run_num: int,
    snapnum: int | None = None,
    target: int | None = None,
    seed: int = 42,
) -> dict:
    """
    Identify a target halo, shuffle its particle ordering in memory, and run
    the standard diagnostics with reordered output filenames.
    """
    if snapnum is None:
        from scripts.helpers import latest_snapshot_num
        snapnum = latest_snapshot_num(run_num)
    if target is None:
        target, _, _ = identify_target_halo(run_num, snapnum)

    particle_data = load_target_halo_particle_data(
        run_num=run_num,
        snapnum=snapnum,
        target=target,
        shuffle=True,
        seed=seed,
    )

    return run_all_diagnostics(
        run_num=run_num,
        snapnum=snapnum,
        particle_data=particle_data,
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
        "test_data_path": "test_data",
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
    return run_test_diagnostics(
        particle_data=particle_data,
        filename_suffix="_reordered",
    )


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
        help="Either 'test' for the toy galaxy, or a production RUN_NUM.",
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
        args.run_num = None
        if args.target is not None:
            parser.error("Test mode takes no additional arguments.")
    else:
        args.mode = "production"
        try:
            args.run_num = int(args.first_arg)
        except ValueError:
            parser.error("RUN_NUM must be an integer.")

    return args


def main() -> int:
    args = _parse_args()
    if args.mode == "test":
        results = run_reordered_test_diagnostics(seed=args.seed)
    else:
        results = run_reordered_diagnostics(
            run_num=args.run_num,
            target=args.target,
            seed=args.seed,
        )

    for key, value in results.items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
