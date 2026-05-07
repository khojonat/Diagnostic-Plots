"""
Run diagnostics after randomizing the target halo's particle ordering.

Snapshot and FOF/Subfind files are only read. The target halo particles are
copied into memory, shuffled there, and passed into the normal diagnostic
orchestration path.

Usage:
    python tests/test_particle_ordering.py DATA_DIR SNAP_NUM [TARGET]
"""

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/diagnostic-plots-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/diagnostic-plots-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from run_all_diagnostics import run_all_diagnostics
from scripts.helpers import identify_target_halo, load_target_halo_particle_data


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomize the target halo's particle order in memory and run the "
            "standard diagnostics with _reordered output filenames."
        )
    )
    parser.add_argument("data_dir", help="Simulation data directory.")
    parser.add_argument("snapnum", type=int, help="Snapshot number.")
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
