"""Run dashboards only for completed simulation runs.

Example
-------
python run_finished_diagnostics.py --snapshot 90 --stop 1024
"""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.helpers import fof_sub_base, plot_dir, snapshot_base


def _snapshot_exists(run_num: int, snapnum: int) -> bool:
    """Support both direct and snapdir snapshot layouts."""
    run_path = Path(snapshot_base) / f"run_{run_num}"
    filename = f"snap_{snapnum:03d}.hdf5"
    return (run_path / filename).is_file() or (run_path / f"snapdir_{snapnum:03d}" / filename).is_file()


def _group_catalog_exists(run_num: int, snapnum: int) -> bool:
    run_path = Path(fof_sub_base) / f"run_{run_num}"
    return any(
        (run_path / f"{prefix}_{snapnum:03d}.hdf5").is_file()
        for prefix in ("groups", "fof_subhalo_tab")
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dashboards for completed simulation runs.")
    parser.add_argument("--start", type=int, default=0, help="First run number to inspect (default: 0).")
    parser.add_argument("--stop", type=int, default=1024, help="Exclusive final run number (default: 1024).")
    parser.add_argument("--snapshot", type=int, default=90, help="Required completed snapshot number (default: 90).")
    parser.add_argument("--force", action="store_true", help="Regenerate dashboards that already exist.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    diagnostics_script = repo_root / "run_all_diagnostics.py"

    finished_runs = [
        run_num
        for run_num in range(args.start, args.stop)
        if _snapshot_exists(run_num, args.snapshot) and _group_catalog_exists(run_num, args.snapshot)
    ]
    print(f"Found {len(finished_runs)} completed runs at snapshot {args.snapshot}.")

    for run_num in finished_runs:
        dashboard = Path(plot_dir) / f"diagnostics_dashboard_run_{run_num}.png"
        if dashboard.is_file() and not args.force:
            print(f"Skipping run_{run_num}: {dashboard} already exists.")
            continue

        print(f"Diagnosing run_{run_num}...")
        subprocess.run(
            [sys.executable, str(diagnostics_script), str(run_num)],
            check=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
