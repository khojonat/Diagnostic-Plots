import argparse
import os
import sys
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.Stellar_Halo_mass import plot_stellar_halo_mass
from scripts.Tully_Fisher import plot_tully_fisher
from scripts.rotation_curve import plot_rotation_curve
from scripts.density_2d import plot_2d_hist
from scripts.SFR_history import plot_sfr_history
from scripts.Kennicutt_Schmidt import plot_kennicutt_schmidt
from scripts.suite_check import plot_suite_check
from scripts.helpers import (
    Hubbleparam,
    _sim_params,
    compute_rotation_curve_and_save,
    latest_snapshot_num,
    plot_dir,
    sim_data_dir,
)


def read_plot_flags(params_path: Path) -> dict:
    flags = {}
    if not params_path.exists():
        raise FileNotFoundError(f"Sim_params.txt not found at {params_path}")
    with params_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                flags[key.strip()] = int(value.strip().split()[0])
            except ValueError:
                continue
    return flags


def _component_density(parttype_str, label, filename, output_dir, run_num, snapnum, target,
                       particle_data=None, ax=None):
    return plot_2d_hist(
        run_num=run_num,
        snapnum=snapnum,
        parttype={"dm": 1, "gas": 0, "stars": 4}[parttype_str],
        target=target,
        output_path=None if ax is not None else os.path.join(output_dir, filename),
        xlabel="x",
        ylabel="y",
        title=label,
        particle_data=particle_data,
        ax=ax,
    )


def _dashboard_figure(panel_count: int, include_sobol: bool):
    """Build the shared dashboard and return axes for normal and Sobol panels."""
    diagnostic_rows = max(1, (panel_count + 2) // 3)
    row_count = diagnostic_rows + (2 if include_sobol else 0)
    fig = plt.figure(figsize=(18, 5.5 * diagnostic_rows + (10 if include_sobol else 0)))
    # Matplotlib's default GridSpec top margin (0.88) leaves a large gap below
    # the suptitle. Reserve only the small amount needed for the title itself.
    grid = fig.add_gridspec(row_count, 3, top=0.965, hspace=0.42, wspace=0.32)
    axes = [fig.add_subplot(grid[index // 3, index % 3]) for index in range(panel_count)]
    sobol_axes = None
    if include_sobol:
        sobol_axes = [
            [fig.add_subplot(grid[diagnostic_rows + row, col]) for col in range(3)]
            for row in range(2)
        ]
    return fig, axes, sobol_axes


def _save_dashboard(fig, output_dir: str, run_num: int | str, filename_suffix: str = "") -> str:
    os.makedirs(output_dir, exist_ok=True)
    run_label = "test" if run_num == "test" else f"run_{run_num}"
    outname = os.path.join(output_dir, f"diagnostics_dashboard_{run_label}{filename_suffix}.png")
    fig.savefig(outname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outname


def run_all_diagnostics(run_num: int, snapnum: int | None = None, particle_data: dict | None = None,
                        filename_suffix: str = ""):
    """Run all enabled diagnostics for ``snapshot_base/run_<run_num>``.

    When ``snapnum`` is omitted, use the highest available snapshot number.
    """
    from scripts.helpers import identify_target_halo

    run_num = int(run_num)
    if snapnum is None:
        snapnum = latest_snapshot_num(run_num)
    plot_flags = read_plot_flags(Path(__file__).resolve().parent / "Sim_params.txt")
    target, min_mass, max_mass = identify_target_halo(run_num, snapnum)
    if particle_data is not None:
        target = particle_data.get("target", target)

    output_dir = plot_dir

    panels = []
    if plot_flags.get("stellar_halo", 0):
        panels.append(partial(plot_stellar_halo_mass, run_num=run_num, snapnum=snapnum,
                              target=target, min_mass=min_mass, max_mass=max_mass,
                              output_dir=output_dir, particle_data=particle_data,
                              filename_suffix=filename_suffix))

    results = {}
    rot_needed = plot_flags.get("rot_curve", 0) or plot_flags.get("Tully_Fisher", 0)
    if rot_needed:
        rot_curve_file = compute_rotation_curve_and_save(
            run_num=run_num, snapnum=snapnum, target=target,
            output_dir=sim_data_dir, particle_data=particle_data,
            filename_suffix=filename_suffix,
        )
        results["rotation_curve_data"] = rot_curve_file
        if plot_flags.get("rot_curve", 0):
            panels.append(partial(plot_rotation_curve, rot_curve_file=rot_curve_file,
                                  run_num=run_num, output_dir=output_dir,
                                  filename_suffix=filename_suffix))
        if plot_flags.get("Tully_Fisher", 0):
            panels.append(partial(plot_tully_fisher, rot_curve_file=rot_curve_file,
                                  run_num=run_num, output_dir=output_dir,
                                  filename_suffix=filename_suffix))

    for parttype, flag, label, filename in (
        ("dm", "DM_density", "DM density", "dm_density"),
        ("gas", "gas_density", "Gas density", "gas_density"),
        ("stars", "star_density", "Stellar density", "stellar_density"),
    ):
        if plot_flags.get(flag, 0):
            panels.append(partial(_component_density, parttype, f"{label} (run_{run_num})",
                                  f"FIRE2_MW_run_{run_num}_{filename}{filename_suffix}.png",
                                  output_dir, run_num, snapnum, target,
                                  particle_data=particle_data))

    if plot_flags.get("SFR_history", 0):
        panels.append(partial(plot_sfr_history, run_num=run_num, max_snapnum=snapnum,
                              target=target, output_dir=output_dir, particle_data=particle_data,
                              filename_suffix=filename_suffix))
    if plot_flags.get("Ken_Schmidt", 0):
        panels.append(partial(plot_kennicutt_schmidt, run_num=run_num, snapnum=snapnum,
                              target=target, output_dir=output_dir, particle_data=particle_data,
                              filename_suffix=filename_suffix))

    sobol_dir = _sim_params.get("sobol_path", "").strip()
    if not sobol_dir or sobol_dir == "0":
        raise ValueError("The combined production dashboard requires Sim_params.txt:sobol_path.")
    fig, axes, sobol_axes = _dashboard_figure(len(panels), include_sobol=True)
    for panel, ax in zip(panels, axes):
        try:
            panel(ax=ax)
        except Exception as exc:
            print(f"Skipping unavailable diagnostic panel: {exc}")
            ax.clear()
            ax.set_axis_off()
    plot_suite_check(sobol_axes, run_num, Path(sobol_dir) / "sobol_params.txt", Hubbleparam)
    fig.suptitle(f"Diagnostic dashboard: run_{run_num} (snapshot {snapnum})", fontsize=20, y=0.99)
    results["dashboard"] = _save_dashboard(fig, output_dir, run_num, filename_suffix)
    return results


def run_test_diagnostics(particle_data: dict | None = None, filename_suffix: str = ""):
    """Run the enabled toy-galaxy diagnostics in one dashboard, without Sobol panels."""
    from tests.generate_test_galaxy import create_test_galaxy_snapshot

    plot_flags = read_plot_flags(Path(__file__).resolve().parent / "Sim_params.txt")
    test_data_path, snapnum, target = "test_data", "test", 0
    snap_path = os.path.join(test_data_path, "test_galaxy.hdf5")
    if particle_data is None and not os.path.exists(snap_path):
        print(f"Generating test galaxy snapshot at {snap_path}...")
        create_test_galaxy_snapshot(output_dir=test_data_path, filename="test_galaxy.hdf5")

    output_dir = plot_dir
    panels = []
    for parttype, flag, label, filename in (
        ("dm", "DM_density", "DM density", "dm_density"),
        ("gas", "gas_density", "Gas density", "gas_density"),
        ("stars", "star_density", "Stellar density", "stellar_density"),
    ):
        if plot_flags.get(flag, 0):
            panels.append(partial(_component_density, parttype, f"{label} (test)",
                                  f"test_{filename}{filename_suffix}.png", output_dir,
                                  test_data_path, snapnum, target, particle_data=particle_data))
    fig, axes, _ = _dashboard_figure(len(panels), include_sobol=False)
    for panel, ax in zip(panels, axes):
        try:
            panel(ax=ax)
        except Exception as exc:
            print(f"Skipping unavailable diagnostic panel: {exc}")
            ax.clear()
            ax.set_axis_off()
    fig.suptitle("Diagnostic dashboard: test", fontsize=20, y=0.99)
    return {"dashboard": _save_dashboard(fig, output_dir, "test", filename_suffix)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a combined diagnostic dashboard for simulation or test data.\n"
        "Usage: python run_all_diagnostics.py test\n"
        "       python run_all_diagnostics.py RUN_NUM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run", help="Either 'test' for the toy galaxy, or a production run number.")
    args = parser.parse_args()
    if args.run == "test":
        args.mode, args.run_num = "test", None
    else:
        try:
            args.run_num = int(args.run)
        except ValueError:
            parser.error("RUN_NUM must be an integer.")
        args.mode = "production"
    return args


def main() -> None:
    args = _parse_args()
    results = run_test_diagnostics() if args.mode == "test" else run_all_diagnostics(args.run_num)
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
