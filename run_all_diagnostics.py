import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Stellar_Halo_mass import plot_stellar_halo_mass
from Tully_Fisher import plot_tully_fisher
from rotation_curve import plot_rotation_curve
from density_2d import plot_2d_hist
from SFR_history import plot_sfr_history
from Kennicutt_Schmidt import plot_kennicutt_schmidt
from load_sim_data import compute_rotation_curve_and_save, plot_dir


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


def run_all_diagnostics(box_num: int, snapnum: int):
    """
    Run all diagnostic plots for a given snapshot.
    Plot production is controlled by flags in Sim_params.txt.
    """
    from load_sim_data import identify_target_halo

    params_path = Path(__file__).resolve().parent / "Sim_params.txt"
    plot_flags = read_plot_flags(params_path)

    # Identify target halo once
    target, min_mass, max_mass = identify_target_halo(box_num, snapnum)

    output_dir = os.path.join(plot_dir, f"run_{box_num}")
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    if plot_flags.get("stellar_halo", 0):
        results["stellar_halo_mass_plot"] = plot_stellar_halo_mass(
            box_num=box_num,
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
            box_num=box_num,
            snapnum=snapnum,
            target=target,
            output_dir=os.path.join(output_dir, "sim_data"),
        )
        results["rotation_curve_data"] = rot_curve_file

        if plot_flags.get("rot_curve", 0):
            results["rotation_curve_plot"] = plot_rotation_curve(
                rot_curve_file=rot_curve_file,
                box_num=box_num,
                output_dir=output_dir,
            )

        if plot_flags.get("Tully_Fisher", 0):
            results["tully_fisher_plot"] = plot_tully_fisher(
                rot_curve_file=rot_curve_file,
                box_num=box_num,
                output_dir=output_dir,
            )

    def _component_density(parttype_str: str, label: str, filename: str):
        output_path = os.path.join(output_dir, filename)
        return plot_2d_hist(
            box_num=box_num,
            snapnum=snapnum,
            parttype={"dm": 1, "gas": 0, "stars": 4}[parttype_str],
            target=target,
            output_path=output_path,
            xlabel="x (code units)",
            ylabel="y (code units)",
            title=label,
        )

    if plot_flags.get("DM_density", 0):
        results["dm_density_map"] = _component_density(
            "dm", f"DM density (run {box_num})", f"FIRE2_MW_run{box_num}_dm_density.png"
        )

    if plot_flags.get("gas_density", 0):
        results["gas_density_map"] = _component_density(
            "gas", f"Gas density (run {box_num})", f"FIRE2_MW_run{box_num}_gas_density.png"
        )

    if plot_flags.get("star_density", 0):
        results["stellar_density_map"] = _component_density(
            "stars",
            f"Stellar density (run {box_num})",
            f"FIRE2_MW_run{box_num}_stellar_density.png",
        )

    if plot_flags.get("SFR_history", 0):
        results["sfr_history_plot"] = plot_sfr_history(
            box_num=box_num,
            max_snapnum=snapnum,
            target=target,
            output_dir=output_dir,
        )

    if plot_flags.get("Ken_Schmidt", 0):
        results["kennicutt_schmidt_plot"] = plot_kennicutt_schmidt(
            box_num=box_num,
            snapnum=snapnum,
            target=target,
            output_dir=output_dir,
        )

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all diagnostic plots for a given simulation snapshot."
    )
    parser.add_argument("box_num", type=int, help="Box number for the simulation.")
    parser.add_argument("snapnum", type=int, help="Snapshot number to analyze.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_all_diagnostics(
        box_num=args.box_num,
        snapnum=args.snapnum,
    )

    # Print a short summary of generated outputs
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

