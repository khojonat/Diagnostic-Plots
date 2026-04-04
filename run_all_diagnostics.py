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
from load_sim_data import compute_rotation_curve_and_save, load_particles


def run_all_diagnostics(sim_path: str, snapnum: int, box_num: int | None = None):
    """
    Run all diagnostic plots for a given snapshot.
    Currently includes:
      - Stellar–halo mass relation.
      - Tully–Fisher relation (via precomputed rotation-curve data).
    """
    results = {}

    results["stellar_halo_mass_plot"] = plot_stellar_halo_mass(
        sim_path=sim_path,
        snapnum=snapnum,
        box_num=box_num,
    )

    if box_num is None:
        raise ValueError("box_num must be provided for Tully–Fisher diagnostics.")

    rot_curve_file = compute_rotation_curve_and_save(
        sim_path=sim_path,
        snapnum=snapnum,
        box_num=box_num,
    )
    results["rotation_curve_data"] = rot_curve_file

    results["rotation_curve_plot"] = plot_rotation_curve(
        rot_curve_file=rot_curve_file,
        box_num=box_num,
    )

    results["tully_fisher_plot"] = plot_tully_fisher(
        rot_curve_file=rot_curve_file,
        box_num=box_num,
    )

    # 2D density maps for DM, gas, and stars
    os.makedirs("Plots", exist_ok=True)

    def _component_density(parttype_str: str, label: str, filename: str):
        data = load_particles(
            sim_path,
            parttype_str,
            fields=["Coordinates"],
            snapnum=snapnum,
            redshift=None,
            verbose=False,
        )
        coords = data["Coordinates"]
        # Center on the mean to focus on the main structure
        coords = coords - coords.mean(axis=0)
        x = coords[:, 0]
        y = coords[:, 1]

        output_path = os.path.join("Plots", filename)
        return plot_2d_hist(
            x,
            y,
            nbins=512,
            output_path=output_path,
            xlabel="x (code units)",
            ylabel="y (code units)",
            title=label,
        )

    results["dm_density_map"] = _component_density(
        "dm", f"DM density (box {box_num})", f"FIRE2_MW_run{box_num}_dm_density.png"
    )
    results["gas_density_map"] = _component_density(
        "gas", f"Gas density (box {box_num})", f"FIRE2_MW_run{box_num}_gas_density.png"
    )
    results["stellar_density_map"] = _component_density(
        "stars",
        f"Stellar density (box {box_num})",
        f"FIRE2_MW_run{box_num}_stellar_density.png",
    )

    # Star formation history up to the current snapshot
    results["sfr_history_plot"] = plot_sfr_history(
        sim_path=sim_path,
        max_snapnum=snapnum,
        box_num=box_num,
    )

    # Kennicutt–Schmidt relation for the current snapshot
    results["kenn_icutt_schmidt_plot"] = plot_kennicutt_schmidt(
        sim_path=sim_path,
        snapnum=snapnum,
        box_num=box_num,
    )

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all diagnostic plots for a given simulation snapshot."
    )
    parser.add_argument("sim_path", help="Path to the simulation snapshot directory.")
    parser.add_argument("snapnum", type=int, help="Snapshot number to analyze.")
    parser.add_argument(
        "--box-num",
        type=int,
        default=None,
        help="Box identifier used for labeling and filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_all_diagnostics(
        sim_path=args.sim_path,
        snapnum=args.snapnum,
        box_num=args.box_num,
    )

    # Print a short summary of generated outputs
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

