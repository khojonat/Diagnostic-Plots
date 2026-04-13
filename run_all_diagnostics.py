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


def run_all_diagnostics(box_num: int, snapnum: int):
    """
    Run all diagnostic plots for a given snapshot.
    Currently includes:
      - Stellar–halo mass relation.
      - Tully–Fisher relation (via precomputed rotation-curve data).
    """
    from load_sim_data import identify_target_halo
    
    # Identify target halo once
    target = identify_target_halo(box_num, snapnum)
    
    results = {}

    results["stellar_halo_mass_plot"] = plot_stellar_halo_mass(
        box_num=box_num,
        snapnum=snapnum,
        target=target,
    )

    rot_curve_file = compute_rotation_curve_and_save(
        box_num=box_num,
        snapnum=snapnum,
        target=target,
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

        output_path = os.path.join("Plots", filename)
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

    results["dm_density_map"] = _component_density(
        "dm", f"DM density (run {box_num})", f"FIRE2_MW_run{box_num}_dm_density.png"
    )
    results["gas_density_map"] = _component_density(
        "gas", f"Gas density (run {box_num})", f"FIRE2_MW_run{box_num}_gas_density.png"
    )
    results["stellar_density_map"] = _component_density(
        "stars",
        f"Stellar density (run {box_num})",
        f"FIRE2_MW_run{box_num}_stellar_density.png",
    )

    # Star formation history up to the current snapshot
    results["sfr_history_plot"] = plot_sfr_history(
        box_num=box_num,
        max_snapnum=snapnum,
        target=target,
    )

    # Kennicutt–Schmidt relation for the current snapshot
    results["kennicutt_schmidt_plot"] = plot_kennicutt_schmidt(
        box_num=box_num,
        snapnum=snapnum,
        target=target,
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

