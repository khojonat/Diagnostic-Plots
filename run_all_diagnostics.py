import argparse

from Stellar_Halo_mass import plot_stellar_halo_mass
from Tully_Fisher import plot_tully_fisher
from rotation_curve import plot_rotation_curve
from load_sim_data import compute_rotation_curve_and_save


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

