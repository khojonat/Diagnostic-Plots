"""Render the Sobol-parameter context for a diagnostic dashboard."""

from pathlib import Path

import numpy as np

from .helpers import _parse_run_index


def _load_sobol(sobol_path: str | Path) -> np.ndarray:
    """Load the configured Sobol table, accepting its documented header row."""
    path = Path(sobol_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sobol parameter file not found: {path}")
    try:
        sobol = np.loadtxt(path, skiprows=1)
    except ValueError:
        sobol = np.loadtxt(path)
    return np.atleast_2d(sobol)


def plot_suite_check(axs, data_dir: str | int, sobol_path: str | Path, hubble: float) -> None:
    """Plot the suite Sobol distributions and highlight ``data_dir`` in red.

    Parameters
    ----------
    axs
        A 2-by-3 array of Matplotlib axes owned by the caller.
    data_dir
        Diagnosed run identifier, either an integer or a name like ``run_12``.
    sobol_path
        Full path to ``sobol_params.txt``.
    hubble
        Hubble parameter used to convert the DM mass column to solar masses.
    """
    axs = np.asarray(axs)
    if axs.shape != (2, 3):
        raise ValueError("The Sobol dashboard section requires a 2-by-3 grid of axes.")

    sobol = _load_sobol(sobol_path)
    if sobol.shape[1] < 10:
        raise ValueError(f"Sobol table {sobol_path} must contain at least 10 columns.")

    run_index = _parse_run_index(data_dir)
    if not 0 <= run_index < len(sobol):
        raise IndexError(
            f"Run index {run_index} is outside the Sobol table (0-{len(sobol) - 1})."
        )

    selected = sobol[run_index]
    log_halo = sobol[:, 0]
    omega_m = sobol[:, 1]
    sigma8 = sobol[:, 2]
    sne = sobol[:, 3]
    pfb = sobol[:, 4]
    sf_eff = sobol[:, 5]
    sf_alpha = sobol[:, 6]
    uvbz = sobol[:, 7]
    dm_mass = (10 ** sobol[:, 8]) / hubble
    softening = sobol[:, 9]
    selected_dm_mass = (10 ** selected[8]) / hubble

    axs[0, 0].hist(log_halo, edgecolor="white")
    axs[0, 0].axvline(selected[0], color="red", linewidth=2)
    axs[0, 1].scatter(omega_m, sigma8, color="grey", alpha=0.35)
    axs[0, 1].scatter(selected[1], selected[2], color="red", zorder=3)
    axs[0, 2].scatter(sne, pfb, color="grey", alpha=0.35)
    axs[0, 2].scatter(selected[3], selected[4], color="red", zorder=3)
    axs[1, 0].scatter(sf_eff, sf_alpha, color="grey", alpha=0.35)
    axs[1, 0].scatter(selected[5], selected[6], color="red", zorder=3)
    axs[1, 1].scatter(dm_mass, softening, color="grey", alpha=0.35)
    axs[1, 1].scatter(selected_dm_mass, selected[9], color="red", zorder=3)
    axs[1, 2].hist(uvbz, edgecolor="white")
    axs[1, 2].axvline(selected[7], color="red", linewidth=2)

    axs[1, 1].vlines(4.3e4, 0.5, 2, label="FIRE2 MW", color="red")
    axs[1, 1].vlines(3.35e5, 0.5, 2, label="FIREbox")

    axs[0, 0].set_xlabel(r"Log Halo mass $\rm M_\odot$")
    axs[0, 1].set_ylabel(r"$\sigma_8$")
    axs[0, 2].set_ylabel("SNe renorm")
    axs[1, 0].set_ylabel(r"SF $\alpha$")
    axs[1, 1].set_ylabel("Softening")
    axs[0, 1].set_xlabel(r"$\Omega_m$")
    axs[0, 2].set_xlabel("pFB strengths")
    axs[1, 0].set_xlabel(r"SF eff")
    axs[1, 1].set_xlabel(r"DM mass [$\rm M_\odot$]")
    axs[1, 2].set_xlabel("UVB z")
    axs[0, 2].set_xscale("log")
    axs[0, 2].set_yscale("log")
    axs[1, 0].set_xscale("log")
    axs[1, 0].set_yscale("log")
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_yscale("log")
    axs[1, 1].legend(fontsize=10, loc=1)
