import os

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from load_sim_data import load_particles


def _compute_annular_surface_densities(
    R: np.ndarray,
    mass: np.ndarray,
    sfr: np.ndarray,
    r_bins: np.ndarray,
):
    """
    Compute gas and SFR surface densities in cylindrical annuli.

    This mimics observational practice where gas and SFR are averaged
    in radial rings and normalized by ring area.

    Parameters
    ----------
    R : array_like
        Cylindrical radii (kpc).
    mass : array_like
        Gas mass per element (Msun).
    sfr : array_like
        Star formation rate per element (Msun/yr).
    r_bins : array_like
        Radial bin edges (kpc).

    Returns
    -------
    R_mid : np.ndarray
        Midpoint radius of each annulus (kpc).
    sigma_gas : np.ndarray
        Gas surface density in each annulus (Msun/kpc^2).
    sigma_sfr : np.ndarray
        SFR surface density in each annulus (Msun/yr/kpc^2).
    """
    R = np.asarray(R)
    mass = np.asarray(mass)
    sfr = np.asarray(sfr)
    r_bins = np.asarray(r_bins)

    n_bins = len(r_bins) - 1
    sigma_gas = np.zeros(n_bins)
    sigma_sfr = np.zeros(n_bins)
    R_mid = 0.5 * (r_bins[:-1] + r_bins[1:])

    area = np.pi * (r_bins[1:] ** 2 - r_bins[:-1] ** 2)

    for i in range(n_bins):
        in_bin = (R >= r_bins[i]) & (R < r_bins[i + 1])
        if not np.any(in_bin):
            sigma_gas[i] = 0.0
            sigma_sfr[i] = 0.0
            continue
        M_bin = np.sum(mass[in_bin])
        SFR_bin = np.sum(sfr[in_bin])
        sigma_gas[i] = M_bin / area[i]
        sigma_sfr[i] = SFR_bin / area[i]

    return R_mid, sigma_gas, sigma_sfr


def plot_kennicutt_schmidt(
    sim_path: str,
    snapnum: int,
    box_num: int | None = None,
    r_max: float | None = None,
    n_annuli: int = 20,
    output_dir: str = "Plots",
) -> str:
    """
    Compute and plot the Kennicutt–Schmidt relation for the target halo.

    The gas is projected into cylindrical coordinates (assuming the disc
    lies in the x–y plane), and gas/SFR surface densities are computed
    in radial annuli, similar to observational analyses of disc galaxies.

    Parameters
    ----------
    sim_path : str
        Path to the simulation snapshot directory.
    snapnum : int
        Snapshot number to analyze.
    box_num : int, optional
        Box identifier for labeling and filenames.
    r_max : float, optional
        Maximum radius in kpc to consider. If None, set from the data.
    n_annuli : int, optional
        Number of radial annuli (default: 20).
    output_dir : str, optional
        Directory where the plot PNG will be saved.

    Returns
    -------
    str
        Path to the saved Kennicutt–Schmidt plot.
    """
    # Load gas properties: mass, positions, and instantaneous SFR
    data = load_particles(
        sim_path,
        "gas",
        fields=["Masses", "Coordinates", "StarFormationRate"],
        snapnum=snapnum,
        redshift=None,
        verbose=False,
    )

    if "StarFormationRate" not in data:
        raise RuntimeError(
            "Gas StarFormationRate field not found; cannot compute Kennicutt–Schmidt law."
        )

    masses = np.asarray(data["Masses"])
    coords = np.asarray(data["Coordinates"])
    sfr = np.asarray(data["StarFormationRate"])

    # Center on the gas center of mass to approximate the galaxy center
    coords_centered = coords - coords.mean(axis=0)
    x = coords_centered[:, 0]
    y = coords_centered[:, 1]

    # Cylindrical radius in the (x, y) plane; assumes disc lies in this plane
    R = np.sqrt(x**2 + y**2)

    if r_max is None:
        r_max = np.percentile(R, 99.0)

    r_bins = np.linspace(0.0, r_max, n_annuli + 1)
    R_mid, sigma_gas, sigma_sfr = _compute_annular_surface_densities(
        R, masses, sfr, r_bins
    )

    # Only keep annuli with non-zero values to avoid log10 issues
    good = (sigma_gas > 0) & (sigma_sfr > 0)
    sigma_gas = sigma_gas[good]
    sigma_sfr = sigma_sfr[good]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        np.log10(sigma_gas),
        np.log10(sigma_sfr),
        c=R_mid[good],
        cmap="viridis",
        s=40,
        edgecolor="none",
    )

    # Simple log–log linear fit (optional, illustrative)
    if sigma_gas.size >= 2:
        coeffs = np.polyfit(np.log10(sigma_gas), np.log10(sigma_sfr), 1)
        xfit = np.linspace(np.min(np.log10(sigma_gas)), np.max(np.log10(sigma_gas)), 50)
        yfit = np.polyval(coeffs, xfit)
        ax.plot(xfit, yfit, color="k", ls="--", label=f"Fit: n = {coeffs[0]:.2f}")
        ax.legend()

    ax.set_xlabel(r"$\log \Sigma_{\rm gas}\ [{\rm M_\odot\,kpc^{-2}}]$")
    ax.set_ylabel(r"$\log \Sigma_{\rm SFR}\ [{\rm M_\odot\,yr^{-1}\,kpc^{-2}}]$")
    title = "Kennicutt–Schmidt relation"
    if box_num is not None:
        title += f" (box {box_num})"
    ax.set_title(title)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    tag = f"_box{box_num}" if box_num is not None else ""
    outname = os.path.join(output_dir, f"Kennicutt_Schmidt{tag}.png")
    fig.savefig(outname, bbox_inches="tight")
    plt.close(fig)

    return outname


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_kennicutt_schmidt()."
    )

