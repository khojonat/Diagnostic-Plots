import os

import h5py
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from load_sim_data import load_particles, identify_target_halo, loadHalos, split_paired_array, code
from SFR_history import _find_snapshot_file

# Literature comparison
Kennicut_Evans_2012 = np.array([0.13207299196279654, -3.7397260273972615, 0.5583784057018653, -3.1506849315068495, 0.9698427943183545, -2.5342465753424657, 1.3668301066572313, -2.013698630136986, 1.822494060557044, -1.3698630136986303, 2.4545518879846338, -0.47945205479452024, 3.0718900065712984, 0.397260273972603, 3.5863519183137043, 1.1232876712328772, 4.144932517818329, 1.9041095890410962, 4.703432239801851, 2.712328767123288])


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
    box_num: int,
    snapnum: int,
    target: int,
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
    box_num : int
        Box identifier for labeling and filenames.
    snapnum : int
        Snapshot number to analyze.
    target : int
        Target halo index.
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
    # Identify target halo
    # target = identify_target_halo(box_num, snapnum)
    
    # Get halo particle indices for gas (PartType0)
    halo_length = loadHalos(box_num, snapnum, 'GroupLenType')
    start_gas = np.sum(halo_length[:target, 0])
    end_gas = start_gas + halo_length[target, 0]
    start_stars = np.sum(halo_length[:target, 4])
    end_stars = start_stars + halo_length[target, 4]

    # Center on the halo center
    halo_pos = loadHalos(box_num, snapnum, 'GroupPos')[target]

    if code == "arepo":
        # Load gas properties: mass, positions, and instantaneous SFR
        data_gas = load_particles(
            box_num,
            "gas",
            fields=["Masses", "Coordinates", "StarFormationRate"],
            snapnum=snapnum,
            redshift=None,
            verbose=False,
        )

        if "StarFormationRate" not in data_gas:
            raise RuntimeError(
                "Gas StarFormationRate field not found; cannot compute Kennicutt–Schmidt law."
            )

        masses_gas = np.asarray(data_gas["Masses"])[start_gas:end_gas]
        coords_gas = np.asarray(data_gas["Coordinates"])[start_gas:end_gas]
        sfr_gas = np.asarray(data_gas["StarFormationRate"])[start_gas:end_gas]

        coords_centered_gas = coords_gas - halo_pos
        R_gas = np.sqrt(coords_centered_gas[:, 0]**2 + coords_centered_gas[:, 1]**2)

        if r_max is None:
            r_max = np.percentile(R_gas, 99.0)

        r_bins = np.linspace(0.0, r_max, n_annuli + 1)
        R_mid, sigma_gas, sigma_sfr = _compute_annular_surface_densities(
            R_gas, masses_gas, sfr_gas, r_bins
        )

        # Get h for unit conversion
        snapfile = _find_snapshot_file(box_num, snapnum)
        with h5py.File(snapfile, "r") as f:
            h = float(f["Header"].attrs.get("HubbleParam", 1.0))
        sigma_gas *= 1e10 / h

    elif code == "gizmo":
        # Load gas for sigma_gas
        data_gas = load_particles(
            box_num,
            "gas",
            fields=["Masses", "Coordinates"],
            snapnum=snapnum,
            redshift=None,
            verbose=False,
        )

        masses_gas = np.asarray(data_gas["Masses"])[start_gas:end_gas]
        coords_gas = np.asarray(data_gas["Coordinates"])[start_gas:end_gas]
        coords_centered_gas = coords_gas - halo_pos
        R_gas = np.sqrt(coords_centered_gas[:, 0]**2 + coords_centered_gas[:, 1]**2)

        # Load stars for sigma_sfr
        data_stars = load_particles(
            box_num,
            "stars",
            fields=["Masses", "Coordinates", "StellarFormationTime"],
            snapnum=snapnum,
            redshift=None,
            verbose=False,
        )

        masses_stars = np.asarray(data_stars["Masses"])[start_stars:end_stars]
        coords_stars = np.asarray(data_stars["Coordinates"])[start_stars:end_stars]
        star_ages = np.asarray(data_stars["StellarFormationTime"])[start_stars:end_stars]

        coords_centered_stars = coords_stars - halo_pos
        R_stars = np.sqrt(coords_centered_stars[:, 0]**2 + coords_centered_stars[:, 1]**2)

        # Compute snapshot time and star formation times
        snapfile = _find_snapshot_file(box_num, snapnum)
        if snapfile is None:
            raise RuntimeError(f"Snapshot file for snapnum {snapnum} not found.")

        with h5py.File(snapfile, "r") as f:
            header = f["Header"]
            if "Time" in header.attrs:
                a_snap = float(header.attrs["Time"])
            else:
                a_snap = 1.0 / (1.0 + float(header.attrs.get("Redshift", 0.0)))

            h = float(header.attrs.get("HubbleParam", 1.0))
            H0 = 100.0 * h * u.km / u.s / u.Mpc
            Om0 = float(header.attrs.get("Omega0", 0.3))
            cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
            z_snap = 1.0 / a_snap - 1.0
            t_snap = cosmo.age(z_snap).to(u.Myr).value

        valid = (star_ages > 0) & (star_ages <= 1.0)
        t_form = np.full_like(star_ages, np.inf, dtype=float)
        if np.any(valid):
            z_form = np.maximum(0.0, 1.0 / star_ages[valid] - 1.0)
            t_form[valid] = cosmo.age(z_form).to(u.Myr).value

        dt = t_snap - t_form
        recent = (dt >= 0.0) & (dt <= 30.0)
        sfr_stars = np.zeros_like(masses_stars)
        sfr_stars[recent] = masses_stars[recent] * 1e10 / h / (30.0e6)  # since masses are in code units, need to convert

        # Wait, in SFR_history.py, sfr = np.sum(star_mass_recent) * 1e10 / h / (30.0e6)
        # Yes, because masses are in 1e10 Msun/h units? No, in FIRE, masses are in Msun / h, I think.
        # In the code, *1e10 / h, but 1e10 / h would be wrong.
        # In SFR_history.py: sfr = np.sum(star_mass_recent) * 1e10 / h / (30.0e6)
        # But star_mass_recent is from f["PartType4"]["Masses"], which in GADGET is in 1e10 Msun/h units.
        # Yes, so *1e10 / h to get Msun, then / 30e6 to get Msun/yr.
        # Yes.

        if r_max is None:
            r_max = np.percentile(R_gas, 99.0)

        r_bins = np.linspace(0.0, r_max, n_annuli + 1)

        # Compute sigma_gas
        R_mid, sigma_gas, _ = _compute_annular_surface_densities(
            R_gas, masses_gas, np.ones(len(masses_gas)), r_bins
        )

        # Compute sigma_sfr
        _, _, sigma_sfr = _compute_annular_surface_densities(
            R_stars, np.ones(len(masses_stars)), sfr_stars, r_bins
        )

        sigma_gas *= 1e10 / h

    else:
        raise ValueError(f"Unsupported code type: {code}")

    # Only keep annuli with non-zero values to avoid log10 issues
    good = (sigma_gas > 0) & (sigma_sfr > 0)
    sigma_gas = sigma_gas[good]
    sigma_sfr = sigma_sfr[good]

    fig, ax = plt.subplots(figsize=(6, 6))

    KE12_x, KE12_y = split_paired_array(Kennicut_Evans_2012, first_is_x=True)
    ax.plot(KE12_x, KE12_y, label=r"$\rm Kennicut~\&~Evans~2012$")
    
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
    tag = f"_run{box_num}" if box_num is not None else ""
    outname = os.path.join(output_dir, f"Kennicutt_Schmidt{tag}.png")
    fig.savefig(outname, bbox_inches="tight")
    plt.close(fig)

    return outname


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_kennicutt_schmidt()."
    )

