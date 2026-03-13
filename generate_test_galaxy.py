import os

import h5py
import numpy as np


HUBBLE_PARAM = 0.6909

# Number of particles per component (order 1e5 as requested)
N_GAS = 100_000
N_DM = 100_000
N_STARS = 100_000

# Rough Milky Way component masses in Msun
M_STAR_MSUN = 5e10      # stellar mass
M_DM_MSUN = 1e12        # dark matter halo mass
M_GAS_MSUN = 1e10       # cold ISM gas mass


def msun_to_code_mass(m_msun: float) -> float:
    """
    Convert physical mass in Msun to AREPO-style code units of 1e10 Msun / h.
    """
    return m_msun * HUBBLE_PARAM / 1.0e10


def generate_disc_positions(n: int,
                            scale_radius_kpc: float = 3.0,
                            scale_height_kpc: float = 0.3) -> np.ndarray:
    """
    Generate an exponential disc distribution centered on (0,0,0).
    Positions are in kpc, but we treat them as generic length units here.
    """
    R = np.random.exponential(scale_radius_kpc, n)
    phi = np.random.uniform(0.0, 2.0 * np.pi, n)
    z = np.random.normal(0.0, scale_height_kpc, n)

    x = R * np.cos(phi)
    y = R * np.sin(phi)

    return np.vstack((x, y, z)).T


def generate_halo_positions(n: int,
                            scale_radius_kpc: float = 20.0,
                            r_max_kpc: float = 200.0) -> np.ndarray:
    """
    Generate a roughly spherical halo distribution centered on (0,0,0).
    Uses a simple gamma-like radial distribution for plausibility.
    """
    # Sample radii with p(r) ~ r^2 * exp(-r / r_s)
    k = 3.0
    theta = scale_radius_kpc
    r = np.random.gamma(shape=k, scale=theta, size=n)
    r = np.clip(r, 0.0, r_max_kpc)

    cos_theta = np.random.uniform(-1.0, 1.0, size=n)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=n)

    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta

    return np.vstack((x, y, z)).T


def circular_velocity(R: np.ndarray,
                      v0: float = 220.0,
                      R0: float = 3.0) -> np.ndarray:
    """
    Simple Milky Way–like rotation curve in km/s:
    rises then flattens at ~v0 beyond a few kpc.
    """
    R_safe = np.maximum(R, 1e-3)
    return v0 * (1.0 - np.exp(-R_safe / R0))


def generate_disc_velocities(positions: np.ndarray,
                              sigma_R: float = 20.0,
                              sigma_z: float = 10.0) -> np.ndarray:
    """
    Generate disc kinematics with roughly circular rotation plus dispersions.
    Velocities are in km/s (AREPO convention).
    """
    x, y, z = positions.T
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    v_circ = circular_velocity(R)

    # Draw velocities in cylindrical coordinates
    vR = np.random.normal(0.0, sigma_R, size=len(R))
    vphi = v_circ + np.random.normal(0.0, 10.0, size=len(R))
    vz = np.random.normal(0.0, sigma_z, size=len(R))

    # Transform to Cartesian
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    vx = vR * cos_phi - vphi * sin_phi
    vy = vR * sin_phi + vphi * cos_phi

    return np.vstack((vx, vy, vz)).T


def generate_halo_velocities(n: int, sigma: float = 150.0) -> np.ndarray:
    """
    Roughly isotropic halo velocity distribution in km/s.
    """
    return np.random.normal(0.0, sigma, size=(n, 3))


def generate_sfr(positions: np.ndarray,
                 total_sfr: float = 1.0) -> np.ndarray:
    """
    Assign a star formation rate to gas particles such that the sum
    is approximately total_sfr (Msun/yr), with a radial exponential profile.
    """
    x, y, z = positions.T
    R = np.sqrt(x**2 + y**2)
    z_abs = np.abs(z)

    # Weight gas closer to the plane and center more heavily
    weights = np.exp(-R / 3.0) * np.exp(-z_abs / 0.3)
    weights_sum = np.sum(weights)

    if weights_sum <= 0.0:
        # Fallback to uniform if numerical issues arise
        sfr = np.full(len(positions), total_sfr / len(positions))
    else:
        sfr = total_sfr * weights / weights_sum

    return sfr


def create_test_galaxy_snapshot(output_dir: str = "sim_data",
                                filename: str = "test_galaxy.hdf5") -> str:
    """
    Create an AREPO-style snapshot for a Milky Way–mass halo with:
      - Header group and attributes (including HubbleParam and MassTable)
      - PartType0 (gas), PartType1 (dm), PartType4 (stars)
      - Coordinates, Masses, Velocities for each component
      - StarFormationRate for gas (summing to ~1 Msun/yr)
    """
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, filename)

    # Seed for reproducibility
    np.random.seed(12345)

    # Total masses in code units of 1e10 Msun / h
    M_star_code = msun_to_code_mass(M_STAR_MSUN)
    M_dm_code = msun_to_code_mass(M_DM_MSUN)
    M_gas_code = msun_to_code_mass(M_GAS_MSUN)

    m_star = M_star_code / N_STARS
    m_dm = M_dm_code / N_DM
    m_gas = M_gas_code / N_GAS

    # Generate positions
    coords_stars = generate_disc_positions(N_STARS)
    coords_gas = generate_disc_positions(N_GAS, scale_radius_kpc=4.0, scale_height_kpc=0.4)
    coords_dm = generate_halo_positions(N_DM)

    # Center on (0,0,0) explicitly (small correction)
    coords_stars -= np.mean(coords_stars, axis=0)
    coords_gas -= np.mean(coords_gas, axis=0)
    coords_dm -= np.mean(coords_dm, axis=0)

    # Generate velocities
    v_stars = generate_disc_velocities(coords_stars, sigma_R=30.0, sigma_z=20.0)
    v_gas = generate_disc_velocities(coords_gas, sigma_R=15.0, sigma_z=8.0)
    v_dm = generate_halo_velocities(N_DM, sigma=150.0)

    # Star formation rates for gas
    sfr_gas = generate_sfr(coords_gas, total_sfr=1.0)

    # Per-particle masses
    masses_stars = np.full(N_STARS, m_star, dtype=np.float32)
    masses_dm = np.full(N_DM, m_dm, dtype=np.float32)
    masses_gas = np.full(N_GAS, m_gas, dtype=np.float32)

    # Header particle counts
    num_part = np.zeros(6, dtype=np.int64)
    num_part[0] = N_GAS
    num_part[1] = N_DM
    num_part[4] = N_STARS

    # MassTable: 0 for all except index 1 (DM), as requested
    mass_table = np.zeros(6, dtype=np.float64)
    mass_table[1] = m_dm

    with h5py.File(outpath, "w") as f:
        # Header group and attributes
        header = f.create_group("Header")
        header.attrs["HubbleParam"] = np.float64(HUBBLE_PARAM)
        header.attrs["MassTable"] = mass_table
        header.attrs["NumPart_ThisFile"] = num_part
        header.attrs["NumPart_Total"] = num_part
        header.attrs["NumFilesPerSnapshot"] = np.int32(1)
        header.attrs["Time"] = np.float64(0.0)
        header.attrs["Redshift"] = np.float64(0.0)
        header.attrs["BoxSize"] = np.float64(500.0)

        # Common AREPO flags (reasonable defaults)
        header.attrs["Flag_Sfr"] = np.int32(1)
        header.attrs["Flag_Cooling"] = np.int32(1)
        header.attrs["Flag_StellarAge"] = np.int32(0)
        header.attrs["Flag_Metals"] = np.int32(0)
        header.attrs["Flag_Feedback"] = np.int32(1)

        # PartType0: gas
        p0 = f.create_group("PartType0")
        p0.create_dataset("Coordinates", data=coords_gas.astype(np.float32))
        p0.create_dataset("Velocities", data=v_gas.astype(np.float32))
        p0.create_dataset("Masses", data=masses_gas)
        p0.create_dataset("StarFormationRate", data=sfr_gas.astype(np.float32))

        # PartType1: dark matter
        p1 = f.create_group("PartType1")
        p1.create_dataset("Coordinates", data=coords_dm.astype(np.float32))
        p1.create_dataset("Velocities", data=v_dm.astype(np.float32))
        p1.create_dataset("Masses", data=masses_dm)

        # PartType4: stars
        p4 = f.create_group("PartType4")
        p4.create_dataset("Coordinates", data=coords_stars.astype(np.float32))
        p4.create_dataset("Velocities", data=v_stars.astype(np.float32))
        p4.create_dataset("Masses", data=masses_stars)

    return outpath


if __name__ == "__main__":
    path = create_test_galaxy_snapshot()
    print(f"Wrote test galaxy snapshot to: {path}")

