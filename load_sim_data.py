import os

import h5py 
import numpy as np
import sys

import astropy.units as u

torreylabtools_path = '/sfs/gpfs/tardis/home/yja6qa/FIRE_MW_suite/torreylabtools'
illustris_python_path = os.path.expanduser('~/')
sys.path.insert(0, str(torreylabtools_path)) # Adding torreylabtools to our path
sys.path.insert(0, str(illustris_python_path)) # Adding illustris_python to our path

import illustris_python_te as il
from torreylabtools import *


def parttype_map(parttype):
    ''' Convert string parttypes to integers '''

    mapping = {
        "gas": 0,
        "dm": 1,
        "stars": 4,
        "bh": 5
    }

    return mapping[parttype.lower()]


def find_snapshot_from_redshift(path, target_z):
    """
    Naive redshift matching.
    Replace with faster table lookup if needed.
    """

    snapnums = []
    redshifts = []

    for fname in sorted(os.listdir(path)):
        if fname.startswith("snapshot_") and fname.endswith(".hdf5"):
            snapnum = int(fname.split("_")[-1].split(".")[0])
            with h5py.File(os.path.join(path, fname), 'r') as f:
                z = f['Header'].attrs['Redshift']

            snapnums.append(snapnum)
            redshifts.append(z)

    snapnums = np.array(snapnums)
    redshifts = np.array(redshifts)

    return snapnums[np.argmin(np.abs(redshifts - target_z))]
    

def load_particles(path, parttype, fields, redshift=None,
                   snapnum=None, verbose=True):
    """
    Load particle fields from AREPO or GIZMO snapshots
    using raw HDF5 access.
    """

    if isinstance(parttype, str):
        try:
            parttype = parttype_map(parttype)
        except Error:
            print("Valid particle types are gas, dm, stars, or bh [0,1,4,5]")
            raise ValueError(f"Error! {parttype} is not a valid particle type.")
            
    # ---------------------------------------
    # Determine snapshot number
    # ---------------------------------------
    if snapnum is None and redshift is None:
        raise ValueError("Must supply either snapnum or redshift")

    if snapnum is None:
        snapnum = find_snapshot_from_redshift(path, redshift)

    if verbose:
        print(f"Loading snapshot {snapnum}")

    # ---------------------------------------
    # Locate snapshot file
    # ---------------------------------------
    snapdir = os.path.join(path, f"snapdir_{snapnum:03d}")

    if os.path.isdir(snapdir):
        snapfile = os.path.join(
            snapdir, f"snap_{snapnum:03d}.hdf5"
        )
    else:
        snapfile = os.path.join(
            path, f"snap_{snapnum:03d}.hdf5"
        )

    # ---------------------------------------
    # Load data
    # ---------------------------------------
    data = {}

    with h5py.File(snapfile, 'r') as f:

        header = f['Header']
        z = header.attrs['Redshift']

        if verbose:
            print("Snapshot redshift:", z)

        pgroup = f[f'PartType{parttype}']

        for field in fields:
            if field in pgroup:
                data[field] = pgroup[field][:]
            else:
                print(f"Warning: {field} not found")

    return data


def identify_target_halo(path,redshift,mass_range = [np.log10(5e11),np.log10(2.5e12)]):
    ''' Identifies a target halo of a given mass '''
    
    min_mass = mass_range[0]
    max_mass = mass_range[1]
    
    snap = find_snapshot_from_redshift(path,redshift)
    
    halo_masses = il.groupcat.loadHalos(path,snap,'GroupMassType') * 1e10 / h
    halo_pos = il.groupcat.loadHalos(path,snap,'GroupPos') / h
    halo_cm = il.groupcat.loadHalos(path,snap,'GroupCM') / h
    halo_rvir = il.groupcat.loadHalos(path,snap,'Group_R_Crit200') / h
    
    with h5py.File(path+f"/snap_{snap:03}.hdf5", "r") as f:

        Header = f['Header']
        h = Header.attrs['HubbleParam']
        UnitLength = Header.attrs['UnitLength_in_cm'] * u.cm
        DMPos = np.array(f['PartType1']['Coordinates'][:] * UnitLength.to(u.kpc)) / h
    
    DM1_masses  = halo_masses[:,1]
    DM2_masses  = halo_masses[:,2]
    
    logM = np.log10(DM1_masses)
    
    # --- Select halos in desired mass range ---
    valid = np.where((logM >= min_mass) & (logM <= max_mass))[0]
    
    # choose the most massive halo *within* the range below 1% contamination
    i = 0 
    contam = 1
    
    all_contam = []
    
    while contam > 0.01:
        target = valid[i]
        contam = DM2_masses[target]/DM1_masses[target]
        all_contam.append(contam)
    
        # If they're all contaminated, just take the least contaminated halo in the mass range
        if i == len(valid) - 1:
            print(f'Run {box_num}: All valid halos are contaminated')
            target = valid[np.argin(all_contam)]
            break
            
        i += 1

    return target


def split_paired_array(arr, first_is_x: bool = True):
    """
    Generic helper to unpack an interleaved 1D array of the form
    [x0, y0, x1, y1, ...] or [y0, x0, y1, x1, ...].
    This is used to unpack data points from other plots.

    Parameters
    ----------
    arr : array_like
        Flat array with an even number of elements, storing paired values.
    first_is_x : bool, optional
        If True, interpret as [x0, y0, x1, y1, ...] and return (x, y).
        If False, interpret as [y0, x0, y1, x1, ...] and return (x, y).

    Returns
    -------
    x, y : np.ndarray
        Arrays of the same length containing the unpacked x and y values.
    """
    arr = np.asarray(arr)
    if arr.size % 2 != 0:
        raise ValueError("split_paired_array expects an array with an even number of elements.")

    first = arr[0::2]
    second = arr[1::2]

    if first_is_x:
        return first, second
    return second, first


def compute_rotation_curve_and_save(
    sim_path: str,
    snapnum: int,
    box_num: int,
    output_dir: str = "sim_data",
) -> str:
    """
    Load gas, dark matter, and stellar particles for a given snapshot,
    compute cumulative mass profiles and rotation curves, and store the
    compiled data in an HDF5 file for later use (e.g. Tully–Fisher plots).
    Code originally from Alex Garcia. Adapted for this project.

    The output file is saved as sim_data/Run_<box_num>_rot.hdf5 by default.
    """
    os.makedirs(output_dir, exist_ok=True)

    snapfile = os.path.join(sim_path, f"snap_{snapnum:03d}.hdf5")
    with h5py.File(snapfile, "r") as f:
        header = f["Header"]
        boxsize = header.attrs["BoxSize"]
        h = header.attrs["HubbleParam"]
        UnitLength = header.attrs["UnitLength_In_CGS"] * u.cm
        UnitMass = header.attrs["UnitMass_In_CGS"] * u.g
        UnitVelocity = header.attrs["UnitVelocity_In_CGS"] * u.cm / u.s

        gas_mass = np.array(
            f["PartType0"]["Masses"][:] * UnitMass.to(u.M_sun).value
        )
        gas_pos = np.array(
            f["PartType0"]["Coordinates"][:] * UnitLength.to(u.kpc).value
        )

        dm_mass = np.array(
            f["PartType1"]["Masses"][:] * UnitMass.to(u.M_sun).value
        )
        dm_pos = np.array(
            f["PartType1"]["Coordinates"][:] * UnitLength.to(u.kpc).value
        )

        dm2_mass = np.array(
            f["PartType2"]["Masses"][:] * UnitMass.to(u.M_sun).value
        )
        dm2_pos = np.array(
            f["PartType2"]["Coordinates"][:] * UnitLength.to(u.kpc).value
        )

        star_mass = np.array(
            f["PartType4"]["Masses"][:] * UnitMass.to(u.M_sun).value
        )
        star_pos = np.array(
            f["PartType4"]["Coordinates"][:] * UnitLength.to(u.kpc).value
        )

    # Default centers as in the original script (only box 3 used here)
    center_box3 = (51532, 56850, 51738)
    center = center_box3

    def center_and_box_wrap(pos, mass, center_vec, boxsize_val):
        pos = np.array(pos, copy=True)
        for ijk in range(3):
            pos[:, ijk] -= center_vec[ijk]
            pos[pos[:, ijk] > 1.0 * boxsize_val / 2.0, ijk] -= boxsize_val
            pos[pos[:, ijk] < -1.0 * boxsize_val / 2.0, ijk] += boxsize_val

        rad = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)
        return rad, mass

    gas_rad, gas_mass = center_and_box_wrap(gas_pos, gas_mass, center, boxsize)
    dm_rad, dm_mass = center_and_box_wrap(dm_pos, dm_mass, center, boxsize)
    dm2_rad, dm2_mass = center_and_box_wrap(dm2_pos, dm2_mass, center, boxsize)
    star_rad, star_mass = center_and_box_wrap(star_pos, star_mass, center, boxsize)

    dr = 0.05
    rmax = 50.0
    rs = np.arange(dr, rmax + dr, dr)
    cum_mass = np.zeros(len(rs))
    cum_mass_dm_only = np.zeros(len(rs))
    cum_mass_gas_only = np.zeros(len(rs))
    cum_mass_stars_only = np.zeros(len(rs))

    for index, r in enumerate(rs):
        gas_within_dr = gas_rad <= r
        dm_within_dr = dm_rad <= r
        dm2_within_dr = dm2_rad <= r
        star_within_dr = star_rad <= r

        cum_mass[index] = np.sum(
            [
                np.sum(gas_mass[gas_within_dr]),
                np.sum(dm_mass[dm_within_dr]),
                np.sum(dm2_mass[dm2_within_dr]),
                np.sum(star_mass[star_within_dr]),
            ]
        )

        cum_mass_gas_only[index] = np.sum(gas_mass[gas_within_dr])
        cum_mass_dm_only[index] = np.sum(
            [np.sum(dm_mass[dm_within_dr]), np.sum(dm2_mass[dm2_within_dr])]
        )
        cum_mass_stars_only[index] = np.sum(star_mass[star_within_dr])

    # Physical constants and unit conversions
    G = 6.67e-11  # m^3 kg / s^2
    rs_m = rs * 3.086e19  # kpc -> m
    cum_mass_kg = cum_mass * 2e30
    cum_mass_dm_only_kg = cum_mass_dm_only * 2e30
    cum_mass_gas_only_kg = cum_mass_gas_only * 2e30
    cum_mass_stars_only_kg = cum_mass_stars_only * 2e30

    vrot = np.sqrt(G * cum_mass_kg / rs_m) / 1000.0
    vrot_dm_only = np.sqrt(G * cum_mass_dm_only_kg / rs_m) / 1000.0
    vrot_gas_only = np.sqrt(G * cum_mass_gas_only_kg / rs_m) / 1000.0
    vrot_stars_only = np.sqrt(G * cum_mass_stars_only_kg / rs_m) / 1000.0

    # Store original rs in kpc for plotting convenience
    outpath = os.path.join(output_dir, f"Run_{box_num}_rot.hdf5")
    with h5py.File(outpath, "w") as f_out:
        f_out.create_dataset("rs", data=rs)
        f_out.create_dataset("cum_mass", data=cum_mass)
        f_out.create_dataset("cum_mass_dm_only", data=cum_mass_dm_only)
        f_out.create_dataset("cum_mass_gas_only", data=cum_mass_gas_only)
        f_out.create_dataset("cum_mass_stars_only", data=cum_mass_stars_only)
        f_out.create_dataset("vrot", data=vrot)
        f_out.create_dataset("vrot_dm_only", data=vrot_dm_only)
        f_out.create_dataset("vrot_gas_only", data=vrot_gas_only)
        f_out.create_dataset("vrot_stars_only", data=vrot_stars_only)

    return outpath

