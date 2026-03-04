import h5py 
import numpy as np
import sys

torreylabtools_path = '/sfs/gpfs/tardis/home/yja6qa/FIRE_MW_suite/torreylabtools'
illustris_python_path = '~/illustris_python_te'
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
            print(f'Box {box_num}: All valid halos are contaminated')
            target = valid[np.argin(all_contam)]
            break
            
        i += 1

    return target
    
    