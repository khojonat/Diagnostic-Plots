from os.path import isfile
import os

import h5py 
import numpy as np
import sys
from read_sobol import read_sobol
import astropy.units as u
import six


def _read_sim_params() -> dict:
    config_file = os.path.join(os.path.dirname(__file__), "Sim_params.txt")
    params = {}
    with open(config_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            params[key.strip()] = value.strip()
    return params


def _normalize_path(path: str) -> str:
    path = path.strip()
    if not path:
        return path
    if path.endswith(os.sep):
        return path
    return path + os.sep


_sim_params = _read_sim_params()
snapshot_base = _normalize_path(_sim_params.get("snapshot_base", ""))
fof_sub_base = _normalize_path(_sim_params.get("fof_sub_base", ""))
plot_dir = _normalize_path(_sim_params.get("plot_dir", "Plots"))
code = _sim_params.get("code", "arepo").strip().lower()

if not snapshot_base or not fof_sub_base:
    raise ValueError("Sim_params.txt must define snapshot_base and fof_sub_base.")


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
    

def load_particles(box_num, parttype, fields, redshift=None,
                   snapnum=None, verbose=True):
    """
    Load particle fields from AREPO or GIZMO snapshots
    using raw HDF5 access.
    """
    path = os.path.join(snapshot_base, f"run_{box_num}")

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


def identify_target_halo(box_num,snapnum):
    ''' Identifies a target halo of a given mass '''
    
    sobol = read_sobol(box_num)
    Mtarget = sobol[0]

    # --- Select halos in desired mass range ---
    min_mass = Mtarget - 0.1 * 2 # Doubling tolerance to try to catch valid halo
    max_mass = Mtarget + 0.1 * 2
    
    path = os.path.join(snapshot_base, f'run_{box_num}')
    with h5py.File(os.path.join(path, f"snap_{snapnum:03d}.hdf5"), "r") as f:

        Header = f['Header']
        h = Header.attrs['HubbleParam']
    
    halo_masses = loadHalos(box_num,snapnum,'GroupMassType') * 1e10 / h

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
            target = valid[np.argmin(all_contam)]
            break
            
        i += 1

    return target, min_mass, max_mass


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
    box_num: int,
    snapnum: int,
    target: int,
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

    # Identify target halo
    # target = identify_target_halo(box_num, snapnum)
    halo_length = loadHalos(box_num, snapnum, 'GroupLenType')
    halo_pos = loadHalos(box_num, snapnum, 'GroupPos')[target]

    snap_path = os.path.join(snapshot_base, f'run_{box_num}')
    snapfile = os.path.join(snap_path, f"snap_{snapnum:03d}.hdf5")
    with h5py.File(snapfile, "r") as f:
        header = f["Header"]
        boxsize = header.attrs["BoxSize"]
        h = header.attrs["HubbleParam"]
        UnitLength = u.kpc # header.attrs["UnitLength_In_CGS"] * u.cm
        UnitMass = 1e10 * u.Msun # header.attrs["UnitMass_In_CGS"] * u.g
        # UnitVelocity = # header.attrs["UnitVelocity_In_CGS"] * u.cm / u.s

        boxsize_kpc = boxsize * UnitLength.to(u.kpc)

        # Load all particles
        gas_mass_all = np.array(
            f["PartType0"]["Masses"][:] * UnitMass.to(u.M_sun)
        )
        gas_pos_all = np.array(
            f["PartType0"]["Coordinates"][:] * UnitLength.to(u.kpc)
        )

        dm_mass_all = np.array(
            f["PartType1"]["Masses"][:] * UnitMass.to(u.M_sun)
        )
        dm_pos_all = np.array(
            f["PartType1"]["Coordinates"][:] * UnitLength.to(u.kpc)
        )

        dm2_mass_all = np.array(
            f["PartType2"]["Masses"][:] * UnitMass.to(u.M_sun)
        )
        dm2_pos_all = np.array(
            f["PartType2"]["Coordinates"][:] * UnitLength.to(u.kpc)
        )

        star_mass_all = np.array(
            f["PartType4"]["Masses"][:] * UnitMass.to(u.M_sun)
        )
        star_pos_all = np.array(
            f["PartType4"]["Coordinates"][:] * UnitLength.to(u.kpc)
        )

    # Slice to halo particles
    start_gas = np.sum(halo_length[:target, 0])
    end_gas = start_gas + halo_length[target, 0]
    gas_mass = gas_mass_all[start_gas:end_gas]
    gas_pos = gas_pos_all[start_gas:end_gas]

    start_dm = np.sum(halo_length[:target, 1])
    end_dm = start_dm + halo_length[target, 1]
    dm_mass = dm_mass_all[start_dm:end_dm]
    dm_pos = dm_pos_all[start_dm:end_dm]

    start_dm2 = np.sum(halo_length[:target, 2])
    end_dm2 = start_dm2 + halo_length[target, 2]
    dm2_mass = dm2_mass_all[start_dm2:end_dm2]
    dm2_pos = dm2_pos_all[start_dm2:end_dm2]

    start_star = np.sum(halo_length[:target, 4])
    end_star = start_star + halo_length[target, 4]
    star_mass = star_mass_all[start_star:end_star]
    star_pos = star_pos_all[start_star:end_star]

    center = halo_pos * UnitLength.to(u.kpc)  # assuming halo_pos is in code units

    def center_and_box_wrap(pos, mass, center_vec, boxsize_val):
        pos = np.array(pos, copy=True)
        for ijk in range(3):
            pos[:, ijk] -= center_vec[ijk]
            pos[pos[:, ijk] > 1.0 * boxsize_val / 2.0, ijk] -= boxsize_val
            pos[pos[:, ijk] < -1.0 * boxsize_val / 2.0, ijk] += boxsize_val

        rad = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)
        return rad, mass

    gas_rad, gas_mass = center_and_box_wrap(gas_pos, gas_mass, center, boxsize_kpc)
    dm_rad, dm_mass = center_and_box_wrap(dm_pos, dm_mass, center, boxsize_kpc)
    dm2_rad, dm2_mass = center_and_box_wrap(dm2_pos, dm2_mass, center, boxsize_kpc)
    star_rad, star_mass = center_and_box_wrap(star_pos, star_mass, center, boxsize_kpc)

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


# ---- Code below adapted from illustris python (https://github.com/illustristng/illustris_python): ---- # 
# Copyright (c) 2017, illustris & illustris_python developers All rights reserved.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The views and conclusions contained in the software and documentation are those of the 
# authors and should not be interpreted as representing official policies, 
# either expressed or implied, of the FreeBSD Project.

def gcPath(basePath, snapNum):
    """ Return absolute path to a group catalog HDF5 file (modify as needed). """

    filePath1 = basePath + 'groups_%03d.hdf5' % (snapNum)
    filePath2 = basePath + 'fof_subhalo_tab_%03d.hdf5' % (snapNum)

    if isfile(filePath1):
        return filePath1
    return filePath2

def loadHalos(box_num, snapNum, fields=None):
    """ Load all halo information from the entire group catalog for one snapshot
       (optionally restrict to a subset given by fields). """
    basePath = os.path.join(fof_sub_base, f'run_{box_num}') + os.sep
    return loadObjects(basePath, snapNum, "Group", "groups", fields)

def loadObjects(basePath, snapNum, gName, nName, fields):
    """ Load either halo or subhalo information from the group catalog. """
    result = {}

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(gcPath(basePath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())
        result['count'] = f['Header'].attrs['N' + nName + '_Total']

        if not result['count']:
            print('warning: zero groups, empty return (snap=' + str(snapNum) + ').')
            return result

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for field in fields:
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Group catalog does not have requested field [" + field + "]!")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = result['count']

            # allocate within return dict
            result[field] = np.zeros(shape, dtype=f[gName][field].dtype)

    # loop over chunks
    wOffset = 0

    for i in range(header['NumFiles']):
        f = h5py.File(gcPath(basePath, snapNum), 'r')

        if not f['Header'].attrs['N'+nName+'_ThisFile']:
            continue  # empty file chunk

        # loop over each requested field
        for field in fields:
            if field not in f[gName].keys():
                raise Exception("Group catalog does not have requested field [" + field + "]!")

            # shape and type
            shape = f[gName][field].shape

            # read data local to the current file
            if len(shape) == 1:
                result[field][wOffset:wOffset+shape[0]] = f[gName][field][0:shape[0]]
            else:
                result[field][wOffset:wOffset+shape[0], :] = f[gName][field][0:shape[0], :]

        wOffset += shape[0]
        f.close()

    # only a single field? then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]

    return result

