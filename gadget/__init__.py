from pathlib import Path
import pathlib
import pprint
from typing import Union, Any, Optional, List

import astropy.units as u
import h5py
import numpy as np
from tqdm import tqdm

# import logging
from gadget.units import *


file_type_options = ['snap', 'subh', 'fof', 'particles']

metals = ['Carbon',
          'Helium',
          'Hydrogen',
          'Iron',
          'Magnesium',
          'Neon',
          'Nitrogen',
          'Oxygen',
          'Silicon']

# list dimensionality for all variables to be reshaped
# after reading in
# n3_snap = ['Coordinates',
#            'Velocity']

n3 = ['Coordinates',
      'Velocity',
      'GroupCentreOfPotential',
      'CenterOfMass',
      'CenterOfMassVelocity',
      'CenterOfPotential',
      'GasSpin',
      'NSFSpin',
      'SFSpin',
      'StarSpin',
      'SubSpin',
      'Position']

n6 = ['MassType',
      'HalfMassProjRad',
      'HalfMassRad',
      'SubHalfMassProj',
      'SubHalfMass'
      'Mass_001kpc',
      'Mass_003kpc',
      'Mass_005kpc',
      'Mass_010kpc',
      'Mass_020kpc',
      'Mass_030kpc',
      'Mass_040kpc',
      'Mass_050kpc',
      'Mass_070kpc',
      'Mass_100kpc']

n9 = ['InertiaTensor']


class Gadget(object):
    """
    An object containing all relevant information for a gadget hdf5 file

    Parameters
    ----------
    model_dir : str

        directory of model to read in

    file_type : [default="snap", "fof", "subh", "particles"]

        Kind of file to read:
            "snap" - standard HDF5 Gadget3 snapshot
            "fof" - friends-of-friends group file
            "subh" - subfind group file
            "particles" - particles file

    snapnum : int

        number of snapshot to look at

    verbose : bool (default=False)
        print messages

    units : bool (default=False)
        keeps quantities in Gadget code units

    Examples
    --------
    >>> import gadget as g

    Read in the halo masses from the subhalos file
    >>> model_dir = "/disks/galform11/BAHAMAS/AGN_TUNED_nu0_L100N256_WMAP9/"
    >>> group_info = g.Gadget(model_dir, "subh", 32)
    >>> m200m = group_info.read_var("FOF/Group_M_Mean200")
    
    Read in gas temperatures from snapshot
    >>> part_info = g.Gadget(model_dir, "particles", 32)
    >>> temp = part_info.read_var("PartType0/Temperature")

    List contents of HDF5 file structure
    >>> group_info.list_items("/")
    Items:
    [(u'Constants', <HDF5 group "/Constants" (0 members)>),
     (u'FOF', <HDF5 group "/FOF" (22 members)>),
     (u'Header', <HDF5 group "/Header" (0 members)>),
     (u'IDs', <HDF5 group "/IDs" (2 members)>),
     (u'Parameters', <HDF5 group "/Parameters" (1 members)>),
     (u'Subhalo', <HDF5 group "/Subhalo" (38 members)>),
     (u'Units', <HDF5 group "/Units" (0 members)>)]
    ============================================================
    Attrs:
    []
    ============================================================    
    """
    def __init__(
            self,
            model_dir: Union[str, Path],
            file_type: str,
            snapnum: int,
            smooth: bool=False,
            verbose: bool=False,
            units: bool=True,
            comoving: bool=False) -> None:
        """Initializes some parameters."""
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists:
            raise ValueError(f'{model_dir} does not exist')
        self.file_type = file_type
        self.snapnum = snapnum
        self.filename = self.get_full_dir(
            self.model_dir, self.file_type, self.snapnum
        )
        self.smooth = smooth
        self.verbose = verbose
        self.units = units
        self.comoving = comoving

    def get_full_dir(
            self, model_dir: Path, file_type: str, snapnum: int) -> Path:
        """Get filename, including full path and load extra info about number
        of particles in file.
        """
        ext_options = ['.0.hdf5', '.hdf5']
        # find data directory
        data_dir_options = [
            model_dir / 'data', model_dir / 'Data'
        ]
        data_dir = None
        for dr in data_dir_options:
            if dr.exists():
                data_dir = dr
                break

        if data_dir is None:
            raise ValueError(f'Could not find valid data directory in {model_dir}')

        # find snapshot file
        if file_type == 'snap':
            snap_dir_options = [
                data_dir / f'snapshot_{snapnum:03}',
                data_dir / f'Snapshots/snapshot_{snapnum:03}',
            ]
            snap_dir = None
            for dr in snap_dir_options:
                if dr.exists():
                    snap_dir = dr
                    break

            if snap_dir is None:
                raise ValueError(f'Could not find valid snapshot directory in {data_dir}')

            snap_fname_options = [
                f'snap_{snapnum:03d}',
            ]
            filename = None
            for fn in snap_fname_options:
                for ext in ext_options:
                    if (snap_dir / fn).with_suffix(ext).exists():
                        filename = (snap_dir / fn)
                        break

            if filename is None:
                raise ValueError(f'Could not find valid snapshot file in {snap_dir}')

        # find FOF file
        if file_type == 'fof':
            fof_dir_options = [
                data_dir / f'groups_{snapnum:03}',
                data_dir / f'EagleSubGroups_5r200/groups_{snapnum:03}',
            ]
            fof_dir = None
            for dr in fof_dir_options:
                if dr.exists():
                    fof_dir = dr
                    break

            if fof_dir is None:
                raise ValueError(f'Could not find valid fof directory in {data_dir}')

            fof_fname_options = [
                f'group{snapnum:03d}',
                f'group_tab_{snapnum:03d}',
            ]
            filename = None
            for fn in fof_fname_options:
                for ext in ext_options:
                    if (fof_dir / fn).with_suffix(ext).exists():
                        filename = (fof_dir / fn)
                        break

            if filename is None:
                raise ValueError(f'Could not find valid FOF file in {fof_dir}')

        # find subhalo file
        if file_type == 'subh':
            subh_dir_options = [
                data_dir / f'subhalos_{snapnum:03}',
                data_dir / f'groups_{snapnum:03}',
                data_dir / f'EagleSubGroups_5r200/groups_{snapnum:03}',
            ]
            fname_base = f'subh_{snapnum:03d}.'
            subh_dir = None
            for dr in subh_dir_options:
                if dr.exists():
                    subh_dir = dr
                    break

            if subh_dir is None:
                raise ValueError(f'Could not find valid subhalo directory in {data_dir}')

            subh_fname_options = [
                f'subhalo_{snapnum:03d}',
                f'eagle_subfind_tab_{snapnum:03d}',
            ]
            filename = None
            for fn in subh_fname_options:
                for ext in ext_options:
                    if (subh_dir / fn).with_suffix(ext).exists():
                        filename = (subh_dir / fn)
                        break

            if filename is None:
                raise ValueError(f'Could not find valid subhalo file in {subh_dir}')

        # find particles file
        if file_type == 'particles':
            part_dir_options = [
                data_dir / f'particledata_{snapnum:03}',
                data_dir / f'EagleSubGroups_5r200/particledata_{snapnum:03}',
            ]
            fname_base = f'part_{snapnum:03d}.'
            part_dir = None
            for dr in part_dir_options:
                if dr.exists():
                    part_dir = dr
                    break

            if part_dir is None:
                raise ValueError(f'Could not find valid particledata directory in {data_dir}')

            part_fname_options = [
                f'eagle_subfind_particles_{snapnum:03d}',
            ]
            filename = None
            for fn in part_fname_options:
                for ext in ext_options:
                    if (part_dir / fn).with_suffix(ext).exists():
                        filename = (part_dir / fn)
                        break

            if filename is None:
                raise ValueError(f'Could not find valid particles file in {part_dir}')

        # if sim_type == 'OWLS':
        #     if file_type == 'snap':
        #         dirname = 'snapshot_%.3i/' % snapnum
        #         fname = 'snap_%.3i.' % snapnum

        #     elif file_type == 'fof':
        #         dirname = 'groups_%.3i/' % snapnum
        #         fname = 'group%.3i.' % snapnum

        #     elif file_type == 'subh':
        #         dirname = 'subhalos_%.3i/' % snapnum
        #         fname = 'subhalo_%.3i.' % snapnum

        # elif sim_type == 'BAHAMAS':
        #     dirpath = model_dir.rstrip('/') + '/data/'
        #     if file_type == 'snap':
        #         dirname = 'snapshot_%.3i/' % snapnum
        #         fname = 'snap_%.3i.' % snapnum

        #     elif file_type == 'particles':
        #         dirname = 'particledata_%.3i/' % snapnum
        #         fname = 'eagle_subfind_particles_%.3i.' % snapnum

        #     elif file_type == 'fof':
        #         dirname = 'groups_%.3i/' % snapnum
        #         fname = 'group_tab_%.3i.' % snapnum

        #     elif file_type == 'subh':
        #         dirname = 'groups_%.3i/' % snapnum
        #         fname = 'eagle_subfind_tab_%.3i.' % snapnum

        # elif sim_type == 'BAHAMAS_NEW':
        #     dirpath = model_dir.rstrip('/') + '/Data/'
        #     if file_type == 'snap':
        #         dirname = 'Snapshots/snapshot_%.3i/' % snapnum
        #         fname = 'snap_%.3i.' % snapnum

        #     elif file_type == 'particles':
        #         dirname = 'EagleSubGroups_5r200/particledata_%.3i/' % snapnum
        #         fname = 'eagle_subfind_particles_%.3i.' % snapnum

        #     else:
        #         dirname = 'EagleSubGroups_5r200/groups_%.3i/' % snapnum
        #         if file_type == 'group':
        #             fname = 'group_tab_%.3i.'%snapnum
        #         elif file_type == 'subh':
        #             fname = 'eagle_subfind_tab_%.3i.'%snapnum

        # load actual file
        try:
            f = h5py.File(filename.with_suffix(ext), 'r')
        except:
            breakpoint()
            raise IOError(f'file {filename} does not exist/cannot be opened')

        self.num_files     = f['Header'].attrs['NumFilesPerSnapshot']
        self.num_part_tot  = f['Header'].attrs['NumPart_Total']
        self.num_part_file = f['Header'].attrs['NumPart_ThisFile']

        if file_type == 'fof':
            self.num_files = f['FOF'].attrs['NTask']
            try:
                self.num_groups_tot  = f['FOF'].attrs['Total_Number_of_groups']
                self.num_groups_file = f['FOF'].attrs['Number_of_groups']
            except KeyError:
                self.num_groups_tot  = f['FOF'].attrs['TotNgroups']
                self.num_groups_file = f['FOF'].attrs['Ngroups']

        elif file_type == 'subh':
            self.num_files = f['FOF'].attrs['NTask']
            try:
                self.num_groups_tot  = f['SUBFIND'].attrs['Total_Number_of_groups']
                self.num_groups_file = f['SUBFIND'].attrs['Number_of_groups']
                self.num_sub_groups_tot  = f['SUBFIND'].attrs['Total_Number_of_subgroups']
                self.num_sub_groups_file = f['SUBFIND'].attrs['Number_of_subgroups']
            except KeyError:
                self.num_groups_tot  = f['FOF'].attrs['TotNgroups']
                self.num_groups_file = f['FOF'].attrs['Ngroups']
                self.num_sub_groups_tot  = f['Subhalo'].attrs['TotNgroups']
                self.num_sub_groups_file = f['Subhalo'].attrs['Ngroups']

        self.read_file_attributes(f)

        f.close()

        return filename

    def read_file_attributes(self, f: h5py.File) -> None:
        """Read in different physical parameters from file, should be
        simulation independent.
        """
        # Read info
        z = f['Header'].attrs['Redshift']
        self.z = round(z, 2)
        try:
            self.a = f['Header'].attrs['ExpansionFactor']
        except:
            self.a = f['Header'].attrs['Time']

        self.h = f['Header'].attrs['HubbleParam']
        self.boxsize = f['Header'].attrs['BoxSize'] * R_UNIT # [h^-1 Mpc]

        # the unit mass is given without h=1 units!
        # masses * mass_unit = Omega_m * rho_crit * V / N ~ h^-1
        self.mass_unit = f['Units'].attrs['UnitMass_in_g'] * u.g
        self.masses = f['Header'].attrs['MassTable'] * M_UNIT       # [h^-1]

        # Read conversion units
        self.pi           = f['Constants'].attrs['PI']
        self.gamma        = f['Constants'].attrs['GAMMA']
        self.gravity      = f['Constants'].attrs['GRAVITY']
        self.solar_mass   = f['Constants'].attrs['SOLAR_MASS'] * u.g
        self.solar_lum    = f['Constants'].attrs['SOLAR_LUM'] * u.erg / u.s
        self.rad_const    = f['Constants'].attrs['RAD_CONST'] * u.erg / (u.cm**3 * u.K**4)
        self.avogadro     = f['Constants'].attrs['AVOGADRO']
        self.boltzmann    = f['Constants'].attrs['BOLTZMANN'] * u.erg / u.K
        self.gas_const    = f['Constants'].attrs['GAS_CONST'] * u.erg / (u.mol * u.K)
        self.c            = f['Constants'].attrs['C'] * u.cm / u.s
        self.planck       = f['Constants'].attrs['PLANCK'] * u.erg * u.s
        self.cm_per_mpc   = f['Constants'].attrs['CM_PER_MPC']
        self.protonmass   = f['Constants'].attrs['PROTONMASS'] * u.g
        self.electronmass = f['Constants'].attrs['ELECTRONMASS'] * u.g
        self.hubble       = f['Constants'].attrs['HUBBLE'] * u.s**(-1) * u.littleh
        self.T_cmb        = f['Constants'].attrs['T_CMB0'] * u.K
        self.sec_per_Myr  = f['Constants'].attrs['SEC_PER_MEGAYEAR']
        self.sec_per_year = f['Constants'].attrs['SEC_PER_YEAR']
        self.stefan       = f['Constants'].attrs['STEFAN']  * u.erg / (u.cm**3 * u.K**4)
        self.thompson     = f['Constants'].attrs['THOMPSON'] * u.cm**2
        self.ev_to_erg    = f['Constants'].attrs['EV_TO_ERG']
        self.z_solar      = f['Constants'].attrs['Z_Solar']

        try:
            self.rho_unit_cgs = f['Units'].attrs['UnitDensity_in_cgs'] * u.g / u.cm**3
            self.omega_b = f['Header'].attrs['OmegaBaryon']
            self.omega_m = f['Header'].attrs['Omega0']
        except: # if no baryons in file
            self.rho_unit = 0
            self.omega_b = 0
            self.omega_m = f['Header'].attrs['Omega0']

        try:
            self.bh_seed = f['RuntimePars'].attrs['SeedBlackHoleMass_Msun'] * M_UNIT
            chem = f['Parameters/ChemicalElements'].ref
            self.solarabundance        = f[chem].attrs['SolarAbundance']
            self.solarabundance_oxygen = f[chem].attrs['SolarAbundance_Oxygen']
            self.solarabundance_iron   = f[chem].attrs['SolarAbundance_Iron']
        except:
            pass

    def list_items(self, var: str="/", j: int=0) -> None:
        """List var items and attributes."""
        f = h5py.File(self.filename.with_suffix(f'.{j}.hdf5'), 'r')
        try:
            items = list(f[var].items())
            print('Items:')
            pprint.pprint(items)
            print('============================================================')
        except AttributeError:
            print('%s is not an HDF5 group'%var)

        attrs = list(f[var].attrs.items())
        print('Attrs:')
        pprint.pprint(attrs)
        print('============================================================')


    def get_units(self, var: str, j: int, verbose: bool=True) -> u.Quantity:
        """Return conversion factor for var."""
        f = h5py.File(self.filename.with_suffix(f'.{j}.hdf5'), 'r')
        # read in conversion factors
        string = var.rsplit('/')
        dset = string[-1]
        try:
            if not 'ElementAbundance' in string[-1]:
                self.a_scaling = f[var].attrs['aexp-scale-exponent']
                self.units = UNITS[dset]
            else:
                metal = f[var+'/'+metals[0]].ref
                self.a_scaling = f[metal].attrs['aexp-scale-exponent']
                self.units = UNITS[dset]

        except:
            print('Warning: no conversion factors found in file 0 for %s!'%var)
            self.a_scaling = 0
            self.units = DIMENSIONLESS

        if self.comoving:
            self.a_scaling = 0.

        units = (self.a**self.a_scaling * self.units)
        return units

    def read_attr(
            self,
            path: str,
            ids: Union[List[int], int]=0,
            dtype: Union[str, type]=float,
            verbose: bool=True) -> np.ndarray:
        """Function to readily read out group attributes."""
        if ids is None:
            ids = range(self.num_files)
            shape = (len(ids), -1)
        else:
            ids = np.atleast_1d(ids)
            shape = ids.shape + (-1,)

        string = path.split('/')
        group = '/'.join(string[:-1])
        attr = string[-1]

        attrs = np.empty((0, ), dtype=dtype)
        if verbose:
            iter_ids = tqdm(ids, desc=f'Reading {attr} in files')
        else:
            iter_ids = ids

        for i in iter_ids:
            with h5py.File(self.filename.with_suffix(f'.{i}.hdf5'), 'r') as h5f:
                attrs = np.append(attrs, h5f[group].attrs[attr])

        return attrs.reshape(shape)

    def read_attrs(
            self,
            dset: str,
            ids: Union[List[int], int]=0,
            dtype: Union[str, type]=float,
            verbose: bool=True) -> dict:
        """Function to readily read out all dset attributes."""
        if ids is None:
            ids = range(self.num_files)
            shape = (len(ids), -1)
        else:
            ids = np.atleast_1d(ids)
            shape = (-1,)

        attrs = {}
        if verbose:
            iter_ids = tqdm(ids, desc=f'Reading attrs in files')
        else:
            iter_ids = ids

        for i in iter_ids:
            with h5py.File(self.filename.with_suffix(f'.{i}.hdf5'), 'r') as h5f:
                for attr, val in h5f[dset].attrs.items():
                    if attr not in attrs.keys():
                        attrs[attr] = [val]
                    else:
                        attrs[attr].append(val)

        attrs = {
            k: np.asarray(v).reshape(shape) for k, v in attrs.items()
        }

        return attrs

    def read_var(
            self,
            var: str,
            units: Optional[bool]=None,
            dtype: Union[str, type]=float,
            verbose: Optional[bool]=False,) -> np.ndarray:
        """Read in var for all files."""
        if units is None:
            units = self.units

        data = self.read_all_files(
            var=var, units=units, verbose=verbose)
        if verbose: print('Finished reading snapshot')
        return data.astype(dtype)

    def get_ndata(
            self,
            f: h5py.File,
            var: str) -> int:
        """Get the number of data points of var in f."""
        string = var.rsplit('/')
        if self.file_type == 'snap' or self.file_type == 'particles':
            num_part_file = f['Header'].attrs['NumPart_ThisFile']
            if 'PartType' in string[0]:
                parttype = np.int(string[0][-1])
                Ndata = num_part_file[parttype]
            elif 'ParticleID' in var:
                Ndata = f['Header'].attrs['Nids']

        else:
            if 'FOF' in var:
                if 'PartType' in var:
                    parttype = string[1][-1]
                    Ndata = f['FOF'].attrs['Number_per_Type'][parttype]
                else:
                    try:
                        Ndata = f['FOF'].attrs['Number_of_groups']
                    except KeyError:
                        Ndata = f['FOF'].attrs['Ngroups']

            elif 'SUBFIND' in var:
                if 'PartType' in var:
                    parttype = string[1][-1]
                    Ndata = f['SUBFIND'].attrs['Number_per_Type'][parttype]
                else:
                    Ndata = f['SUBFIND'].attrs['Number_of_subgroups']

            elif 'Subhalo' in var:
                Ndata = f['Subhalo'].attrs['Ngroups']

            elif 'ParticleID' in var:
                Ndata = f['IDs'].attrs['Nids']

        return int(Ndata)

    def read_single_file(
            self,
            i: int,
            var: str,
            units: Optional[bool]=None,
            verbose: Optional[bool]=False,
            reshape: Optional[bool]=True) -> np.ndarray:
        """Read in a single file i"""
        if units is None:
            units = self.units

        if i >= self.num_files:
            raise ValueError(f'{i} should be smaller than {self.num_files}')

        filename = self.filename.with_suffix(f'.{i}.hdf5')
        try:
            f = h5py.File(filename, 'r')
        except OSError:
            raise FileNotFoundError(f'file {filename} does not exist')

        string = var.rsplit('/')

        Ndata = self.get_ndata(f=f, var=var)
        if Ndata == 0:
            if verbose: print(f'Npart in file {i} is {Ndata}...')
            return

        else:
            data = f[var][:].flatten()
            f.close()

            # add units
            if units:
                units = self.get_units(var, i, verbose=verbose)
                data = data * units
                if verbose: print('Returning data with astropy.units.Unit attached')

            if reshape:
                # still need to reshape output
                if string[-1] in n3:
                    return data.reshape(-1,3)
                elif string[-1] in n6:
                    return data.reshape(-1,6)
                elif string[-1] in n9:
                    return data.reshape(-1,9)
                return data

            return data

    def append_result(
            self,
            f: h5py.File,
            var: str,
            j: int,
            verbose: bool=True) -> Union[None, bool]:
        """Append var data from file j to self.data."""
        try:
            self.data = np.append(self.data, f[var][:].flatten(), axis=0)
            return
        except KeyError:
            print(f'KeyError: Variable {var} not found in file {j}')
            print('Returning value of False')
            return False

    def read_all_files(
            self,
            var: str,
            units: Optional[bool]=None,
            verbose: bool=True) -> Union[None, np.ndarray]:
        """Reading routine that does not use hash table."""
        #Set up array depending on what we're reading in
        if units is None:
            units = self.units

        string = var.rsplit('/')

        self.data = np.empty([0])

        #read data from first file
        if verbose: print('Reading variable ', var)

        # loop over files to find the first one with data in it
        # j tracks file number
        # Ndata contains dimension of data
        read = True
        withdata = -1
        Ndata = 0
        filename = self.filename

        if verbose:
            iter_num_files = tqdm(
                range(0, self.num_files),
                desc=f'Reading {var} in files'
            )
        else:
            iter_num_files = range(0, self.num_files)

        for j in iter_num_files:
            f = h5py.File(f'{filename}.{j}.hdf5', 'r')
            Ndata = self.get_ndata(f=f, var=var)
            if Ndata == 0:
                f.close()
            else:
                if withdata < 0: withdata = j
                # read data
                self.append_result(f, var, j, verbose)
                f.close()

        if withdata < 0:
            print('No particles found in any file!')
            return

        # add units
        if units:
            units = self.get_units(var, j-1, verbose=verbose)
            self.data = self.data * units
            if verbose: print('Returning data with astropy.units.Unit attached')

        if verbose: print('Finished reading data')

        # still need to reshape output
        if string[-1] in n3:
            self.data = self.data.reshape(-1,3)
        elif string[-1] in n6:
            self.data = self.data.reshape(-1,6)
        elif string[-1] in n9:
            self.data = self.data.reshape(-1,9)

        return self.data
