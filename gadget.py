import pprint

import h5py
import numpy as np
from tqdm import tqdm

# import logging

import pdb

file_type_options = ['snap', 'subh', 'fof', 'particles']

sims = ['OWLS',
        'BAHAMAS',
        'BAHAMAS_NEW']

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

    gadgetunits : bool (default=False)
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
            self, model_dir, file_type, snapnum, sim='BAHAMAS',
            smooth=False, verbose=False, gadgetunits=False, **kwargs):
        """Initializes some parameters."""
        self.model_dir = model_dir
        if (file_type not in file_type_options or
            (sim == "BAHAMAS" and file_type == "fof")):
            raise ValueError('file_type %s not in options %s'%(file_type,
                                                               file_type_options))
        else:
            self.file_type = file_type
        if sim not in sims:
            raise ValueError('sim %s not in options %s'%(sim, sims))
        else:
            self.sim = sim
        self.filename = self.get_full_dir(model_dir, file_type, snapnum, sim)
        self.snapnum = snapnum
        self.smooth = smooth
        self.verbose = verbose
        self.gadgetunits = gadgetunits

    def get_full_dir(self, model_dir, file_type, snapnum, sim):
        """Get filename, including full path and load extra info about number
        of particles in file.

        """
        if sim == 'OWLS':
            dirpath = model_dir.rstrip('/') + '/data/'
            if file_type == 'snap':
                dirname = 'snapshot_%.3i/' % snapnum
                fname = 'snap_%.3i.' % snapnum

            elif file_type == 'fof':
                dirname = 'groups_%.3i/' % snapnum
                fname = 'group%.3i.' % snapnum

            elif file_type == 'subh':
                dirname = 'subhalos_%.3i/' % snapnum
                fname = 'subhalo_%.3i.' % snapnum

        elif sim == 'BAHAMAS':
            dirpath = model_dir.rstrip('/') + '/Data/'
            if file_type == 'snap':
                dirname = 'Snapshots/snapshot_%.3i/' % snapnum
                fname = 'snap_%.3i.' % snapnum
            elif file_type == 'particles':
                dirname = 'EagleSubGroups_5r200/particledata_%.3i/' % snapnum
                fname = 'eagle_subfind_particles_%.3i.' % snapnum
            else:
                dirname = 'EagleSubGroups_5r200/groups_%.3i/' % snapnum
                if file_type == 'group':
                    fname = 'group_tab_%.3i.'%snapnum
                elif file_type == 'subh':
                    fname = 'eagle_subfind_tab_%.3i.'%snapnum

        elif sim == 'BAHAMAS_NEW':
            dirpath = model_dir.rstrip('/') + '/data/'
            if file_type == 'snap':
                dirname = 'snapshot_%.3i/' % snapnum
                fname = 'snap_%.3i.' % snapnum
            elif file_type == 'snip':
                dirname = 'snipshot_%.3i/' % snapnum
                fname = 'snip_%.3i.' % snapnum
            elif file_type == 'particles':
                dirname = 'particledata_%.3i/' % snapnum
                fname = 'eagle_subfind_particles_%.3i.' % snapnum
            else:
                dirname = 'groups_%.3i/' % snapnum
                if file_type == 'group':
                    fname = 'group_tab_%.3i.'%snapnum
                elif file_type == 'subh':
                    fname = 'eagle_subfind_tab_%.3i.'%snapnum

        # load actual file
        filename = dirpath + dirname + fname
        try:
            try:
                # open first file
                f = h5py.File(filename + '0.hdf5', 'r')
            except :
                f = h5py.File(filename + 'hdf5', 'r')
        except:
            pdb.set_trace()
            raise IOError('file %s does not exist/cannot be opened'%filename)

        if sim == 'OWLS':
            # Read in file and particle info
            self.num_files     = f['Header'].attrs['NumFilesPerSnapshot']
            self.num_part_tot  = f['Header'].attrs['NumPart_Total']
            self.num_part_file = f['Header'].attrs['NumPart_ThisFile']
            if file_type == 'fof':
                self.num_files = f['FOF'].attrs['NTask']
                self.num_groups_tot  = f['FOF'].attrs['Total_Number_of_groups']
                self.num_groups_file = f['FOF'].attrs['Number_of_groups']
            elif file_type == 'subh':
                self.num_files = f['FOF'].attrs['NTask']
                self.num_groups_tot  = f['SUBFIND'].attrs['Total_Number_of_groups']
                self.num_groups_file = f['SUBFIND'].attrs['Number_of_groups']
                self.num_sub_groups_tot  = f['SUBFIND'].attrs['Total_Number_of_subgroups']
                self.num_sub_groups_file = f['SUBFIND'].attrs['Number_of_subgroups']

        elif sim == 'BAHAMAS' or sim == 'BAHAMAS_NEW':
            # Read in file and particle info
            self.num_files     = f['Header'].attrs['NumFilesPerSnapshot']
            self.num_part_tot  = f['Header'].attrs['NumPart_Total']
            self.num_part_file = f['Header'].attrs['NumPart_ThisFile']
            if file_type == 'fof':
                self.num_files = f['FOF'].attrs['NTask']
                self.num_groups_tot  = f['FOF'].attrs['TotNgroups']
                self.num_groups_file = f['FOF'].attrs['Ngroups']
            elif file_type == 'subh':
                self.num_files = f['FOF'].attrs['NTask']
                self.num_groups_tot  = f['FOF'].attrs['TotNgroups']
                self.num_groups_file = f['FOF'].attrs['Ngroups']
                self.num_sub_groups_tot  = f['Subhalo'].attrs['TotNgroups']
                self.num_sub_groups_file = f['Subhalo'].attrs['Ngroups']

        self.read_file_attributes(f)

        f.close()

        return filename

    def read_file_attributes(self, f):
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
        self.boxsize = f['Header'].attrs['BoxSize'] # [h^-1 Mpc]

        # the unit mass is given without h=1 units!
        # masses * mass_unit = Omega_m * rho_crit * V / N ~ h^-1
        self.mass_unit = f['Units'].attrs['UnitMass_in_g']  # [g]
        self.masses = f['Header'].attrs['MassTable']        # [h^-1]

        # Read conversion units
        self.pi           = f['Constants'].attrs['PI']
        self.gamma        = f['Constants'].attrs['GAMMA']
        self.gravity      = f['Constants'].attrs['GRAVITY']
        self.solar_mass   = f['Constants'].attrs['SOLAR_MASS']   # [g] -> no 1/h
        self.solar_lum    = f['Constants'].attrs['SOLAR_LUM']    # [erg/s]
        self.rad_const    = f['Constants'].attrs['RAD_CONST']    # [g]
        self.avogadro     = f['Constants'].attrs['AVOGADRO']
        self.boltzmann    = f['Constants'].attrs['BOLTZMANN']    # [erg/K]
        self.gas_const    = f['Constants'].attrs['GAS_CONST']
        self.c            = f['Constants'].attrs['C']
        self.planck       = f['Constants'].attrs['PLANCK']
        self.cm_per_mpc   = f['Constants'].attrs['CM_PER_MPC']
        self.protonmass   = f['Constants'].attrs['PROTONMASS']   # [g]
        self.electronmass = f['Constants'].attrs['ELECTRONMASS'] # [g]
        self.hubble       = f['Constants'].attrs['HUBBLE']       # [h s^-1]
        self.T_cmb        = f['Constants'].attrs['T_CMB0']       # [K]
        self.sec_per_Myr  = f['Constants'].attrs['SEC_PER_MEGAYEAR']
        self.sec_per_year = f['Constants'].attrs['SEC_PER_YEAR']
        self.stefan       = f['Constants'].attrs['STEFAN']
        self.thompson     = f['Constants'].attrs['THOMPSON']
        self.ev_to_erg    = f['Constants'].attrs['EV_TO_ERG']
        self.z_solar      = f['Constants'].attrs['Z_Solar']

        try:
            self.rho_unit = f['Units'].attrs['UnitDensity_in_cgs']
            self.omega_b = f['Header'].attrs['OmegaBaryon']
            self.omega_m = f['Header'].attrs['Omega0']
        except: # if no baryons in file
            self.rho_unit = 0
            self.omega_b = 0
            self.omega_m = f['Header'].attrs['Omega0']

        try:
            self.bh_seed = f['RuntimePars'].attrs['SeedBlackHoleMass_Msun']
            chem = f['Parameters/ChemicalElements'].ref
            self.solarabundance        = f[chem].attrs['SolarAbundance']
            self.solarabundance_oxygen = f[chem].attrs['SolarAbundance_Oxygen']
            self.solarabundance_iron   = f[chem].attrs['SolarAbundance_Iron']
        except:
            pass

    def list_items(self, var="/", j=0):
        """List var items and attributes."""
        f = h5py.File(self.filename + str(j) + '.hdf5', 'r')
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

    def convert_cgs(self, var, j, verbose=True):
        """Return conversion factor for var."""
        f = h5py.File(self.filename + str(j) + '.hdf5', 'r')
        # read in conversion factors
        string = var.rsplit('/')

        try:
            if not 'ElementAbundance' in string[-1]:
                self.a_scaling = f[var].attrs['aexp-scale-exponent']
                self.h_scaling = f[var].attrs['h-scale-exponent']
                self.CGSconversion = f[var].attrs['CGSConversionFactor']
            else:
                metal = f[var+'/'+metals[0]].ref
                self.a_scaling = f[metal].attrs['aexp-scale-exponent']
                self.h_scaling = f[metal].attrs['h-scale-exponent']
                self.CGSconversion = f[metal].attrs['CGSConversionFactor']

        except:
            print('Warning: no conversion factors found in file 0 for %s!'%var)
            self.a_scaling = 0
            self.h_scaling = 0
            self.CGSconversion = 1

        f.close()
        conversion = (self.a**self.a_scaling * self.h**self.h_scaling *
                      self.CGSconversion)

        if verbose:
            print('Converting to physical quantities in CGS units: ')
            print('a-exp-scale-exponent = ', self.a_scaling)
            print('h-scale-exponent     = ', self.h_scaling)
            print('CGSConversionFactor  = ', self.CGSconversion)

        return conversion

    def read_attr(self, path, ids=0, dtype=float):
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
        for idx in tqdm(ids, desc=f'Reading {attr} in files'):
            with h5py.File(f'{self.filename}{idx}.hdf5', 'r') as h5f:
                attrs = np.append(attrs, h5f[group].attrs[attr])

        return attrs.reshape(shape)

    def read_attrs(self, dset, ids=0, dtype=float):
        """Function to readily read out all dset attributes."""
        if ids is None:
            ids = range(self.num_files)
            shape = (len(ids), -1)
        else:
            ids = np.atleast_1d(ids)
            shape = (-1,)

        attrs = {}
        for idx in tqdm(ids, desc=f'Reading attrs in files'):
            with h5py.File(f'{self.filename}{idx}.hdf5', 'r') as h5f:
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
            self, var, gadgetunits=None, verbose=False, dtype=float):
        """Read in var for all files."""
        if gadgetunits is None:
            gadgetunits = self.gadgetunits

        data = self.read_all_files(
            var=var, gadgetunits=gadgetunits, verbose=verbose)
        if verbose: print('Finished reading snapshot')
        return data.astype(dtype)

    def get_ndata(self, f, var):
        """Get the number of data points of var in f."""
        string = var.rsplit('/')
        if self.sim == 'OWLS':
            if self.file_type == 'snap':
                num_part_file = f['Header'].attrs['NumPart_ThisFile']
                # parttype is always first part of var
                if 'PartType' in string[0]:
                    parttype = np.int(string[0][-1])
                    Ndata = num_part_file[parttype]

            else:
                if 'FOF' in var:
                    if 'PartType' in var:
                        parttype = string[1][-1]
                        Ndata = f['FOF'].attrs['Number_per_Type'][parttype]
                    else:
                        Ndata = f['FOF'].attrs['Number_of_groups']
                elif 'SUBFIND' in var:
                    if 'PartType' in var:
                        parttype = string[1][-1]
                        Ndata = f['SUBFIND'].attrs['Number_per_Type'][parttype]
                    else:
                        Ndata = f['SUBFIND'].attrs['Number_of_subgroups']

        elif self.sim == 'BAHAMAS' or self.sim == 'BAHAMAS_NEW':
            if self.file_type == 'snap' or self.file_type == 'particles':
                num_part_file = f['Header'].attrs['NumPart_ThisFile']
                # parttype is always first part of var
                if 'PartType' in string[0]:
                    parttype = np.int(string[0][-1])
                    Ndata = num_part_file[parttype]
                elif 'ParticleID' in var:
                    Ndata = f['Header'].attrs['Nids']

            else:
                if 'FOF' in var:
                    Ndata = f['FOF'].attrs['Ngroups']
                elif 'Subhalo' in var:
                    Ndata = f['Subhalo'].attrs['Ngroups']
                elif 'ParticleID' in var:
                    Ndata = f['IDs'].attrs['Nids']

        return Ndata

    def read_single_file(
            self, i, var, gadgetunits=None, verbose=True, reshape=True):
        """Read in a single file i"""
        if gadgetunits is None:
            gadgetunits = self.gadgetunits

        if i >= self.num_files:
            raise ValueError(f'{i} should be smaller than {self.num_files}')

        try:
            f = h5py.File(f'{self.filename}{i}.hdf5', 'r')
        except OSError:
            raise FileNotFoundError(f'file {self.filename}{i}.hdf5 does not exist')

        string = var.rsplit('/')

        Ndata = self.get_ndata(f=f, var=var)
        if Ndata == 0:
            if verbose: print(f'Npart in file {i} is {Ndata}...')
            return

        else:
            data = f[var][:].flatten()
            f.close()

            # convert to CGS units
            if not gadgetunits:
                conversion = self.convert_cgs(var, i, verbose=verbose)
                data *= conversion
                if verbose: print('Returning data in CGS units')

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


    def append_result(self, f, var, j, verbose):
        """Append var data from file j to self.data."""
        try:
            self.data = np.append(self.data, f[var][:].flatten(), axis=0)
            return
        except KeyError:
            print(f'KeyError: Variable {var} not found in file {j}')
            print('Returning value of False')
            return False

    def read_all_files(self, var, gadgetunits=None, verbose=True):
        """Reading routine that does not use hash table."""
        #Set up array depending on what we're reading in
        if gadgetunits is None:
            gadgetunits = self.gadgetunits

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
        for j in tqdm(
                range(0, self.num_files),
                desc=f'Reading {var} in files'):
            f = h5py.File(f'{self.filename}{j}.hdf5', 'r')
            Ndata = self.get_ndata(f=f, var=var)
            if Ndata == 0:
                f.close()
            else:
                if withdata < 0: withdata = j
                # read data
                self.append_result(f, var, j,  verbose)
                f.close()

        if withdata < 0:
            print('No particles found in any file!')
            return

        # convert to CGS units
        if not gadgetunits:
            conversion = self.convert_cgs(var, j-1, verbose=verbose)
            self.data = self.data * conversion
            if verbose: print('Returning data in CGS units')

        if verbose: print('Finished reading data')

        # still need to reshape output
        if string[-1] in n3:
            self.data = self.data.reshape(-1,3)
        elif string[-1] in n6:
            self.data = self.data.reshape(-1,6)
        elif string[-1] in n9:
            self.data = self.data.reshape(-1,9)

        return self.data
