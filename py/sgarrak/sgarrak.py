import sys
import os
import time

import numpy as np
import tables as tb
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import astropy.cosmology as cosmo
from scipy.interpolate import interp1d

from astropy.table import Table

import copy

# Default config
if not 'SATGEN_PATH' in globals():
    SATGEN_PATH = None

config_dir = os.getenv('SGARRAK_CONFIG_DIR')
if config_dir is None:
    config_dir = '~/.config/sgarrak'
config_path = os.path.expanduser(os.path.join(config_dir,'config.py'))

import importlib.util

# Read a config module if there is one
if os.path.exists(config_path):
    # Create a module spec from the file path
    spec = importlib.util.spec_from_file_location('sgarrak_config', config_path)
    # Create a new module object based on the spec
    sgarrak_config = importlib.util.module_from_spec(spec)
    # Add the module to sys.modules (optional, but good practice for proper module management)
    sys.modules['sgarrak_config'] = sgarrak_config
    # Execute the module's code within the newly created module object
    spec.loader.exec_module(sgarrak_config)

    # print('Module')
    # print(sgarrak_config)

    if hasattr(sgarrak_config, 'SATGEN_PATH'):
        SATGEN_PATH = sgarrak_config.SATGEN_PATH

# Bad practice, disable this warning if it's too noisy
if SATGEN_PATH is None:
    print('SATGEN_PATH is not defined -- see README!')
    if __name__ == '__main__':
        sys.exit(99)
    else:
        raise ImportError

# Put SatGen on the pythonpath
if not SATGEN_PATH in sys.path:
    sys.path.append(SATGEN_PATH)
SATGEN_ETC_PATH = os.path.join(SATGEN_PATH,'etc')
if not SATGEN_ETC_PATH in sys.path:
    sys.path.append(SATGEN_ETC_PATH)

# SatGen Imports
import config as cfg

# Might want to pass these explicitly to evolve
cfg.Mres    = 100.0
cfg.Rres    = 0.001
cfg.psi_res = 1.0e-5

import cosmo as co
import evolve as ev
from   profiles import NFW,Dekel,MN,Einasto,Green
from   orbit import orbit
import galhalo as gh
import aux
import init

############################################################
def read_hdf5(path,datasets,group='/'):
    """
    Simple pytables read; avoids depending on hdf5_tools.
    """
    with tb.open_file(path, 'r') as f:
        if isinstance(datasets,str):
            # Just one dataset, read it as an array
            data = f.get_node(f'{group}/{datasets}').read()
        else:
            # Assume nodes is an iterable of dataset names under group
            data = dict([ (name,f.get_node(f'{group}/{name}').read()) for name in datasets])
    return data

############################################################
def is_iterable(x):
    """
    Returns true if the argument can be iterated over with
    iter(x).
    """
    try:
        iterator = iter(x)
        return True
    except TypeError:
        pass
    return False

############################################################
def avg_smhm(n,return_edge=False):
    """
    Averaged stellar masses given redshifts and halo masses.
    By drawing N number of random samples with different z
    for each halo mass.
    
    The halo mass range was chosen by min-cfg.mres max-max tree mass
    
    Note init.Mstar reads linear halo mass
    
    """
    zrange  = np.arange(0,14,0.2)
    N = n
    Nz = np.random.choice(zrange,N)
    
    hmrange = 10**np.arange(2,13,0.5)

    avgsm = []
    maxsm = []
    minsm = []

    for hm in hmrange:
        smlist = [init.Mstar(hm,z, choice='B13') for z in Nz]
        avgsm.append(np.median(smlist))
        maxsm.append(max(smlist))
        minsm.append(min(smlist))

    f = interp1d(hmrange,avgsm)

    if return_edge:
        return np.array(maxsm),np.array(minsm),hmrange

    return f


############################################################
class Host():
    def __init__(self, mass, zred, cosmology, fd=0.0, flattening=0., disk_method='fd',
                 output_zred=None, walk_tree='backward', static_halo=False):
        """
        
        Parameters: fd: disk mass fraction
                    flattening: disk scale radius/disk scale height
                    disk_method: fd: use a constant disk mass fraction 
                                 interp: use z=0 SMHM relation to get the stellar mass
                                         and rescale to the hm at infall
                                 z: directly return the stellar mass given z and hm at infall
                                 interp_zavg: Average redshift scatter of SMHM relation
                    output_zred: interpolated if passed

        Note: When assigning a disk_method, fd needs to give a number other than 0.
        """
        self.cosmology = cosmology

        self.evolving_mass = is_iterable(mass)

        # The corresponding pairs of mass and zred are usually taken from
        # a merger tree. They should be the same dimension / size.
        if self.evolving_mass:
            assert(is_iterable(zred))
            assert(len(mass) > 1)
            assert(len(zred) == len(mass))
        else:
            mass = np.repeat(mass, len(zred))
            assert(output_zred is None)
            
        self._tree_mass = np.atleast_1d(mass)
        self._tree_zred = np.atleast_1d(zred)

        # We need the ages at each tree redshift to compute the concentration
        # evolution with Zhao's method, among other things.
        self._tree_t_age = self.cosmology.age(self._tree_zred).value
        self._tree_t_lbk = self.cosmology.lookback_time(self._tree_zred).value       

        # If timesteps are given, interpolate the discrete merger tree
        # history at those timesteps
        if output_zred is not None:
            if is_iterable(output_zred):
                self._output_zred = np.atleast_1d(output_zred)
                self.interpolated = True
            else:
                raise Exception
        else:
            self._output_zred = None
            self.interpolated = False
            
        # Set the number of tree levels    
        if self.evolving_mass:
            if self.interpolated:
                self.nlev = len(self._output_zred)
            else:
                self.nlev = len(self._tree_mass)
        else:
            self.nlev = 1
            
        # Set the properties at output levels
        if self.interpolated:
            # These are the actual output t_ages, to which we interpolate
            self.zred  = self._output_zred
            self.t_age = self.cosmology.age(self._output_zred).value
            self.mass  = np.interp(self.t_age[::-1], self._tree_t_age[::-1], self._tree_mass[::-1])[::-1] 
            
            self._tree_level_to_output_level = np.digitize(self._tree_t_age,self.t_age)
            self._output_level_to_tree_level = np.digitize(self.t_age,self._tree_t_age)

        else:
            self.t_age = self._tree_t_age
            self.mass  = self._tree_mass
            self.zred  = self._tree_zred
            self._tree_level_to_output_level = np.arange(0,self.nlev)
            
        # Level 0 <=> Latest time (root node)
        
        # We need the lookback times to compute the timesteps
        self.t_lbk = self.cosmology.lookback_time(self.zred).value       
        
        if self.evolving_mass:
            self.concentration = halo_mah_to_zhao_c_nfw(self.mass, self.t_age)
        else:
            self.concentration = np.atleast_1d(init.concentration(self.mass[0], self.zred[0], choice='DM14'))
            
        # Make a profile for each timestep
        self.dens_profile = list()
        self.halo_dens_profile = list()
        self.has_disk = fd > 0

        if self.has_disk:

            mean_sm_z0   = 10**gh.lgMs_B13(np.log10(self.mass[0]),z=0.)
            mean_sm_nlev = 10**gh.lgMs_B13(np.log10(self.mass[self.nlev-1]),self.zred[self.nlev-1])
 
            if walk_tree == 'backward':
                starting_disk_mass = init.Mstar(self.mass[0], self.zred[0], choice='B13')
            else:
                # Assign starting mass in the iteration
                starting_disk_mass = 0
                # FIXME this breaks the non-step methods
                #starting_disk_mass = init.Mstar(self.mass[self.nlev-1], 
                #                                self.zred[self.nlev-1], choice='B13')

            prev_disk_mass = starting_disk_mass

            self.disk_mass = list()
            self.disk_reff = list()
            self.disk_dens_profile = list()
            
            # Including the disk potential
            # .rh: halo radius within which density is Delta times rhoc [kpc]

            # idx_iter is the order in which stellar masses are computed
            idx_iter = range(self.nlev) if walk_tree == 'backward' else reversed(range(self.nlev))
            
            for i in idx_iter:
                
                mass_i = self.mass[i]
                conc_i = self.concentration[i]
                z_i = self.zred[i]
                
                halo_profile = NFW(mass_i,conc_i,Delta=200.,z=z_i,sf=1.)

                Reff = gh.Reff(halo_profile.rh,conc_i) # Virial radius & concentration
                scale_radius = 0.766421/(1.+1./flattening) * Reff
                scale_height = scale_radius / flattening

                if disk_method=='interp':
                    # Use z=0 SMHM relation to linearly scale down the stellar mass
                    # with the difference of the halo masses.
                    if walk_tree == 'backward':
                        disk_mass = starting_disk_mass * (mass_i/self.mass[0])
                    else:
                        disk_mass = starting_disk_mass * (mass_i/self.mass[self.nlev-1])
                    self.disk_mass.append(disk_mass)
                    self.disk_reff.append(Reff)

                elif disk_method == 'interp_zavg':
                    # Averaging redshift scatter of SMHM.
                    # Ignore starting_disk_mass
                    avg_fit   = avg_smhm()
                    disk_mass = avg_fit(mass_i)
                    self.disk_mass.append(disk_mass)
                    self.disk_reff.append(Reff)

                elif disk_method == 'interp_sm':
                    # Use the mean z=0 SMHM relation to get the stellar mass difference.
                    # And use the difference to scale down the z=0 disk mass.
                    mean_sm_ilev = 10**gh.lgMs_B13(np.log10(mass_i),z_i)
                    if walk_tree == 'backward':
                        disk_mass = starting_disk_mass * (mean_sm_ilev/mean_sm_z0)
                    else:
                        disk_mass = starting_disk_mass * (mean_sm_ilev/mean_sm_nlev)
                    self.disk_mass.append(disk_mass)
                    self.disk_reff.append(Reff)

                #elif disk_method == 'roll_dice':
                #    # Take the stellar mass as long as it goes up in the past
                #    # If init.Mstar finds a higher mass, do it again.
                #    # This might cost many resources if the current level gives the lowest
                #    # stellar mass (The chance of finding a lower stellar mass at eairlier 
                #    # time is lower).
                #    # Note this method makes job never finish if the rolling result is 
                #    # close to the low edge (it is difficult for the next level to find 
                #    # another lower stellar mass.).
                #    disk_mass = init.Mstar(mass_i,z_i, choice='B13')
                #    while later_disk_mass<disk_mass:
                #        disk_mass = init.Mstar(mass_i,z_i, choice='B13')
                #    self.disk_mass.append(disk_mass)
                #    later_disk_mass = disk_mass

                elif disk_method == 'step':
                    # Similar to roll dice, but only roll the dice once.
                    # If the previous disk mass is higher, we propagate that mass forwards.
                    # (i.e. the disk does not grow)

                    # The disk size does not depend on the disk mass
                    self.disk_reff.append(Reff)
                    
                    # For the forward method, we also required that the disk
                    # mass only grows if the halo mass is above the cooling
                    # threshold.
        
                    # HACK: always use the z=0 relation here for consistency
                    # across redshfits
                    #disk_mass = init.Mstar(mass_i, z_i, choice='B13')
                    disk_mass = init.Mstar(mass_i, 0, choice='B13')
                    grow_disk = threshold_check(mass_i,z_i)

                    if walk_tree == 'backward':
                        # Skip the first step
                        if idx_iter == 0:
                            continue

                        # We have drawn a disk mass for an earlier time
                        if disk_mass <= prev_disk_mass:
                            # Disk was less massive in the past;
                            # accept the mass drawn and update prev_disk_mass
                            # for the next step.
                            self.disk_mass.append(disk_mass)
                            prev_disk_mass = disk_mass
                        else:
                            # Disk can't be more massive in the past;
                            # force no change in mass for this step.
                            self.disk_mass.append(prev_disk_mass)
                    else: 
                        # We have drawn a disk mass for a later time
                        if disk_mass >= prev_disk_mass:
                            if grow_disk == 1:
                                # The halo can cool gas, disk grows
                                self.disk_mass.append(disk_mass)
                                prev_disk_mass = disk_mass
                            else:
                                # The halo can't cool gas, no growth
                                self.disk_mass.append(prev_disk_mass)
                        else:
                            # Disk can't be less massive in the future;
                            # force no change in mass for this step.
                            self.disk_mass.append(prev_disk_mass)

                elif disk_method == 'fd':
                    # Use a fixed disk mass fraction
                    disk_mass = fd * mass_i
                    self.disk_mass.append(disk_mass)
                    self.disk_reff.append(Reff)
                else:
                    raise ValueError(f"Disk method {disk_mathod:s} not supported")
                
                disk_profile = MN(disk_mass,scale_radius,scale_height)

                # Pre-generate the interpolator for M(<r) by computing M(<10kpc)
                disk_profile.M(10,0)

                self.dens_profile.append([halo_profile, disk_profile])
                self.halo_dens_profile.append(halo_profile)
                self.disk_dens_profile.append(disk_profile)
        else:
            for i in range(self.nlev):
                halo_profile = NFW(self.mass[i],
                                   self.concentration[i],
                                   Delta=200.,
                                   z=self.zred[i],
                                   sf=1.)
                # The host and halo density profiles are identical in this case
                self.dens_profile.append(halo_profile)
                self.halo_dens_profile.append(halo_profile)

        # Reverse the array for forward method, so no need to change other function.
        if self.has_disk:
            if walk_tree != 'backward':
                self.disk_mass = self.disk_mass[::-1]
                self.disk_reff = self.disk_reff[::-1]
                self.dens_profile = self.dens_profile[::-1]
                self.halo_dens_profile = self.halo_dens_profile[::-1]
                self.disk_dens_profile = self.disk_dens_profile[::-1]
        return

############################################################
def Tvir_threshold_rez10fit():
    """
    Virial temperature threshold for dark halos to form galaxies.
    
    The parameters are the polynomial fit from Galform with reionization z=10

    Return: fitted function for before the reionization and after the redshift part
    """

    fit_hi = lambda hiz: 5.65454557e-04*hiz**2 - 5.53557011e-02*hiz + 8.26126393e+00
    fit_low = lambda lowz: 1.68741576e-04*lowz**4 - 4.80806788e-03*lowz**3 + 5.52611417e-02*lowz**2 - 3.81763325e-01*lowz + 9.96120795e+00

    return fit_hi,fit_low

def threshold_check(hm,z):
    """
    The cooling threshold check is only taken at before mergers.
    
    Parameters: hm: linear progenitor halo mass before the merger
                z: reshift before the merger
    """
    if z<10:
        _,threshold = Tvir_threshold_rez10fit()
    else:
        threshold,_ = Tvir_threshold_rez10fit()
    
    if np.log10(hm)>threshold(z):
        return 1
    
    return 0

############################################################
class Progenitor():
    def __init__(self, mass, host,
                 cosmology=None, zred=None, level=None, mstar=None,
                 orbit_init_method='li2020', xc=None, eps=None):
        """
        orbit_init_method: one of the following:
            - None: use xc,eps method
            - 'li2020': draw from a distribution per Li et al. 2020
        zred_or_level:
        """
        self.mass_init = mass
        self.mass = self.mass_init

        # A progenitor must be associated with a host
        self.host = host

        # By default the cosmology is the same as the host!
        if cosmology is not None:
            self.cosmology = cosmology
        else:
            cosmology = host.cosmology

        if level is not None:
            # The progenitor infall time is specified by a level (an index in the
            # list of host masses/redshifts, with level=0 at the root of the tree)
            assert(host.evolving_mass)

            self._tree_level = level
            self.zred  = host._tree_zred[ self._tree_level]
            self.infall_t_lbk = host._tree_t_lbk[self._tree_level]

            # In an interpolated tree, the progenitor level should be
            # set to the earlier of whichever two interpolated levels it
            # falls between.
            self.level = host._tree_level_to_output_level[self._tree_level]

        elif zred is not None:
            # The progenitor infall time is specified as a redshift.
            self.zred  = zred
            if host.evolving_mass:
                # This should work even if the host is interpolated at
                # the given zred. Note that this still discretizes
                # the progenitor infall time on whatever is the host
                # redshift grid.
                self.level = np.flatnonzero(host.zred >= self.zred)[0]
            else:
                # There is only one level in a non-evolving host
                self.level = 0

            # Starting time for progenitor
            self.infall_t_lbk = cosmology.lookback_time(self.zred).value

        # Sanity check for interpolated host
        assert(self.zred <= self.host.zred.max())
        assert(self.zred >  self.host.zred.min())

        self.init_host_dens_profile = self.host.dens_profile[self.level]
        self.init_host_halo_dens_profile = self.host.halo_dens_profile[self.level]
        
        self.init_host_concentration = self.host.concentration[self.level]

        if host.has_disk:
            self.init_disk_mass = self.host.disk_mass[self.level]

        # Draw progenitor concentration
        self.concentration = init.concentration(self.mass,self.zred,choice='DM14')

        # The halo potential is a "Green" profile; an NFW with additional
        # methods to adjust for the effects of tidal stripping.
        self.dens_profile = Green(self.mass,self.concentration,Delta=200,z=self.zred)

        # Draw stellar mass from mstar-mhalo releation
        if mstar is None:
            self.mstar_init = init.Mstar(self.mass_init, self.zred, choice='B13')
        else:
            self.mstar_init = mstar
        self.mstar = self.mstar_init

        # Check if the progenitor above the cooling threshold at infall.
        # It would be better to check the entire assembly history.
        self.has_galaxy = threshold_check(mass,self.zred)

        # The mass within rmax is used in the stripping calculations
        self.m_max_init = self.dens_profile.M(self.dens_profile.rmax)

        if orbit_init_method is None:
            # xv in cylindrical coordinates: np.array([R,phi,z,VR,Vphi,Vz])
            self.xc  = xc
            self.eps = eps
            self.xv  = init.orbit(self.init_host_halo_dens_profile, xc=self.xc, eps=self.eps)
        elif orbit_init_method == 'li2020':
            # APC note the use of host_halo_dens_profile, rather than host_dens_profile
            self.vel_ratio, self.gamma = init.ZZLi2020(self.init_host_halo_dens_profile,
                                                       self.mass_init,
                                                       self.zred)
            self.xv = init.orbit_from_Li2020(self.init_host_halo_dens_profile,
                                             self.vel_ratio,
                                             self.gamma)
        else:
            raise Exception

        self.r_init = np.sqrt(self.xv[0]**2+self.xv[2]**2)

        # J/J_circ
        # APC: I have checked this gives j/j_circ consistent with the xc,eps input.
        self.j_tot_init = xv_to_j_tot(self.xv,self.mass_init)

        # A.M. of a circular orbit
        self.j_circ_tot_init = profile_j_circ(self.init_host_halo_dens_profile,
                self.init_host_halo_dens_profile.rh,
                mass=self.mass_init)

        self.circularity_init = self.j_tot_init/self.j_circ_tot_init
        return

############################################################
def profile_j_circ(dens_prof, r, mass=1):
    """
    Finds j_circ at a given radius in the SatGen potential
    supplied.
    """
    # Unit of r is kpc
    # Unit of v is kpc/Gyr (satgen internal unit)
    v_circ = dens_prof.Vcirc(r)
    return mass*r*v_circ

 
############################################################
def xv_to_j_tot(xv,mass=1):
    """
    Return the total angular momentum.

    We should really just compute the angular momentum in cyl. coords.
    For now we convert to cartesian coords and take the regular cross product.
    """
    # Unit of r is kpc
    # Unit of v is kpc/Gyr (satgen internal unit)
    # Phi is in radians
    R, phi, z    = xv[0], xv[1], xv[2]
    x, y         = R*np.cos(phi), R*np.sin(phi)
    vR, vphi, vz = xv[3], xv[4], xv[5]
    vx, vy       = vR*np.cos(phi) - vphi*np.sin(phi), vR*np.sin(phi) + vphi*np.cos(phi)

    r_cart = np.array([x,y,z])
    v_cart = np.array([vx,vy,vz])
    j = mass*np.cross(v_cart, r_cart)
    j_tot = np.sqrt(np.sum(j**2, dtype=np.float64))
    return j_tot

############################################################
def halo_mah_to_zhao_c_nfw(mass, t_age_gyr):
    """
    Returns Zhao et al. concentration from NFW halo mass
    and formation time (age of universe at formation, in Gyr)
    """
    h_c_nfw = list()
    nlev = mass.shape[0]
    for i in range(0,nlev):
        h_c_nfw.append(init.c2_fromMAH(mass[i:],t_age_gyr[i:]))
    return np.array(h_c_nfw)

############################################################
def cyl_to_cart_position(xv):
    """
    Convert the 3D space phase xv [R,phi,z,VR,Vphi,Vz] to Cartesian
    coordinates (configuration space only).
    
    The orbit object is used to calculate the orbital evolution,
    so the length of the array is tsteps-1 because there is no
    integration at the initial step?
    
    """
    R,phi,Z = xv[0],xv[1],xv[2]

    X = R*np.cos(phi)
    Y = R*np.sin(phi)
    return np.array([X,Y,Z])

############################################################
def reshape_coors(coorlist,Narray):
    """
    Re-shape the coors list to be the same size to the 
    other parameters when it is below mres/rres(?).
    """
    
    below_res_coor = [-1,-1,-1]
    if len(coorlist)!=Narray:
        below_res_steps = Narray-len(coorlist)
        coorlist.extend([below_res_coor]*below_res_steps)
    return

############################################################
def evolve_orbit(host, prog ,tsteps=None, 
                 evolve_prog_mass=False, 
                 evolve_past_res_limits=False):
    """
    tstep: timesteps measured forwards from the initial conditions at 
        infall. 
           
    evolve_past_res_limits: if True, keep evolving past SatGen resolution
        limits, as set in cfg.phi_res, cfg.Mres and cfg.Rres. If False,
        if any quantity is below the corresponding resolution limit, do
        not compute the orbit, mass loss etc.. Instead, propagate the last
        computed values forward (i.e. repeat them) in the output arrays.
    
    """
    # An enum
    STATUS_PROG_INTACT = 0
    STATUS_PROG_LOST   = 1
    
    # The first entry in the output arrays is the initial conditions
    radii       = [prog.r_init]
    prog_masses = [prog.mass_init]
    prog_mstars = [prog.mstar_init]
    prog_status = [STATUS_PROG_INTACT]
    prog_coors  = [cyl_to_cart_position(prog.xv)]    
    prog_circularity = [prog.circularity_init]

    has_galaxy  = [prog.has_galaxy]

    if host.has_disk:
        host_disk_masses = [prog.init_disk_mass]

    # Working variables
    prog_mass  = prog_masses[0]
    prog_mstar = prog_mstars[0]
    
    prog_mass_init  = prog_mass
    prog_mstar_init = prog_mstar

    # We DO NOT update the host and prog objects in place;
    # Instead make copies.
    host_dp    = copy.deepcopy(prog.init_host_dens_profile)
    prog_dp    = copy.deepcopy(prog.dens_profile)
    
    prog_m_max_init = prog_dp.M(prog_dp.rmax)

    # We don't need this
    # hc = prog.init_host_concentration
 
    # Store the initial concentraiton for use in the mass loss routine
    init_pc = prog.concentration
    
    o = orbit(prog.xv)    
    xv     = o.xv 
    r      = np.sqrt(xv[0]**2+xv[2]**2)    
    r_init = r

    j_tot  = xv_to_j_tot(xv,prog_mass)
    j_circ = profile_j_circ(host_dp, r, mass=prog_mass)
    j_tot_init = j_tot
    j_circ_init = j_circ

    initial_level = prog.level
    
    # istep = 1 corresponds to evolution from the initial conditions up to the end of the first step 
    # (i.e. from t = 0 up to t = 0 + dt)
    
    # Evolution across the step assumes the host properties to be constant at their intial values.
    # Hence evolution for istep = 1 assumes the host properites to be those at istep = 0
    
    # The timesteps need not be the same as the tree levels (substepping)
   
    # This is the time coordinate of each host level after the initial level, measured from the same
    # t=0 as the orbit evolution timesteps and increasing forwards in time towards the root node.
    host_times_starting_from_initial_level = (host.t_age[:initial_level+1] - host.t_age[initial_level])
    
    # We need to reverse the above, so that the first element corresponds to the infall time 
    # rather than the root of the tree.
    host_times_starting_from_initial_level = host_times_starting_from_initial_level[::-1]

    # Find the reference tree level for each timestep.
    if tsteps is None:
        tsteps = host_times_starting_from_initial_level
        # The reversal is because tree level zero is the root of the tree, not the infall
        # time.
        levels_at_tstep = np.linspace(0,initial_level,initial_level+1,dtype=int)[::-1]        
    else:
        # Interpolate timesteps (t=0 at infall, idx=0) onto grid of tree levels
        # The reversal is because tree level zero is the root of the tree, not the infall
        # time.
        levels_at_tstep = prog.level - (np.searchsorted(host_times_starting_from_initial_level,tsteps,side='right')-1)

    if cfg.Mres is None:
        mres_effective = 0
    else:
        mres_effective = cfg.Mres
        
    nsteps = len(tsteps)
    for istep in range(1,nsteps):    
        t  = tsteps[istep]
        dt = t - tsteps[istep-1]

        # Absolute levels in the tree
        start_step_level = levels_at_tstep[istep] - 1
        end_step_level   = start_step_level + 1

        # Threshold values at resolution limit and skip explicit calculation of remaining steps
        # (i.e. propagate values at rehost_dpsolution limit forward.
        if (prog_mass <= mres_effective) or (r <= cfg.Rres) or ((prog_mass/prog_mass_init) <= cfg.phi_res):
            prog_status.append(STATUS_PROG_LOST)
            if not evolve_past_res_limits:
                radii.append(r)
                prog_circularity.append(j_tot/j_circ)
                prog_masses.append(prog_mass)
                prog_mstars.append(prog_mstar)
                if host.has_disk:
                    host_disk_masses.append(host.disk_mass[start_step_level])
                continue 
        else:
            prog_status.append(STATUS_PROG_INTACT)

        # Update the host profile if needed
        host_dp = copy.deepcopy(host.dens_profile[start_step_level])
        host_concentration = host.concentration[start_step_level]
        if host.has_disk:
            host_disk_mass = host.disk_mass[start_step_level]

        # Evolve the progenitor orbit based on the current mass
        # and host halo profile.

        o.integrate(t, host_dp, prog_mass)
        
        # Note that the coordinates are updated 
        # internally in the orbit instance "o" when calling
        # the ".integrate" method, here we assign them to 
        # a new variable "xv" only for bookkeeping
        xv  = o.xv 
        prog_coors.append(cyl_to_cart_position(xv))
        r   = np.sqrt(xv[0]**2+xv[2]**2)
        radii.append(r)

        # FIXME possible consitency issues here
        # - computing before mass update, ok?
        # - dp includes disk or not?
        j_tot  = xv_to_j_tot(xv,mass=prog_mass)
        j_circ = profile_j_circ(host_dp, r, mass=prog_mass)
        prog_circularity.append(j_tot/j_circ)
        
        if evolve_prog_mass:
            # Evolve the progenitor mass for dt in the current potential
            # Following SatGen (SatEvo), msub takes the initial potentials
            # and orbit at the start of the step.
            # dt is the length of the step (right? APC)
            # SatGen requires the *initial* progenitor concentration and the
            # *instantaneous* host concentration
            alpha_strip = ev.alpha_from_c2(host_concentration,init_pc)

            # prog_dp and host_dp are the instantaneous values, updated
            # for each step.
            prog_evolved_mass, prog_tidal_raidus = ev.msub(prog_dp,
                                                           host_dp,
                                                           xv,
                                                           dt,
                                                           choice='King62',
                                                           alpha=alpha_strip)
            
            # Now update the potential of the satellite to the end of the step, after
            # mass loss. This update function claims to handle the resolution limit.
            prog_dp.update_mass(prog_evolved_mass)

            # Evolve baryonic properties
            
            # This is done in terms of the ratio of mass within r_max *now* to the
            # mass within r_max *at infall*.
            prog_m_max = prog_dp.M(prog_dp.rmax)
            
            # Alpha and leff/lmax here are a little subtle...
            g_le, g_ms = ev.g_EPW18(prog_m_max/prog_m_max_init, 
                                    alpha=1., 
                                    lefflmax=0.1) 
            
            # APC: g_le and g_ms are arrays
            g_le = g_le[0][0]
            g_ms = g_ms[0][0]
            
            # Stellar mass after tidal stripping
            # This is calculated from int *initial* stellar mass,
            # not the current stellar mass!
            prog_mstar = float(prog_mstar_init * g_ms) 
            
            # Progenitor mass after mass loss
            prog_mass  = prog_evolved_mass
            
            prog_masses.append(prog_mass)
            prog_mstars.append(prog_mstar)

            if host.has_disk:
                host_disk_masses.append(host_disk_mass)
        else:
            # No mass evolution
            prog_masses.append(prog_mass_init)
            prog_mstars.append(prog_mstar_init)
          
    reshape_coors(prog_coors,len(tsteps))
  
    # Return
    retdict = dict()

    retdict['prog_masses'] = np.array(prog_masses)   
    retdict['prog_mstars'] = np.array(prog_mstars)  
    retdict['status']      = np.array(prog_status) 
    if host.has_disk:
        retdict['host_disk_masses'] = np.array(host_disk_masses)
    retdict['radii']       = np.array(radii)
    retdict['circularity'] = np.array(prog_circularity)
    retdict['tsteps']      = tsteps
    retdict['tage']        = host.t_age[initial_level] + tsteps
    retdict['prog_dp']     = prog_dp
    retdict['levels_at_tsteps'] = levels_at_tstep
    retdict['host_times_starting_from_initial_level'] = host_times_starting_from_initial_level
    retdict['has_galaxy']  = prog.has_galaxy

    # Note that the orbit xvArray property contains the phase space coordinate at each 
    # timestep, but, since this this computed by SatGen internally, it does not include
    # the initial conditions or any steps below the resolution limit. TODO?
    retdict['orbit'] = np.array(prog_coors)
    
    return retdict
