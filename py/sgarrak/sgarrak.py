import sys
import os
import time

import numpy as np
import tables as tb
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import astropy.cosmology as cosmo

from astropy.table import Table

import copy

SATGEN_PATH = '/data/apcooper/sfw/SatGen'
if not SATGEN_PATH in sys.path:
    sys.path.append(SATGEN_PATH)

SATGEN_ETC_PATH = '/data/apcooper/sfw/SatGen/etc'
if not SATGEN_ETC_PATH in sys.path:
    sys.path.append(SATGEN_ETC_PATH)

# SatGen Imports
import config as cfg
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
class Host():
    def __init__(self, mass, zred, cosmology, fd=0.1, flattening=25., output_zred=None):
        """
        
        Parameters: fd: disk mass fraction
                    flattening: disk scale radius/disk scale height
                    output_zred: interpolated if passed
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
            assert(not is_iterable(zred))
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
            self.disk_dens_profile = list()
            # Including the disk potential
            # .rh: halo radius within which density is Delta times rhoc [kpc]
            for i in range(self.nlev):
                
                mass_i = self.mass[i]
                conc_i = self.concentration[i]
                z_i = self.zred[i]
                
                halo_profile = NFW(mass_i,conc_i,Delta=200.,z=z_i,sf=1.)

                Reff = gh.Reff(halo_profile.rh,conc_i) # Virial radius & concentration
                scale_radius = 0.766421/(1.+1./flattening) * Reff
                scale_height = scale_radius / flattening
                disk_mass = fd * mass_i
                
                disk_profile = MN(disk_mass,scale_radius,scale_height)
                
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

        return

############################################################
class Progenitor():
    def __init__(self, mass, host,
                 cosmology=None, zred=None, level=None, mstar=None,
                 orbit_init_method='li2020', xc=None, eps=None):
        """
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

        # The mass within rmax is used in the stripping calculations
        self.m_max_init = self.dens_profile.M(self.dens_profile.rmax)

        if orbit_init_method is None:
            # xv in cylindrical coordinates: np.array([R,phi,z,VR,Vphi,Vz])
            self.xc  = xc
            self.eps = eps
            self.xv  = init.orbit(self.init_host_dens_profile, xc=self.xc, eps=self.eps)
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
        return

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
    
    hc = prog.init_host_concentration
    pc = prog.concentration
    
    o = orbit(prog.xv)    
    xv     = o.xv 
    r      = np.sqrt(xv[0]**2+xv[2]**2)    
    r_init = r
    
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
        
        # Threshold values at resolution limit and skip explicit calculation of remaining steps
        # (i.e. propagate values at rehost_dpsolution limit forward.
        if (prog_mass <= mres_effective) or (r <= cfg.Rres) or ((prog_mass/prog_mass_init) <= cfg.phi_res):
            prog_status.append(STATUS_PROG_LOST)
            if not evolve_past_res_limits:
                radii.append(r)
                prog_masses.append(prog_mass)
                prog_mstars.append(prog_mstar)
                continue 
        else:
            prog_status.append(STATUS_PROG_INTACT)
            
        # Absolute levels in the tree
        start_step_level = levels_at_tstep[istep] - 1
        end_step_level   = start_step_level + 1

        # Update the host profile if needed
        hp = host.dens_profile[start_step_level]
        
        # Evolve the progenitor orbit based on the current mass
        # and host halo profile.
        
        o.integrate(t, host_dp, prog_mass)
        
        # Note that the coordinates are updated 
        # internally in the orbit instance "o" when calling
        # the ".integrate" method, here we assign them to 
        # a new variable "xv" only for bookkeeping
        xv  = o.xv 
        r   = np.sqrt(xv[0]**2+xv[2]**2)
        radii.append(r)

        if evolve_prog_mass:
            # Evolve the progenitor mass for dt in the current potential
            # Following SatGen (SatEvo), msub takes the initial potentials
            # and orbit at the start of the step.
            # dt is the length of the step (right? APC)
            alpha_strip = ev.alpha_from_c2(hc,pc)

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
            
    # Return
    retdict = dict()

    retdict['prog_masses'] = np.array(prog_masses)   
    retdict['prog_mstars'] = np.array(prog_mstars)  
    retdict['status']      = np.array(prog_status)        
    retdict['radii']       = np.array(radii)
    retdict['tsteps']      = tsteps
    retdict['tage']        = host.t_age[initial_level] + tsteps
    retdict['prog_dp']     = prog_dp
    retdict['levels_at_tsteps'] = levels_at_tstep
    retdict['host_times_starting_from_initial_level'] = host_times_starting_from_initial_level
    
    # Note that the orbit xvArray property contains the phase space coordinate at each 
    # timestep, but, since this this computed by SatGen internally, it does not include
    # the initial conditions or any steps below the resolution limit. TODO?
    retdict['orbit'] = o
    
    return retdict
