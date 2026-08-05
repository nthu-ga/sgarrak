import sys
import os
import time

import numpy as np
import tables as tb
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import astropy.cosmology as cosmo
from scipy.interpolate import interp1d
from scipy.stats import truncnorm

from astropy.table import Table
from astropy import constants as const
from astropy import units as u

from importlib import reload

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

# Setting the config parameters
cfg.h  = 0.73
cfg.Om = 0.25
cfg.Ob = 0.0465
cfg.OL = 0.75
cfg.s8 = 0.8
cfg.ns = 1.
# These are the default dynamical friction parameters
cfg.lnL_type = 0
cfg.lnL_pref = 0.75
# I remember at some point we want to use this paramters, not the default.
# cfg.lnL_type = 3
# cfg.lnL_pref = 1

LUDLOW_C_PATH = '/data/chungwen/cwt/ludlowc/ludlowc/py/ludlowc'
if not LUDLOW_C_PATH in sys.path:
    sys.path.append(LUDLOW_C_PATH)

import ludlowc as llc

# Set cosmology for colossus
# SatGen use M200, but how about orther physical models we add like EMERGE?
#COLOSSUS_PATH = '/data/chungwen/colossus/colossus'
#if not COLOSSUS_PATH in sys.path:
#    sys.path.append(COLOSSUS_PATH)

#astropy_cosmo = cosmo.FlatLambdaCDM(hubble_parameter*100,0.25,Ob0=cfg.Ob)

#from colossus.cosmology import cosmology
#from colossus.halo import mass_defs

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
def mod_Mstar(hm,h=0.7,z=0.,model='m18',task='Mstar',**kwargs):
    """
    Customized Mstar 
    The halo mass from our eps tree already dealt with h
    
    B13 assumed parameters: Omega_M = 0.27
                            Omega_lambda = 0.73
                            h = 0.7
                            ns = 0.95
                            sigma8 = 0.82

    B19 assumed parameters: h = 0.678

    m18 adjusts h in the parameters. delta_t is now total age.
    It only gives Mstar for the smhm relation.
    m18 assume a baryonic fraction value, but satgen uses cfg.Ob/cfg.Om.

    Parameters: h: Hubble parameter assumed in the merger tree
                hm: linear halo mass

    Return: linear stellar mass

    """

    if model == 'B13':
        h_model = 0.7
        hm *= (h/h_model)
        if task=='Mstar':
            sm = init.Mstar(hm,z,choice=choice)
        elif task=='lgMs_B13':
            sm = 10**gh.lgMs_B13(np.log10(hm),z)
        else:
            raise ValueError(f"Invalid task '{task}' for choice 'B13'")
        
    elif model == 'RP17':
        h_model = 0.7 # Not used, needs to check
        hm *= (h/h_model)
        if task=='Mstar':
            sm = init.Mstar(hm,z,choice=choice)
        elif task=='lgMs_RP17':
            sm = 10**gh.lgMs_RP17(np.log10(hm),z)
        else:
            raise ValueError(f"Invalid task '{task}' for choice 'RP17'")

    elif model == 'B19':
        h_model = 0.678
        hm *= (h/h_model)
        raise NotImplementedError('B19: Not provided yet')

    elif model == 'm18':
        h_model = 0.678
        fb = 0.156 # cfg.Ob/cfg.Om
        e,sigma = integrated_baryon_conversion_eff(z,hm,h=h_model)
        # Should it have a lower limit of hm*fb?
        sm = np.minimum(fb*hm,10**(np.random.normal(np.log10(fb*hm*e),sigma)))

    sm *= (h_model/h)
    return sm 

############################################################
### EMERGE
## EMERGE is an empirical model for star formation. see Moster et al. 2018
## NOTE: EMERGE only use 'forward' method.
#        (Although the logic of tdyn is current time - tdyn.)
#        EMERGE uses h=0.6781
def mstar(delta_t,z,
          M_dot_dyn,Rv_dot_dyn,e,dens_profile,
          h=0.7):
    """
    The main part of the equations. It calculates the stellar mass growth 
    between each time step.

    Syntax: stellar mass growth = (in-situ formation - dying stars)
    
    f_loss is from Chabrier 2003 IMF    

    Parameters: delta_t: t-t'
                M_dot_dyn: halo growth rate over a dynamical time
                Rv_dot_dyn: virial radius growth rate over a dynamical time
                e: instantanious baryon conversion efficiency
                dens_profile: halo density profile

    Return: Stellar mass increased in delta_t

    NOTE: 
    m_loss_dot does not include the instantaneous recycling.

    macc is not needed for our purpose.

    cww:
    The model only fit to z=10?
    """

    mstar_dot = sfr(M_dot_dyn,Rv_dot_dyn,e,dens_profile,z)
    f_loss = 0.05 * np.log(1+delta_t/(1.4/1e3)) # eq.12 in Gyr
    delta_mstar = delta_t * (mstar_dot * (1 - f_loss)) # eq.11
    return delta_mstar

############################################################
def sfr(M_dot_dyn,Rv_dot_dyn,e,dens_profile,z):
    """
    The halo mass growth due to accretions is the accreted masses - pseudo evolution of
    the virial radius growth over time as the background density decrease.

    Syntax: M_dot = <M_dot>_dyn - 4 * pi * Rv^2 * rho(Rv) * <Rv_dot>_dyn

    The baryonic growth rate at two time steps is the baryons brought from mergers
    
    Syntax: mb_dot = fb * M_acc

    The stellar mass forms between two time steps is the mb_dot times the instantaneous 
    baryonic conversion efficiency.

    Syntax: mstar_dot = mb_dot * e

    Parameter: M_dot_dyn: halo growth rate over a dynamical time
               Rv_dot_dyn: virial radius growth rate over a dynamical time
               e: instantanious baryon conversion efficiency
               dens_profile: halo density profile
               z: redshift

    NOTE: The cosmology in our model does not set Ob0. Maybe this needs to be changed.

    cww:
    This method acctually calculate the baryonic budget from mergers, which solves the 
    ambiguity of gas brought by accreted subhalos when using a purely SMHM relation.
    
    """

    # Cosmology we use
    Omega_b = 0. 
    Omega_m = 0.25
    # Omega_b/Omega_m from Moster18
    fb = 0.156 
    #Omega_b = 0.0484
    #Omega_m = 0.308

    # virial radius
    Rv = dens_profile.rh
    # density at the virial radius.
    rho_Rv = dens_profile.rho(Rv,z)

    # halo growth rate due to accretion and pseudo virial radius growth
    M_dot = M_dot_dyn - 4 * np.pi * Rv**2 * rho_Rv * Rv_dot_dyn # eq.4

    # baryonic growth rate
    mb_dot = fb * M_dot # eq.2

    mstar_dot = mb_dot * e # eq.1
    return mstar_dot

############################################################
def instant_baryon_conversion_eff(M,z,h=0.7):
    """
    Instantaneous baryon conversion efficiency e
    
    Table 6 is the fitting result from MCMC.
    """

    # Table 6; best fit
    M0 = 11.339
    Mz = 0.692
    e0 = 0.005
    ez = 0.689
    beta0   = 3.344
    betaz   = -2.079
    gamma0  = 0.966
    h_model = 0.6781

    # h correction
    M0 += np.log10(h_model/h)
    Mz += np.log10(h_model/h)
    
    a = 1/(z+1)
    lgM1 = M0 + Mz*(1-a)          # eq.7
    en = e0 + ez * (1-a)          # eq.8
    beta  = beta0 + betaz * (1-a) # eq.9
    gamma = gamma0                # eq.10
    M1 = 10**lgM1
    return 2 * en / ((M/M1)**-beta + (M/M1)**gamma) # eq.5

############################################################
def calculate_tdyn(age,mass,conc,zred,h=0.7,return_lgmass=False):
    """

    Syntax:
    tdyn = (Rv^3/GM)^0.5

    Parameters: age: age of main halos in Gyr
                mass: linear mass of main halos
                conc: concentration from NFW
                zred: redshift
                return_lgmass: This gives mass and Rv in log10 for
                               analysis.

    cww:
    Maybe h affects the mass and the virial radius here?

    """

    #G = const.G.to(u.kpc**3/u.M_sun/u.Gyr**2).value

    halo_profile = NFW(mass,conc,Delta=200.,z=zred,sf=1.)
    Rv = halo_profile.rh
    tdyn_pf = halo_profile.tdyn(Rv)

    # Interpolate in log mass
    lgmass = np.log10(mass)
    fm = interp1d(age,lgmass,fill_value='extrapolate')
    fr = interp1d(age,Rv,fill_value='extrapolate')

    m_dot_dyn  = []
    Rv_dot_dyn = []
    lgm_dot_dyn = []
    tdyn_cal = []
    for rv,m,ag in zip(Rv,mass,age):
        tdyn = (rv**3/cfg.G/m)**0.5 # Gyr
        tdyn_cal.append(tdyn)
        if ag<tdyn:
            raise ValueError(f"At the age '{ag}', the tdyn is larger")
        m_dot_dyn.append((m-10**fm(ag-tdyn))/tdyn) #eq.3
        Rv_dot_dyn.append((rv-fr(ag-tdyn))/tdyn)
        lgm_dot_dyn.append(np.log10(m-10**fm(ag-tdyn))/tdyn)

    if return_lgmass:
        return lgm_dot_dyn,Rv_dot_dyn,tdyn_pf,tdyn_cal,Rv

    return m_dot_dyn,Rv_dot_dyn,tdyn_pf,tdyn_cal,Rv

############################################################
def integrated_baryon_conversion_eff(zred,mass,h=0.7):
    """
    Integrated baryon conversion efficiency in Moster18, for computing the
    EMERGE SMHM relation.

    Parameters: zred: redshift. scalar
                mass: halo mass. array or scalar
    """

    # coefficients, table 8 all centrals
    coef = dict()
    z               = np.array([0.1,0.5,1.0,2.0,4.0,8.0])
    coef['M1']      = np.array([11.80,11.85,11.95,12.,12.05,12.10])
    coef['en']      = np.array([0.14,0.16,0.18,0.18,0.19,0.24])
    coef['beta']    = np.array([1.75,1.70,1.60,1.55,1.50,1.30])
    coef['gamma']   = np.array([0.57,0.58,0.60,0.62,0.64,0.64])
    coef['M_sigma'] = np.array([10.80,10.70,10.60,10.50,10.40,10.30])
    coef['sigma0']  = np.array([0.16,0.14,0.12,0.10,0.08,0.02])
    coef['alpha']   = np.array([1.0,0.90,0.75,0.50,0.40,0.10])

    m18_h = 0.6781

    # h correction
    coef['M1'] += np.log10(m18_h/h)
    coef['M_sigma'] += np.log10(m18_h/h)

    # interpolate, masses are in log.
    for k,d in coef.items():
        f = interp1d(z,d,fill_value='extrapolate')
        coef[k] = f(zred)

    # uncertainty, eq.25
    sigma = coef['sigma0'] + np.log10((mass/10**coef['M_sigma'])**(-coef['alpha']) + 1)

    # eq.5
    e = 2 * coef['en'] / ((mass/10**coef['M1'])**(-coef['beta']) + (mass/10**coef['M1'])**coef['gamma'])
    return e,sigma 

############################################################
class Host():
    def __init__(self, mass, zred, cosmology, fd=0.0, flattening=0., disk_method='fd',
                 output_zred=None, walk_tree='backward', smhm='m18', concentration_method='ludlow',
                 cooling_threshold=False, z0_smhm=False):
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
        #colossus_cosmo = cosmology.fromAstropy(self.cosmology, cfg.s8, cfg.ns, cosmo_name = 'my_cosmo')

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
            if concentration_method=='zhao':
                self.concentration = halo_mah_to_zhao_c_nfw(self.mass, self.t_age)
            elif concentration_method=='smooth':
                self.concentration = smooth_c(self.mass,self.t_age,version='zhao')
            # Ludlow 2016
            elif concentration_method=='ludlow':
                self.concentration = llc.ludlow_concentration(self.mass,
                                        np.where(self.zred <= 7, self.zred, 7),self.cosmology)
            else:
                raise ValueError(f"This concentration method '{concentration}' not support.")
        else:
            self.concentration = np.atleast_1d(init.concentration(self.mass[0], self.zred[0], choice='DM14'))
            
        # Convert Mvir and Cvir to M200 and C200
        # cww: Is is reasonable to Mvir to input ludlow concentration which is based on M200?
        #self.mass, R200c, self.concentration = mass_defs.changeMassDefinition(self.mass, self.concentration, self.zred, 'vir', '200c')

        if disk_method in ('interp_zavg'):
            # Add a check for those disk_methods don't have forward method, so idx_iter does not accidentally
            # reverse the index.
            if walk_tree=='forward':
                raise ValueError(f"This disk method '{disk_method}' does not support forward method.")
        elif disk_method=='EMERGE':
            # Progenitor infos for mstar_acc...
            # I think this step will break the self.interpolated part (zred & output_zred),
            # unless it also gets interpolated. Or just remove this output_zred?
            if walk_tree=='backward':
                raise ValueError(f"This disk method '{disk_method}' only integrate stellar mass forward.")

        # halo growth rate over the dynamical time
        M_dot_dyn,Rv_dot_dyn,self.tdyn_pf,self.tdyn_cal,self.Rv = calculate_tdyn(self.t_age,self._tree_mass,self.concentration,self.zred,return_lgmass=False)


        # Make a profile for each redshift
        self.dens_profile = list()
        self.halo_dens_profile = list()
        self.has_disk = fd > 0

        if self.has_disk:

            mean_sm_z0   = mod_Mstar(self.mass[0],h=0.73,z=0.,choice=smhm,task='lgMs_B13')
            mean_sm_nlev = mod_Mstar(self.mass[self.nlev-1],h=0.73,z=self.zred[self.nlev-1],choice=smhm,task='lgMs_B13')
 
            if walk_tree == 'backward':
                starting_disk_mass = mod_Mstar(self.mass[0],h=0.73,z=self.zred[0], choice=smhm,task='Mstar')
            elif walk_tree=='forward':
                # Assign starting mass in the iteration
                starting_disk_mass = 0
                # FIXME this breaks the non-step methods
                #starting_disk_mass = init.Mstar(self.mass[self.nlev-1], 
                #                                self.zred[self.nlev-1], choice='B13')
            else:
                # Add a safety check so that the arrays are not reverted for disk methods 
                # that do not have forward method
                raise ValueError(f"walk tree method '{walk_tree}'  not supported")

            prev_disk_mass = starting_disk_mass

            # Save the scale radius for checking
            self.scale_radius = list()

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
                # 3D half-stellar-mass radius
                Reff = gh.Reff(halo_profile.rh,conc_i) # Virial radius & concentration
                scale_radius = 0.766421/(1.+1./flattening) * Reff
                self.scale_radius.append(scale_radius)
                scale_height = scale_radius / flattening

                if disk_method=='interp':
                    # Use z=0 SMHM relation to linearly scale down the stellar mass
                    # with the difference of the halo masses.
                    if walk_tree == 'backward':
                        disk_mass = starting_disk_mass * (mass_i/self.mass[0])
                    else:
                        disk_mass = starting_disk_mass * (mass_i/self.mass[self.nlev-1])
                    self.disk_mass.append(disk_mass)

                elif disk_method == 'interp_sm':
                    # Use the mean z=0 SMHM relation to get the stellar mass difference.
                    # And use the difference to scale down the z=0 disk mass.
                    mean_sm_ilev = mod_Mstar(mass_i,h=0.73,z=z_i,choice='B13',task='lgMs_B13')
                    if walk_tree == 'backward':
                        disk_mass = starting_disk_mass * (mean_sm_ilev/mean_sm_z0)
                    else:
                        disk_mass = starting_disk_mass * (mean_sm_ilev/mean_sm_nlev)
                    self.disk_mass.append(disk_mass)

                #elif disk_method == 'roll_dice':
                #    # Take the stellar mass as long as it goes up in the past
                #    # If init.Mstar finds a higher mass, do it again.
                #    # This might cost many resources if the current level gives the lowest
                #    # stellar mass (The chance of finding a lower stellar mass at eairlier 
                #    # time is lower).`
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
                    
                    # For the forward method, we can require that the disk
                    # mass only grows if a halo mass is above the cooling
                    # threshold.

                    # Adds disk mass with z0 SMHM relation for checking.
                    if z0_smhm: 
                        disk_mass = mod_Mstar(mass_i,h=0.73,z=0.,choice='B13',task='Mstar')
                    else:
                        disk_mass = mod_Mstar(mass_i,h=0.73,z=z_i,choice='B13',task='Mstar')
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
                            # Check additional cooling threshold to grow disk mass
                            if cooling_threshold:
                                if grow_disk:
                                    # The halo can cool gas, disk grows
                                    self.disk_mass.append(disk_mass)
                                    prev_disk_mass = disk_mass
                                else:
                                    # The halo can't cool gas, no growth
                                    self.disk_mass.append(prev_disk_mass)
                            else:
                                self.disk_mass.append(disk_mass)
                                prev_disk_mass = disk_mass
                        else:
                            # Disk can't be less massive in the future;
                            # force no change in mass for this step.
                            self.disk_mass.append(prev_disk_mass)

                elif disk_method == 'EMERGE':
                    if i==131:
                        
                        disk_mass = mod_Mstar(mass_i,h=0.73,z=z_i,choice=smhm)
                        self.disk_mass.append(disk_mass)
                    else:
                        e = instant_baryon_conversion_eff(mass_i,z_i,h=0.73)
                        t = self.t_age[i]
                        delta_t = t - self.t_age[i+1]
                        delta_disk_mass = mstar(delta_t,z_i,
                                          M_dot_dyn[i],Rv_dot_dyn[i],e,halo_profile,
                                          h=0.73)
                        disk_mass += delta_disk_mass
                        self.disk_mass.append(disk_mass)



                elif disk_method == 'fd':
                    # Use a fixed disk mass fraction
                    disk_mass = fd * mass_i
                    self.disk_mass.append(disk_mass)
                else:
                    raise ValueError(f"Disk method {disk_method:s} not supported")
                
                self.disk_reff.append(Reff)

                disk_profile = MN(disk_mass,scale_radius,scale_height)

                # Pre-generate the interpolator for M(<r) by computing M(<10kpc)
                disk_profile.M(10,0)

                self.dens_profile.append([halo_profile, disk_profile])
                self.halo_dens_profile.append(halo_profile)
                self.disk_dens_profile.append(disk_profile)

        else:
            ## Debug test
            # Create NFW object with array
            #halo_profiles = NFW(self.mass,self.concentration,Delta=200.,z=self.zred,sf=1.)
            #self.dens_profile = halo_profiles
            #self.halo_dens_profile = halo_profiles
            for i in range(self.nlev):
                halo_profile = NFW(self.mass[i],
                                   self.concentration[i],
                                   Delta=200.,
                                   z=self.zred[i],
                                   sf=1.)
                # The host and halo density profiles are identical in this case
                self.dens_profile.append(halo_profile)
                self.halo_dens_profile.append(halo_profile)

        self.rs = [hhdp.rs for hhdp in self.halo_dens_profile]

        # Reverse the array for forward method, so no need to change other function.
        if disk_method in ('step', 'interp', 'interp_sm', 'EMERGE'):
            if walk_tree != 'backward':
                self.disk_mass = self.disk_mass[::-1]
                self.disk_reff = self.disk_reff[::-1]
                self.scale_radius = self.scale_radius[::-1]
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

############################################################
def threshold_check(hm,z):
    """
    Returns True if the halo mass hm is above the threshold for atomic hydrogen
    cooling at the redshift z.

    The cooling threshold check is only taken at before mergers.
    
    Parameters: hm: linear progenitor halo mass before the merger
                z: reshift before the merger
    """
    if z<10:
        _,threshold = Tvir_threshold_rez10fit()
    else:
        threshold,_ = Tvir_threshold_rez10fit()
    
    return np.log10(hm) > threshold(z)

############################################################
class Progenitor():
    def __init__(self, mass, host,
                 cosmology=None, zred=None, level=None, mstar=None,
                 orbit_init_method='li2020', xc=None, eps=None, xv=None,
                 conc=None, mstar_shift=None,smhm='m18'):
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
            self.cosmology = host.cosmology

        #colossus_cosmo = cosmology.fromAstropy(self.cosmology, cfg.s8, cfg.ns, cosmo_name = 'my_cosmo')


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
        self.init_host_mass = self.host.mass[self.level]
        self.init_host_rs = self.host.rs[self.level]

        if host.has_disk:
            self.init_disk_mass = self.host.disk_mass[self.level]

        # Draw progenitor concentration
        if conc is None:
            self.concentration = init.concentration(self.mass,self.zred,choice='DM14')
        else:
            self.concentration = conc

        # The halo potential is a "Green" profile; an NFW with additional
        # methods to adjust for the effects of tidal stripping.
        self.dens_profile = Green(self.mass,self.concentration,Delta=200,z=self.zred)

        # Draw stellar mass from mstar-mhalo releation
        if mstar is None:
            if mstar_shift is None:
                self.mstar_init = mod_Mstar(self.mass_init,h=0.7,z=self.zred,choice=smhm)
            else:
                # Add a shift on the SMHM relation for progenitors to simulate different 
                # formation models (?).
                self.mstar_init = mod_Mstar(self.mass_init,h=0.7,z=self.zred,choice=smhm)
        else:
            self.mstar_init = mstar
        self.mstar = self.mstar_init

        # Check if the progenitor above the cooling threshold at infall.
        # It would be better to check the entire assembly history.
        # This is only a flag, the prog mstar is still computed by the smhm relation even not above the threshold.
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
        elif orbit_init_method == 'xv':
            self.xv = xv
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

def smooth_c(Mv,t,version='zhao'):
    """
    Setting 4% of the z=0 halo mass, as it gives a smoother concentration.
    """
    if(version == 'vdb'):
        coeff1 = 3.40
        coeff2 = 6.5
    elif(version == 'zhao'):
        coeff1 = 3.75
        coeff2 = 8.4
    idx = aux.FindNearestIndex(Mv,0.04*Mv[0])
    return 4.*(1.+(t/(coeff1*t[idx]))**coeff2)**0.125

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
def cylindrical_to_cartesian_phase(R, phi, z, VR, Vphi, Vz):
    """
    Convert 6D cylindrical (R,phi,z,VR,Vphi,Vz)
    to Cartesian (x,y,z,vx,vy,vz).

    Parameters
    ----------
    R, phi, z, VR, Vphi, Vz : array_like
        Cylindrical position–velocity components.

    Returns
    -------
    x, y, z, vx, vy, vz : ndarray
        Cartesian components, same broadcasted shape.
    """
    R  = np.asarray(R)
    phi = np.asarray(phi)
    z  = np.asarray(z)
    VR = np.asarray(VR)
    Vphi = np.asarray(Vphi)
    Vz = np.asarray(Vz)

    # Position
    x = R * np.cos(phi)
    y = R * np.sin(phi)

    # Velocities
    vx = VR * np.cos(phi) - Vphi * np.sin(phi)
    vy = VR * np.sin(phi) + Vphi * np.cos(phi)
    vz = Vz

    return x, y, z, vx, vy, vz

############################################################
def valid_coors(coors):
    """
    This function filters [-1,-1,-1] in the coordinate array.

    [-1,-1,-1] coordinates were fiiled in when a progenitor was·
    below the cfg resolution to keep the arrays having the same shape.
    """

    mask = ~(np.all(coors == -1, axis=1))
    coors_valid = coors[mask]

    return coors_valid

############################################################
def reshape_to_tsteps_shape(input_array, t_shape, task='coor'):
    """
    Pad an array along the time axis to have length t_shape.

    Parameters
    ----------
    input_array : array-like
        Shape (n_steps, ndim) or list of length n_steps.
    t_shape : int
        Desired number of timesteps.
    task : {'coor', 'xv'}
        'coor' -> 3D positions, pad with [-1, -1, -1]
        'xv'   -> 6D phase space, pad with 6 x -1

    Returns
    -------
    arr : np.ndarray
        Array of shape (t_shape, ndim), padded with -1 rows
        at the end if input is shorter. If already length
        t_shape, returned unchanged. If longer, raises.
    """

    if task == 'coor':
        below_res_arr = [-1, -1, -1]
    elif task == 'xv':
        below_res_arr = [-1, -1, -1, -1, -1, -1]
    else:
        raise ValueError('Not a task')

    # Convert to array
    arr = np.asarray(input_array)

    # Handle the 1D case like a list of rows
    ndim = len(below_res_arr)
    if arr.ndim == 1:
        # Try to infer if it's a single vector (len=ndim)
        # or a flat list of length n_steps with unknown layout.
        if arr.size == ndim:
            arr = arr.reshape(1, ndim)
        else:
            # If this happens in your use case, we can refine this logic.
            raise ValueError(f"Cannot interpret 1D input of length {arr.size} as (n_steps, {ndim}).")

    if arr.shape[1] != ndim:
        raise ValueError(
            f"Expected second dimension {ndim} for task='{task}', "
            f"got shape {arr.shape}."
        )

    n_steps = arr.shape[0]
    if n_steps == t_shape:
        return arr
    if n_steps > t_shape:
        raise ValueError(
            f"Input has more timesteps ({n_steps}) than t_shape={t_shape}."
        )

    # Need to pad with -1 rows
    below_res_steps = t_shape - n_steps
    pad_block = np.full((below_res_steps, ndim), -1, dtype=arr.dtype)

    return np.vstack([arr, pad_block])

############################################################
def nfw_aux(x):
    return np.log(1.+x) - x/(1.+x)

def calculate_nfw_acc(M,r,rs,c):
    return cfg.G*M/r**2*nfw_aux(r/rs)/nfw_aux(c)
############################################################
def evolve_orbit(host, prog ,tsteps=None, 
                 evolve_prog_mass=False, 
                 evolve_past_res_limits=False,
                 alpha_shift=None):
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
    prog_circularity = [prog.circularity_init]
    prog_coors  = [cyl_to_cart_position(prog.xv)]    
    prog_xv = []
    acceleration_fgrav = []
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
    # For j_circ
    host_halo_dp = copy.deepcopy(prog.init_host_halo_dens_profile)

    prog_m_max_init = prog_dp.M(prog_dp.rmax)

    # Store the initial concentraiton for use in the mass loss routine
    init_pc = prog.concentration
    
    o = orbit(prog.xv)    
    o_copy = copy.deepcopy(o)
    orbit_object_list = [o_copy]
    xv     = o.xv 
    prog_xv.append(xv)

    # Store the initial acceleration
    if host.has_disk:
        halo_fgrav = host_dp[0].fgrav(xv[0],xv[2])
        disk_fgrav = host_dp[1].fgrav(xv[0],xv[2])
        fgrav = tuple(hf + df for hf, df in zip(halo_fgrav, disk_fgrav))
        acceleration_fgrav.append(np.sqrt(fgrav[0]**2+fgrav[2]**2))
    else:
        fgrav = host_dp.fgrav(xv[0],xv[2])
        acceleration_fgrav.append(np.sqrt(fgrav[0]**2+fgrav[2]**2))

    r      = np.sqrt(xv[0]**2+xv[2]**2)    
    r_init = r

    j_tot  = xv_to_j_tot(xv,prog_mass)
    j_circ = profile_j_circ(host_halo_dp, r, mass=prog_mass)
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
        # searchsorted right gives 0 for the last element, therefore return -1 in the final element in start_step_level.
        levels_at_tstep[-1] += 1

    if cfg.Mres is None:
        mres_effective = 0
    else:
        mres_effective = cfg.Mres

    # This is for debugging
    host_concentration = prog.init_host_concentration
    host_mass = prog.init_host_mass
    host_rs = prog.init_host_rs
    nfw_acc = []
    nfw_acc.append(calculate_nfw_acc(host_mass,r,host_rs,host_concentration))

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
                nfw_acc.append(calculate_nfw_acc(host_mass,r,host_rs,host_concentration))
                if host.has_disk:
                    host_disk_masses.append(host.disk_mass[start_step_level])
                    halo_fgrav = host_dp[0].fgrav(xv[0],xv[2])
                    disk_fgrav = host_dp[1].fgrav(xv[0],xv[2])
                    fgrav = tuple(hf + df for hf, df in zip(halo_fgrav, disk_fgrav))
                    acceleration_fgrav.append(np.sqrt(fgrav[0]**2+fgrav[2]**2))
                else:
                    fgrav = host_dp.fgrav(xv[0],xv[2])
                    acceleration_fgrav.append(np.sqrt(fgrav[0]**2+fgrav[2]**2))

                continue
        else:
            prog_status.append(STATUS_PROG_INTACT)

        # Update the host profile if needed
        host_dp = copy.deepcopy(host.dens_profile[start_step_level])
        host_concentration = host.concentration[start_step_level]
        host_rs = host.rs[start_step_level] # Debug usage
        host_mass = host.mass[start_step_level] # Debug usage
        if host.has_disk:
            host_disk_mass = host.disk_mass[start_step_level]

        #host_dp_steps.append(host_dp)

        # Evolve the progenitor orbit based on the current mass
        # and host halo profile.
        o.integrate(t, host_dp, prog_mass)
        o_copy = copy.deepcopy(o)
        orbit_object_list.append(o_copy)
        # Note that the coordinates are updated 
        # internally in the orbit instance "o" when calling
        # the ".integrate" method, here we assign them to 
        # a new variable "xv" only for bookkeeping
        xv  = o.xv

        prog_xv.append(xv)
        prog_coors.append(cyl_to_cart_position(xv))
        r   = np.sqrt(xv[0]**2+xv[2]**2)
        radii.append(r)

        # Compute circularity
        # FIXME possible consitency issues here
        # - computing before mass update, ok?
        # - dp includes disk or not?
        j_tot  = xv_to_j_tot(xv,mass=prog_mass)
        j_circ = profile_j_circ(host_halo_dp, r, mass=prog_mass)
        prog_circularity.append(j_tot/j_circ)

        # Save the accelerations
        nfw_acc.append(calculate_nfw_acc(host_mass,r,host_rs,host_concentration)) # Debug usage
        # There are two types of acceleration-ish functions (phi & fgrav)
        if host.has_disk:
            halo_fgrav = host_dp[0].fgrav(xv[0],xv[2])
            disk_fgrav = host_dp[1].fgrav(xv[0],xv[2])
            fgrav = tuple(hf + df for hf, df in zip(halo_fgrav, disk_fgrav))
            acceleration_fgrav.append(np.sqrt(fgrav[0]**2+fgrav[2]**2))
        else:
            fgrav = host_dp.fgrav(xv[0],xv[2])
            acceleration_fgrav.append(np.sqrt(fgrav[0]**2+fgrav[2]**2))

        if evolve_prog_mass:
            # Evolve the progenitor mass for dt in the current potential
            # Following SatGen (SatEvo), msub takes the initial potentials
            # and orbit at the start of the step.
            # dt is the length of the step (right? APC)
            # SatGen requires the *initial* progenitor concentration and the
            # *instantaneous* host concentration
            alpha_strip = ev.alpha_from_c2(host_concentration,init_pc)
            if alpha_shift is not None:
                alpha_strip *= alpha_shift
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
          
    prog_coors = reshape_to_tsteps_shape(np.array(prog_coors),len(tsteps))

    prog_xv = np.array(prog_xv)
    valid_prog_xv = valid_coors(prog_xv)
    x, y, z, vx, vy, vz = cylindrical_to_cartesian_phase(
    valid_prog_xv[:,0], valid_prog_xv[:,1], valid_prog_xv[:,2], valid_prog_xv[:,3],
    valid_prog_xv[:,4], valid_prog_xv[:,5])
    prog_pos_vel = np.array([x,y,z,vx,vy,vz]).T
    prog_pos_vel = reshape_to_tsteps_shape(prog_pos_vel,len(tsteps),task='xv')
  
    # Return
    retdict = dict()

    retdict['prog_masses'] = np.array(prog_masses)   
    retdict['prog_mstars'] = np.array(prog_mstars)  
    retdict['status']      = np.array(prog_status) 
    if host.has_disk:
        retdict['host_disk_masses'] = np.array(host_disk_masses)
        retdict['scale_raidus'] = np.array(host.scale_radius)
    retdict['radii']       = np.array(radii)
    retdict['circularity'] = np.array(prog_circularity)
    retdict['tsteps']      = tsteps
    retdict['tage']        = host.t_age[initial_level] + tsteps
    retdict['prog_dp']     = prog_dp
    retdict['levels_at_tsteps'] = levels_at_tstep
    retdict['host_times_starting_from_initial_level'] = host_times_starting_from_initial_level
    retdict['has_galaxy']  = prog.has_galaxy
    retdict['prog_xv'] = prog_xv
    retdict['prog_pos_vel'] = prog_pos_vel
    # Note that the orbit xvArray property contains the phase space coordinate at each 
    # timestep, but, since this this computed by SatGen internally, it does not include
    # the initial conditions or any steps below the resolution limit. TODO?

    # Storing data for Agama check
    retdict['orbit'] = np.array(prog_coors)
    retdict['acc_fgrav'] = np.asarray(acceleration_fgrav)
    retdict['nfw_acc'] = np.asarray(nfw_acc)

    # Storing data for stripping method
    retdict['orbit_object'] = orbit_object_list
    retdict['host_dp']  = host.dens_profile
    return retdict
