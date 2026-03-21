# Integrate the particles after being stripped from the satellites

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

if not 'SATGEN_PATH' in globals():
    SATGEN_PATH = None

config_dir = os.getenv('SGARRAK_CONFIG_DIR')
if config_dir is None:
    config_dir = '~/.config/sgarrak'
config_path = os.path.expanduser(os.path.join(config_dir,'config.py'))

import importlib.util

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

if not SATGEN_PATH in sys.path:
    sys.path.append(SATGEN_PATH)
import config as cfg
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

######################################################
###

def post_integrate_stripped_time_varying_point_mass(mass,xv,host_dp,tage,status):
    """
    The orbit of stripped particles is only the Satgen o.integrate without the dynamical friction?
    But without a prog density profile and tidal stripping.

    Parameters: mass: progenitor mass
                xv: prog.xv at tsteps
                potential: host density profile at tsteps
                status: 0 for intact, 1 for lost

    Return: a list of arrays of coordinates

    Note: Maybe the satgen orbit integration consider more parameters.
          The host halo potential does not instantanious add up the stripped mass.

    Update: 2026/1/19 Storing the profile object may not be practical due to memory usage
            (and potentially writing a hdf5 array). Postprocessing might not be easy.
    """

    # no dynamical friction
    cfg.lnL_type = 0.
    cfg.lnL_pref = 0.

    # record the coordinates of stripped mass.
    strip_coors_list = []

    # mass loss at each step
    nsteps = len(tage)
    for i in range(nsteps):
        # only calculate stripped mass when the satellite is intact, no mass loss when a satellite is below res.
        if ststus[i]==0:
            # mass stripped at i step
            strip_mass = mass[i+1]-mass[i]
            o = orbit(xv)
        else:
            break
        # give a new variable for this stripped mass
        strip_i = i
        strip_coors = []
        while strip_i <= nsteps-1:
            o.integrate(tage[strip_i], host_dp[strip_i], strip_mass)
            xv  = o.xv
            strip_coors.append(compute_coordinates(xv))

            strip_i = strip_i+1
        strip_coors_list.append(strip_coors)

    return strip_coors_list

def integrate_stripped_point_mass(orbit,dm,host_dp_list,tage,starting_istep,nstep,levels_at_tstep,tage_formed,potential='varying'):
    """
    Run the orbit calcualtion inside the evolve loop

    Parameters: dm: mass differ at this step
                xv: 6d phase space velocity at this step
                host_dp: host density profile array (or instance)
                tage: age array

    Return: Nstep times 3D positions array

    Note: If not postprocessing, setting cfg needs be reset.
          It might be ok to approximate stripped star initial orbit to be the same as
          center, but outer dark matter will be far from the center.

          For the satgen/agama comparison, the nsteps is set to 10000. 
          It is impossible to do this in the loop for all dm.

          It seems for no dynamical friction, the orbit can be integrated and give 
          coordinates if dm=0. The stored coordinates has outer len=len(t)-1
          But for the one with dynamical friction, the orbit will not be integrated.
          The length will be len=len(t)-1-len(dm=0)
    """
    #print('dynamical friction check1: ',cfg.lnL_pref)
    # no dynamical friction
    cfg.lnL_type = 0.
    cfg.lnL_pref = 0.
    #print('dynamical friction check2: ',cfg.lnL_pref)
    #print('input orbit xv check: ',orbit.xv)

    # copy the current orbit object for integration
    strip_o = copy.deepcopy(orbit)

    # record the coordinates of stripped mass.
    strip_coors = []
    strip_tage  = []

    istep = starting_istep
    while istep <= nstep-1:
        if potential=='varying':
            start_step_level = levels_at_tstep[istep] -1
            host_dp_istep = host_dp_list[start_step_level]
        elif potential=='static':
            host_dp_istep = host_dp_list[starting_istep]
        strip_o.integrate(tage[istep],host_dp_istep,dm)
        xv = strip_o.xv
        strip_coors.append(compute_coordinates(xv))
        strip_tage.append(tage_formed + tage[istep])
        istep +=1

    # Reset dynamical friction for other tsteps
    cfg.lnL_type = 0.
    cfg.lnL_pref = 0.75
    # cfg.lnL_type = 3
    # cfg.lnL_pref = 1

    #print('dynamical friction check3: ',cfg.lnL_pref)
    return strip_coors,strip_tage


def mass_distribution(mass,coors):
    """
    Distribute the stripped mass on the integrated orbit to get 
    the density profile.
    The amount of mass that deposit on the orbit is proportional 
    to the time it spend at the location.

    Parameters: mass: the array of stripped mass
                coors: the coordinates of each stripped mass

    Return:
    """
    r = []
    equal_dm = []
    for i in len(mass):
        r.append(np.linalg.norm(coors[i],axis=1))
        # equal mass at each saved coordinates
        dm = mass[i]/len(coors[i])
        equal_dm.append(np.repeat(dm,len(r)))
    #r = np.concatenate(r)
    #equal_dm = np.concatenate(equal_dm)

    return np.array(r),np.array(equal_dm)


######################################################
### auxiliary functions

def compute_coordinates(xv):
    """
    Compute the 3D space phase xv [R,phi,z,VR,Vphi,Vz] to coordinates
····
    The orbit object is used to calculate the orbital evolution,
    so the length of the array is tsteps-1 because there is no
    integration at the initial step?
····
    """
    R,phi,Z = xv[0],xv[1],xv[2]

    X = R*np.cos(phi)
    Y = R*np.sin(phi)
    return np.array([X,Y,Z])







