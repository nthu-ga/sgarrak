#############################################################
### Try to set up a simple point mass.
### No mass loss, darkm amstter only and fixed host potential.
###
### From this test, it can rule out satgen version inconsistancy.
#############################################################
### import libary
import sys
import os
import time

import numpy as np
import tables as tb
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import astropy.cosmology as cosmo
from astropy.cosmology import z_at_value

from scipy.interpolate import interp1d
from scipy.stats import truncnorm

from astropy.table import Table
from astropy import constants as const
from astropy import units as u

import matplotlib.pyplot as pl
import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'serif'

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

# cosmology
hubble_parameter=0.73
cosmology = cosmo.FlatLambdaCDM(hubble_parameter*100,0.25)

#############################################################
### 
def evolve(dp,tage,pm,xv):
    """
    The orbital integration function

    Parameters: dp: the host density profile
                tage: the integration time
                pm: the test particle mass
                *xv: 6-D phase space position and velocity

    Return: cartesian coordinates xyz
    """
    # unpack xv
    xv = np.asarray(xv)
    R, phi, z, VR, Vphi, Vz = xv

    # the initial orbit of the point mass
    o = orbit(xv)

    # store coordinates
    init_coors = compute_coordinates(xv)
    pcoors = [init_coors]

    # integrate the orbit for 100 time steps
    nsteps = np.arange(0,len(tage))
    for istep in nsteps:
        t = tage[istep]
        o.integrate(t, dp, pm)
        coors = compute_coordinates(o.xv)
        pcoors.append(coors)

    return np.array(pcoors)

def compute_coordinates(xv):
    """
    Compute the 3D space phase xv [R,phi,z,VR,Vphi,Vz] to coordinates
    """
    R,phi,Z = xv[0],xv[1],xv[2]

    X = R*np.cos(phi)
    Y = R*np.sin(phi)
    return np.array([X,Y,Z])

def plot_coors(coors):
    """
    Visualized the orbits
    """
    pl.figure(figsize=(20,6))
    pl.subplot(131)
    pl.plot(coors[:,0],coors[:,1])
    pl.xlabel('x (kpc)')
    pl.ylabel('y (kpc)')
    
    pl.subplot(132)
    pl.plot(coors[:,0],coors[:,2])
    pl.xlabel('x')
    pl.ylabel('z')

    pl.subplot(133)
    pl.plot(coors[:,1],coors[:,2])
    pl.xlabel('y')
    pl.ylabel('z')

    pl.savefig('test_point_mass_coors.pdf')
    return
#############################################################
### Main program

if __name__ == '__main__':
    """
    """
    # 100 time steps over 10 Gyr
    time = np.linspace(0,10,100)
    # calculate the redshift of these time
    # maybe it is easier to always assume z=0 NFW profile
    #t0 = cosmology.age(0).to(u.Gyr).value   # age of Universe at z=0
    #t_cosmic = t0 - (10 - time)
    #redshift = np.array([z_at_value(cosmology.age, t * u.Gyr)
    #                    for t in t_cosmic])

    # test particle mass M_sun
    pmass = 10**8
    # initial particle 6D phase space coordinates and velocities
    pxv = [86.66355751,-0.67994895,26.44,-261.11727294,54.38706805,57.79]

    # host properties
    host_mass = 10**12 # host mass M_sun
    host_c = 10 # host concentration

    # a fixed host density profile
    dens_profile = NFW(host_mass,
                       host_c,
                       Delta=200.,
                       z=0,
                       sf=1.)

    coordinates = evolve(dens_profile,time,pmass,pxv)

    np.savetxt('test_particle_coors.txt',coordinates, fmt="%.6f")
    print('test particle coordinates written')
    plot_coors(coordinates)
    print('plot created')
