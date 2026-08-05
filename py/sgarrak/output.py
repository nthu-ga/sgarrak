#########################################################
### Modules for analysing satgen outputs
# This is the python module mainly base on the jupyter notebook
# that produces the technical note. The definition of different 
# accreted mass type is in "compute_mass" function.

### TODO & Comments

## Disk mass
# What we want to compute is the tidal disruption mass which is the 
# stellar halo. However, it is not possible to minus M_stellar_halo
# for the z=0 stellar mass to get the in-situ stellar mass, because
# the disruption events depend on the disk mass at previous time.
# Thus, It becomes a chicken and egg problem. The proper way is to use a 
# semi-analytical model to evolve the disk mass with time.
# The gas brought in when mergers also trigger star formation, so  
# it is not perfect to minus the accreted satellite mass directly.


#########################################################
### envs
import sys
from importlib import reload

SATGEN_PATH = '/data/chungwen/SatGen'
if not SATGEN_PATH in sys.path:
    sys.path.append(SATGEN_PATH)

SATGEN_ETC_PATH = '/data/chungwen/SatGen/etc'
if not SATGEN_ETC_PATH in sys.path:
    sys.path.append(SATGEN_ETC_PATH)

PY_PATH = '/data/chungwen/sgarrak/py/sgarrak'
if not PY_PATH in sys.path:
    sys.path.append(PY_PATH)

import sgarrak as sga
import strip as strip

LUDLOW_C_PATH = '/data/chungwen/cwt/ludlowc/ludlowc/py/ludlowc'
if not LUDLOW_C_PATH in sys.path:
    sys.path.append(LUDLOW_C_PATH)

import ludlowc as llc


sys.path.append('/data/apcooper/sfw/hdf5_tools/py/')
sys.path.append('/data/apcooper/sfw/apcpy3/py/')
sys.path.append('/data/apcooper/sfw/stings/py/')
sys.path.append('/data/apcooper/sfw/coco/py/')

import coco.system as COCO
import coco.runmanager as runmanager
import coco.density
import coco.astrophysics
reload(COCO)
reload(runmanager)
reload(coco.density)
reload(coco.astrophysics)
l153pc = runmanager.CocoRun('cdm','lacey15','3pc','trees_mmax_vmax')
l153pc.load_standard_tables()

density_profiles = coco.density.DensityProfiles(l153pc)

galaxies = coco.runmanager.GalaxySet(l153pc)

import numpy as np
import os
import time

np.bool = bool # This is a quick fix for galaxies.subset
import hmf
from hmf import MassFunction

import astropy.cosmology as cosmo

from astropy.io import fits
from astropy.table import Table
from functools import partial
import tables as tb
from scipy.interpolate import interp1d
from scipy.integrate import quad

# <<< for clean on-screen prints, use with caution, make sure that 
# the warning is not prevalent or essential for the result
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import config as cfg
import cosmo as co
import evolve as ev
from   profiles import NFW,Dekel,MN,Einasto,Green
from   orbit import orbit
import galhalo as gh
import aux
import init

import matplotlib.pyplot as pl
from matplotlib import cm
import matplotlib.colors as mcolors

# Millennium
hubble_parameter = 0.73
OMEGA_B = 0.04455
cosmology = cosmo.FlatLambdaCDM(hubble_parameter*100,0.25)

#########################################################
### Read data
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

def read_b18():
    """
    """
    b18smhm_path = "/data/apcooper/coco/obs/behroozi18/umachine-dr1/data/smhm/median_raw/smhm_a1.002312.dat"
    b18smhm_scatter_path = "/data/apcooper/coco/obs/behroozi18/umachine-dr1/data/smhm/median_raw/smhm_scatter_a1.002312.dat"

    b18smhm = Table.read(b18smhm_path,format='ascii.commented_header')
    b18smhm_scatter = Table.read(b18smhm_scatter_path,format='ascii.commented_header')

    return b18smhm, b18smhm_scatter

def read_tree(tree_setup='default'):
    """
    pchtrees output files

    Parameters
    ----------
    tree_setup: different mass ranges
    """

    # Millennium
    hubble_parameter = 0.73
    OMEGA_B = 0.04455

    cosmology = cosmo.FlatLambdaCDM(hubble_parameter*100,0.25)
    if tree_setup=='default':
        tree_file = '/data/chungwen/sgarrak/pchtrees/runs/1000_mixed_logmass/output_satgen_1000_mixed_logmass.hdf5'
    elif tree_setup=='extend':
        tree_file = '/data/chungwen/sgarrak/pchtrees/runs/1000_mixed_logmass/debug_extend_mass_range_run/output_satgen_1000_mixed_logmass.hdf5'
    elif tree_setup=='lower':
        tree_file = '/data/chungwen/sgarrak/pchtrees/runs/1000_mixed_logmass/debug_lower_mass_range_run/output_satgen_1000_mixed_logmass.hdf5'
    else:
        ValueError(f"Invalid tree file '{tree_setup}'")
    tree_main_branch_masses = read_hdf5(tree_file,'/Mainbranch/MainbranchMass')
    tree_main_branch_masses = tree_main_branch_masses/hubble_parameter

    ntrees, nlev = tree_main_branch_masses.shape

    progenitor_dataset_names = ["HostMass","ProgenitorZred","ProgenitorMass","ProgenitorIlev","TreeID"]

    progenitors = read_hdf5(tree_file,progenitor_dataset_names,group='/Progenitors')
    progenitors['ProgenitorMass'] = progenitors['ProgenitorMass']/hubble_parameter
    progenitors['HostMass']       = progenitors['HostMass']/hubble_parameter

    tree_redshifts = read_hdf5(tree_file,'Redshift',group='/OutputTimes')
    tree_t_lbk_gyr = cosmology.lookback_time(tree_redshifts).value
    tree_t_age_gyr = cosmology.age(tree_redshifts).value

    root_mass = tree_main_branch_masses[:,0]

    return tree_redshifts,root_mass,progenitors,tree_main_branch_masses


def read_satgen(ver=3):
    """
    Satgen output files

    Recipies
    --------
    disk: fd=0/fd/EMERGE/interp/interp_sm/step
    fd=0: No disk
    interp: Use the z=0 B13 SMHM relation to get the disk mass,
                           and rescale to the earlier halo mass
    Runs
    ----
    reionization/EMERGE/ludlow, status: 2026/4/26 updated

    ver: lessoutput: Contains only initial and final outputs
    ver2:
    ver3: Adds main branch infomation

    Notes: host_disk_mass arrays not fixed in ver2 and ver3. 
           The disk masses of hosts are recorded in MainBranches instead.

    """

    dir_path = '/data/chungwen/sgarrak/runs/1000_mixed_logmass/'
    data = dict()
    main = dict()
    if ver==0:
        data_path = dir_path+'lessoutput/'
        
        output_dataset_names = ['fd01_disk','fd002_disk','interp_disk','no_disk']
        satgen_dataset_names = ['initial_mass','initial_mstar','final_mass',
                                'final_mstar','final_radius','final_status','tree_idx']

        for odn in output_dataset_names:
            data[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_{}_lessoutput.hdf5'.format(odn),
                                  satgen_dataset_names,group='/Progenitors')
    
    elif ver==0:
        data_path = dir_path+'reionoutput/'

        output_dataset_names = ['fd01','no','step_forward','step_backward']#,'interp_sm','interp']
        satgen_dataset_names = ['orbit','has_galaxy','itree','levels_at_tsteps','nprog',
                                'prog_masses','prog_mstars','radii','circularity','status','t_proc','tage',
                                'tree_idx','tsteps']

        for odn in output_dataset_names:
            print('Building data: ',odn)
            data[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_{}_disk.hdf5'.format(odn),
                                  satgen_dataset_names,group='/Progenitors')

    elif ver==1:
        data_path = dir_path+'reionoutput/evolve/'

        output_dataset_names = ['fd01','no','step_forward','step_forward_shift15','step_forward_shift05']#,'interp_sm','interp']
        satgen_dataset_names = ['coors','has_galaxy','itree','levels_at_tsteps','nprog',
                                'prog_masses','prog_mstars','radii','status','t_proc','tage',
                                'tree_idx','tsteps']

        for odn in output_dataset_names:
            print('Building data: ',odn)
            data[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_{}_disk.hdf5'.format(odn),
                                  satgen_dataset_names,group='/Progenitors')

    elif ver==2:
        data_path = dir_path+'reionoutput/evolve_ver2/'

        output_dataset_names = ['no','fd01','step_backward','step_forward','step_forward_threshold_off','step_forward_threshold_off_z0smhm_on']
        prog_dataset_names = ['coors','has_galaxy','itree','levels_at_tsteps','nprog',
                                'prog_masses','prog_mstars','radii','status','t_proc','tage',
                                'tree_idx','tsteps']
        main_dataset_names = ['main_branch_disk_mass']
        for odn in output_dataset_names:
            print('Building data: ',odn)
            data[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_{}_disk.hdf5'.format(odn),
                                  prog_dataset_names,group='/Progenitors')
            if odn != 'no':
                main[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_{}_disk.hdf5'.format(odn),
                                      main_dataset_names,group='/MainBranches')

    # This version uses smooth concentration
    elif ver==3:
        data_path = dir_path+'reionoutput/evolve_ver3/smooth_concentration/'

        output_dataset_names = ['emerge']
        prog_dataset_names = ['coors','has_galaxy','itree','levels_at_tsteps','nprog',
                                'prog_masses','prog_mstars','radii','status','t_proc','tage',
                                'tree_idx','tsteps']
        main_dataset_names = ['main_branch_halo_mass','main_branch_halo_c','main_branch_disk_mass','main_branch_disk_reff']

        for odn in output_dataset_names:
            data[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_EMERGE_disk.hdf5',prog_dataset_names,group='/Progenitors')
            if odn != 'no':
                main[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_EMERGE_disk.hdf5',main_dataset_names,group='/MainBranches')

    # This version uses zhao concentration
    elif ver==301:

        data_path = dir_path+'reionoutput/evolve_ver3/zhao_concentration/'
        output_dataset_names = ['emerge']
        prog_dataset_names = ['coors','has_galaxy','itree','levels_at_tsteps','nprog',
                                'prog_masses','prog_mstars','radii','status','t_proc','tage',
                                'tree_idx','tsteps']
        main_dataset_names = ['main_branch_halo_mass','main_branch_halo_c','main_branch_disk_mass','main_branch_disk_reff']

        for odn in output_dataset_names:
            data[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_EMERGE_disk.hdf5',prog_dataset_names,group='/Progenitors')
            if odn != 'no':
                main[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_EMERGE_disk.hdf5',main_dataset_names,group='/MainBranches')

    elif ver==-1:
        data_path = dir_path

        prog_dataset_names = ['levels_at_tsteps','host_disk_masses']
        main_dataset_names = ['main_branch_disk_mass']

        data = read_hdf5(data_path+'test.hdf5',prog_dataset_names,group='/Progenitors')
        main = read_hdf5(data_path+'test.hdf5',main_dataset_names,group='/MainBranches')

    elif ver==-2:
        data_path = dir_path+'debug_lower_mass_range/'

        output_dataset_names = ['emerge']
        prog_dataset_names = ['coors','has_galaxy','itree','levels_at_tsteps','nprog',
                                'prog_masses','prog_mstars','radii','status','t_proc','tage',
                                'tree_idx','tsteps']
        main_dataset_names = ['main_branch_halo_mass','main_branch_halo_c','main_branch_disk_mass','main_branch_disk_reff']

        for odn in output_dataset_names:
            data[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_EMERGE_disk.hdf5',prog_dataset_names,group='/Progenitors')
            if odn != 'no':
                main[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_EMERGE_disk.hdf5',main_dataset_names,group='/MainBranches')
        
    if ver in (2,3,301,-1,-2):
        return data,main
    return data   
#########################################################
### Physics
def moster_18_eff_func(m,M1=1,epsilon=0,beta=1,gamma=1):
    """
    Equation 5
    """
    return 2*epsilon/(((m/M1)**(-beta)) + ((m/M1)**gamma))

def moster_18_eff_sigma_func(m,Msigma=1,sigma0=1,alpha=1):
    """
    Equation 25
    """
    return sigma0 + np.log10(((m/Msigma)**(-alpha))+1)

def halo_mah_to_zhao_c_nfw(tree_main_branch_masses, tree_t_age_gyr):
    """
    """
    h_c_nfw = list()
    
    nlev = tree_main_branch_masses.shape[0]
    for i in range(0,nlev):
        h_c_nfw.append(init.c2_fromMAH(tree_main_branch_masses[i:],tree_t_age_gyr[i:]))
    
    return np.array(h_c_nfw)

def compute_mass(prog,root_mass,task='m_acc',ntrees=1000,reionization=False):
    """
    Compute different mass accretion type
    
    Parameters: prog: progenitor_wdisk: progenitor dict with a disk potential
                      progenitor_wodisk: without a disk potential
                task: m_acc: total destroyed satellite mstar
                      m_stream: stripped mstar, removed the intact part
                      m_total: total satellite mstar
                      
    """
    final_status = prog['status'][:,-1]
    initial_mass = prog['prog_masses'][:,0]
    initial_mstar = prog['prog_mstars'][:,0]
    final_mstar = prog['prog_mstars'][:,-1]
    final_mass = prog['prog_masses'][:,-1]

    survives = final_status == 0
    tree_idx = []
    for itree,nprog in zip(prog['itree'],prog['nprog']):
        tree_idx.append(np.repeat(itree,nprog))
    tree_idx = np.concatenate(tree_idx)
    # Current has_galaxy is only a tag, progenitors still form galaxies depending on smhm.
    has_galaxy = True #prog['has_galaxy'] == 1

    mass_arr = np.zeros(ntrees)

    for itree in range(0,ntrees):
        this_tree    = tree_idx == itree
        imax = np.argmax(initial_mass[this_tree])
        
        if task=='m_acc':
            if reionization:
                mass_arr[itree] = np.sum(initial_mstar[this_tree & (~survives) & has_galaxy],dtype=np.float64)
            else:
                mass_arr[itree] = np.sum(initial_mstar[this_tree & (~survives)],dtype=np.float64)

        elif task=='m_stream':
            if reionization:
                mass_arr[itree] = np.sum(initial_mstar[this_tree & (survives) & has_galaxy] - final_mstar[this_tree & (survives) & has_galaxy],dtype=np.float64)
            else:
                mass_arr[itree] = np.sum(initial_mstar[this_tree & (survives)] - final_mstar[this_tree & (survives)],dtype=np.float64)
        
        elif task=='m_total':
            if reionization:
                mass_arr[itree] = np.sum(initial_mstar[this_tree & has_galaxy],dtype=np.float64)       
            else:
                mass_arr[itree] = np.sum(initial_mstar[this_tree],dtype=np.float64)

        elif task=='hm_smooth':
            mass_arr[itree] = root_mass[itree]-np.sum(initial_mass[this_tree])

        elif task=='hm_acc':
            mass_arr[itree] = np.sum(initial_mass[this_tree & (survives)] - final_mass[this_tree & (survives)],dtype=np.float64)
            
        elif task=='max_prog':
            mass_arr[itree] = initial_mass[this_tree][imax]
            
        elif task=='hm_total':
            mass_arr[itree] = np.sum(initial_mass[this_tree],dtype=np.float64)
        
    return mass_arr

def volume_weight(root_mass):
    # Define mass bins (working in h^-1 Msol)
    logM = np.log10(root_mass*hubble_parameter)
    mass_bins = np.arange(10, 14.1, 0.0001)  # in log10(M/Msun/h)
    bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
    bin_indices = np.digitize(logM, mass_bins) - 1
    
    # Compute halo mass function using hmf
    # Initialize hmf at z=0
    cosmology = cosmo.FlatLambdaCDM(hubble_parameter*100,0.25,Ob0=OMEGA_B,Tcmb0=2.73)
    mf = MassFunction(Mmin=11.0, Mmax=13.0, dlog10m=0.00001, z=0, cosmo_model=cosmology)

    # Interpolate dn/dlog10M at bin centers
    mass_function_vals = mf.dndlog10m  # units: (h/Mpc)^3 per dex

    mass_vals  = np.log10(mf.m) # log10 masses from hmf
    phi_interp = np.interp(bin_centers, mass_vals, mass_function_vals)

    # Count trees per bin
    N_trees_per_bin = np.bincount(bin_indices, minlength=len(mass_bins) - 1)

    # Avoid division by zero
    valid_bins = N_trees_per_bin > 0
    weights_per_bin = np.zeros_like(bin_centers)
    weights_per_bin[valid_bins] = phi_interp[valid_bins] / N_trees_per_bin[valid_bins]

    # Assign a weight to each halo/tree
    weights = weights_per_bin[bin_indices]
    
    return weights


def avg_smhm(n,return_edge=False):
    """
    Averaged stellar masses given redshifts and halo masses.
    By drawing N number of random samples with different z
    for each halo mass.
    
    The halo mass range was chosen by min-cfg.mres max-max tree mass
    
    Note init.Mstar reads linear halo mass
    
    """
    zrange  = np.arange(0,14,0.2)
    Nz = np.random.choice(zrange,n)
    h = 0.73

    hmrange = 10**np.arange(9,12.7,0.2)/h

    avgsm = []
    maxsm = []
    minsm = []

    for hm in hmrange:
        smlist = [sga.mod_Mstar(hm,z=z, choice='B13',task='Mstar') for z in Nz]
        avgsm.append(np.median(smlist))
        maxsm.append(max(smlist))
        minsm.append(min(smlist))

    f = interp1d(hmrange,avgsm)

    if return_edge:
        return np.array(maxsm),np.array(minsm),hmrange

    return f


def first_pericenter_distance(r):
    """
    Compute the first pericenter distance for a particle spiraling into the origin.
    
    Parameters
    ----------
    r : ndarray, shape (N, D)
        Array of particle radii over time (1 dimensions).
    
    Returns
    -------
    float
        The first pericenter distance.
    int
        The index where the first pericenter occurs.
    """

    # The particle starts at large r, falls in, reaches pericenter (min r), then goes out again.
    # Find first local minimum in r after the initial decrease.
    dr = np.diff(r)

    # Condition for local min: slope goes negative -> then positive
    pericenters=[]
    for i in range(1, len(dr)):
        if dr[i-1] < 0 and dr[i] > 0:
            pericenters.append(i,r[i])
    if pericenters:
        return pericenters
    # If no pericenter is found, return global min
    return np.min(r), np.argmin(r)

def density_profile(data,method='first'):
    """
    Deposite mass at the first pericenter (or each pericenter) to simulate the density profile.
    If no pericenter, deposite the mass at the local minimum (the last location)
    
    But, does the stellar mass start to deposite at the first pericenter?
    """
    radii = data['radii']
    tree_idx = data['tree_idx']
    

    return

def scale_radius_gao(m,z,h=0.73):
    """
    M in h^-1 M_sun
    """
    aexp = 1/(1+z)
    M= m*h
    lgM = np.log10(M)
    la  = np.log10(aexp)
    
    A = -0.14*np.exp(-((la+0.05)/0.35)**2)
    B = 2.646*np.exp(-((la+0.0)/0.5)**2)
    return 0.1**(A*lgM+B)


## mass distribution for the strip particles
# maybe there will be more method for distributing mass in the future
def mass_distribution(star_coor,dmstar):
    """
    Distribute the stripped mass on the integrated orbit to get
    the density profile
    The amount of mass that deposit on the orbit is proportional
    to the time it spend at the location

    Parameters: dmstar: the array of stripped mass at each step
                star_coor: the coordinates of each stripped mass at each step

    Return: A list of arrays that assigns equal masses at all the positions
    """

    equal_dm = []
    for i in np.arange(0,len(dmstar)):

        # equal mass at each saved coordinates
        dm = dmstar[i]/len(star_coor[i])
        equal_dm.append(np.repeat(dm,len(star_coor[i])))

    return np.concatenate(equal_dm)

def surface_density_profile(star_coor_list,dmstar_list, r_bins, axis='xy'):
    """
    Compute 2-D surface density profile Σ(R)
    
    Parameters
    ----------
    coor : array
        3-D coordinates (kpc)
    m : array
        Particle masses (Msun)
    r_bins : array
        Radial bin edges (kpc)

    Returns
    -------
    R_mid : array
        Midpoint radius of bins (kpc)
    Sigma : array
        Surface density (Msun / kpc^2)
    """
    nprog = len(dmstar_list)
    #print('check nprog: ',nprog)
    mass_in_bin_each = []
    for i in range(nprog):
        star_coor = star_coor_list[i]
        dmstar    = dmstar_list[i]
        sc = np.concatenate(star_coor)
        if axis=='xy':
            r = np.sqrt(sc[:,0]**2+sc[:,1]**2)
        elif axis=='xz':
            r = np.sqrt(sc[:,0]**2+sc[:,2]**2)
        elif axis=='yz':
            r = np.sqrt(sc[:,1]**2+sc[:,2]**2)

        m = mass_distribution(star_coor,dmstar)
        lgr = np.log10(r)
        
        # Total mass in each radial bin
        mass_in_bin, _ = np.histogram(lgr, bins=r_bins, weights=m)

        mass_in_bin_each.append(mass_in_bin)
        
    mass_in_bin_tot = np.sum(mass_in_bin_each,axis=0)
    #print(sum(mass_in_bin_tot))
    # Area of each annulus
    area = np.pi * ((10**r_bins[1:])**2 - (10**r_bins[:-1])**2)
    
    # Surface density
    Sigma = mass_in_bin_tot / area
    Sigma_each = mass_in_bin_each / area
    #print(Sigma,sum(Sigma))
    # Midpoint radius
    R_mid = 0.5 * (r_bins[1:] + r_bins[:-1])

    return R_mid, Sigma, Sigma_each

## Calculate the mass of MN disk within a cylindrical annuli.
def rho_MN(R, z, Md, a, b):
    """
    Miyamoto-Nagai density [Msun/kpc^3].
    R, z, a, b in kpc; Md in Msun.
    """
    x = np.sqrt(z**2 + b**2)

    numerator = (
        a * R**2
        + (a + 3*x) * (a + x)**2
    )

    denominator = (
        x**3
        * (R**2 + (a + x)**2)**2.5
    )

    return (b**2 * Md / (4*np.pi)) * numerator / denominator

def Sigma_faceon(R, Md, a, b):
    """
    Face-on projected surface density Sigma(R) [Msun/kpc^2].
    2 * (integration 0 -> inf)
    """
    # density is symmetric in z
    val, err = quad(
        lambda z: rho_MN(R, z, Md, a, b),
        0,
        np.inf,
        epsabs=0, # ignore absolute error target
        epsrel=1e-6, # integrate until |error| <= epsrel * |val|
        limit=200 # maximum number of subinterval
    )

    return 2 * val


def M_annulus_faceon(R1, R2, Md, a, b):
    """
    Projected face-on mass inside cylindrical annulus R1 < R < R2.
    """
    val, err = quad(
        lambda R: 2*np.pi * R * Sigma_faceon(R, Md, a, b),
        R1,
        R2,
        epsabs=0,
        epsrel=1e-5,
        limit=100
    )

    return val

def disk_scale_radius_height(reff):
    """
    """
    flattening = 25.
    scale_radius = 0.766421/(1.+1./flattening) * reff
    scale_height = scale_radius / flattening
    return scale_radius,scale_height

def sigma_exponential_mass_normalized(R, Md, Rd):
    Sigma0 = Md / (2 * np.pi * Rd**2)
    return Sigma0 * np.exp(-R / Rd)
#########################################################
### Data arrangement
def small_tickmarks(ax,minor_x,minor_y):
    import matplotlib.ticker as ticker
    
    ax.tick_params(labelsize=10)
    minorLocator   = ticker.MultipleLocator(minor_x)
    minorFormatter = ticker.FormatStrFormatter('')
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_minor_formatter(minorFormatter)

    minorLocator   = ticker.MultipleLocator(minor_y)
    minorFormatter = ticker.FormatStrFormatter('')
    ax.yaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_formatter(minorFormatter)
    return

# From https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def valid_coors(coors):
    """
    This function filters [-1,-1,-1] in the coordinate array.

    [-1,-1,-1] coordinates were fiiled in when a progenitor was 
    below the cfg resolution to keep the arrays having the same shape.
    """

    mask = ~(np.all(coors == -1, axis=1))
    coors_valid = coors[mask]

    return coors_valid

def dict_comprehensive(d,mask,inhomogenious=True):
    """
    """
    # pick an array shape to reshape the evolved history arrays
    evo_arr_len = d['radii'][:,-1]

    d_filtered = {}
    if inhomogenious:
        for k,v in d.items():
            if v.shape[0] == evo_arr_len.shape[0]:
                # apply the mask
                d_filtered[k] = v[mask]
            else:
                # keep untouched
                d_filtered[k] = v
    else:
        d_filtered = {k:v[mask] for k,v in d.items()}

    return d_filtered

## regroup the results and streams with indices for the strips
def rebuild_top_indices(top_indices):
    """
    out[itree]
    """

    top_indices = np.asarray(top_indices)
    return

def rebuild_dmstar_coor(dmstar_coor, strip_each_length, strip_start_index, prog_start_index):
    """
    Rebuild nested structure:
        out[itree][iprog][istrip] -> ndarray of shape (Nstrip, 3)

    Parameters
    ----------
    dmstar_coor : ndarray, shape (N, 3)
        Fully concatenated coordinate array.
    strip_each_length : ndarray, shape (Nstrip,)
        Number of rows in each strip.
    strip_start_index : ndarray, shape (Nstrip,)
        Resets to 0 when a new prog starts.
    prog_start_index : ndarray, shape (Nprog,)
        Resets to 0 when a new tree starts.

    Returns
    -------
    out : list[list[list[np.ndarray]]]
    """
    dmstar_coor       = np.asarray(dmstar_coor)
    strip_each_length = np.asarray(strip_each_length)
    strip_start_index = np.asarray(strip_start_index)
    prog_start_index  = np.asarray(prog_start_index)

    assert np.sum(strip_each_length) == len(dmstar_coor)

    # global offsets into dmstar_coor for each strip
    strip_offsets = np.r_[0, np.cumsum(strip_each_length)]

    # strip indices where a new prog starts
    prog_breaks = np.r_[np.where(strip_start_index == 0)[0], len(strip_each_length)]

    # prog indices where a new tree starts
    tree_breaks = np.r_[np.where(prog_start_index == 0)[0], len(prog_start_index)]

    # number of progs inferred from strip data should match prog metadata
    assert len(prog_breaks) - 1 == len(prog_start_index)

    out = []
    for itree in range(len(tree_breaks) - 1):
        p0, p1 = tree_breaks[itree], tree_breaks[itree + 1]

        tree_list = []
        for iprog in range(p0, p1):
            s0, s1 = prog_breaks[iprog], prog_breaks[iprog + 1]

            prog_list = []
            for istrip in range(s0, s1):
                prog_list.append(dmstar_coor[strip_offsets[istrip]:strip_offsets[istrip + 1]])

            tree_list.append(prog_list)

        out.append(tree_list)

    return out

def rebuild_dmstar(dmstar, prog_each_length, prog_start_index):
    """
    Rebuild nested structure new_dmstar[itree][iprog] from concatenated prog-level data.

    Parameters
    ----------
    dmstar : ndarray, shape (N,)
        Concatenated 1-D data.
    prog_each_length : ndarray, shape (Nprog,)
        Length of each prog block in dmstar.
    prog_start_index : ndarray, shape (Nprog,)
        Start index within each tree. Resets to 0 for each new tree.

    Returns
    -------
    new_dmstar : list[list[np.ndarray]]
        new_dmstar[itree][iprog]
    """
    dmstar = np.asarray(dmstar)
    prog_each_length = np.asarray(prog_each_length)
    prog_start_index = np.asarray(prog_start_index)

    # Global offsets of each prog block in dmstar
    prog_offsets = np.concatenate(([0], np.cumsum(prog_each_length)))

    # Prog indices where a new tree starts
    tree_breaks = np.where(prog_start_index == 0)[0]
    tree_breaks = np.append(tree_breaks, len(prog_each_length))

    new_dmstar = []

    for itree in range(len(tree_breaks) - 1):
        p0 = tree_breaks[itree]
        p1 = tree_breaks[itree + 1]

        tree_list = []
        for iprog in range(p0, p1):
            tree_list.append(dmstar[prog_offsets[iprog]:prog_offsets[iprog + 1]])

        new_dmstar.append(tree_list)

    return new_dmstar

def regroup_data(stream):
    """
    Rebuild strip masses and strip orbits data from starting indices and length
    """
    star_coor = stream['star_coor']
    dmstar    = stream['dmstar']
    prog_start_index  = stream['prog_start_index']
    prog_each_length  = stream['prog_each_length']
    strip_each_length = stream['strip_each_length']
    strip_start_index = stream['strip_start_index']

    tree_start_index = np.where(prog_start_index == 0)[0]
    tree_each_length = np.diff(np.append(tree_start_index, len(prog_start_index)))

    new_dmstar = rebuild_dmstar(dmstar, prog_each_length, prog_start_index)
    new_dmstar_coor = rebuild_dmstar_coor(star_coor, strip_each_length, strip_start_index, prog_start_index)

    return new_dmstar,new_dmstar_coor

def regroup_result(prog):
    """
    Rebuild results to have itree layers.
    """
    prog_mhalo = prog['prog_masses']
    prog_mstar = prog['prog_mstars']
    prog_tage  = prog['tage']
    nprog = prog['nprog']

    start_index = 0
    re_prog_mhalo = []
    re_prog_mstar = []
    re_prog_tage  = []
    for n in nprog:
        re_prog_mhalo.append(prog_mhalo[start_index:start_index+n])
        re_prog_mstar.append(prog_mstar[start_index:start_index+n])
        re_prog_tage.append(prog_tage[start_index:start_index+n])
        start_index += n
    return re_prog_mhalo,re_prog_mstar,re_prog_tage

def regroup_result_main(main):
    main_disk = main['main_branch_disk_mass']
    main_reff = main['main_branch_disk_reff']
    remain_disk = []
    remain_reff = []
    for i in range(1000):
        remain_disk.append(main_disk[i:i+132])
        remain_reff.append(main_reff[i:i+132])
    return remain_disk,remain_reff
#########################################################
### Plot functions
def larger_ticks(ax=None):
    if ax is None:
        ax = pl.gca()
    ax.tick_params(which='major', width=1.5, length=5, direction='in', labelsize=20)
    ax.tick_params(which='minor', direction='in')
    ax.minorticks_on()
    return


def plot_b18(ax,h=0.678,centrals=True,mean_only=False,
             scale_to_baryons=1,as_eff=False,edges=False,**kwargs):
    """
    """
    b18smhm_path = "/data/apcooper/coco/obs/behroozi18/umachine-dr1/data/smhm/median_raw/smhm_a1.002312.dat"
    b18smhm_scatter_path = "/data/apcooper/coco/obs/behroozi18/umachine-dr1/data/smhm/median_raw/smhm_scatter_a1.002312.dat"

    b18smhm = Table.read(b18smhm_path,format='ascii.commented_header')
    b18smhm_scatter = Table.read(b18smhm_scatter_path,format='ascii.commented_header')

    b18_h = 0.678
    hubble_factor = np.log10(b18_h/h)
    
    if centrals:
        x,y = b18smhm['HM(0)'], b18smhm['Med_Cen(4)']
        dyp = b18smhm_scatter['Med_Cen(4)'] # b18smhm['Err+(5)']
        dym = b18smhm_scatter['Med_Cen(4)'] # b18smhm['Err-(6)']
    else:
        x,y = b18smhm['HM(0)'], b18smhm['Med_Sat(13)']
        dyp = b18smhm_scatter['Med_Sat(13)'] # b18smhm['Err+(5)']
        dym = b18smhm_scatter['Med_Sat(13)'] # b18smhm['Err-(6)']

    w = (y < 0) & (x > 6) #& (np.abs(dyp) < 1) & (np.abs(dym) < 1)
    
    xx = x[w]+hubble_factor
    ym = y[w]+2*hubble_factor
    y1 = y[w]+2*hubble_factor+2*dym[w]
    y2 = y[w]+2*hubble_factor-2*dym[w]    
    if edges:
        ax.plot(xx,y1, **kwargs)
        ax.plot(xx,y2, **kwargs)

    ym = ym - np.log10(scale_to_baryons)
    y1 = y1 - np.log10(scale_to_baryons)
    y2 = y2 - np.log10(scale_to_baryons)

    # Multiply (add in log) by halo mass
    if as_eff:
        # Not log
        ym = 10**ym
        y1 = 10**y1
        y2 = 10**y2
    else:
        ym += x[w]
        y1 += x[w]
        y2 += x[w]
    
    if mean_only:
        pl.plot(xx, ym, color=kwargs.get('c','g'),lw=2)
    else:
        ax.fill_between(xx,y1,y2,color=kwargs.get('c','g'),
                        zorder=kwargs.get('zorder',-20),
                        alpha=kwargs.get('alpha',0.2),
                        label='Behroozi+ (2019)')    
    return

def plot_avgb13_satgen(ax):
    """
    Repeat b13 init.Mstar relation 10000 times to get the range. 
    """
    maxsm,minsm,hmrange = avg_smhm(10000,return_edge=True)

    ax.fill_between(np.log10(hmrange),np.log10(maxsm),np.log10(minsm),
                    alpha=0.5,label='B13')
    
    return

def plot_b13_satgen(ax,sigma=1,color='grey',z0_smhm=False):
    """
    Similar to the above function but directly uses 0.2 scatter in log sm.

    For each halo mass bins, it takes the highest and lowest value of hm(z) and 
    add/minus 0.2.
    """
    h = 0.7
    zrange = np.arange(0,14,0.2)
    hmrange = 10**np.arange(8,12.7,0.2)/h
    lghmrange = np.log10(hmrange)
    smmax = []
    smmin = []
    for hmr in hmrange:
        if z0_smhm:
            sm = np.log10([sga.mod_Mstar(hmr,z=0,task='lgMs_B13') for zr in zrange])
        else:
            sm = np.log10([sga.mod_Mstar(hmr,z=zr,task='lgMs_B13') for zr in zrange])
        smmax.append(max(sm)+0.2*sigma)
        smmin.append(min(sm)-0.2*sigma)

    ax.fill_between(lghmrange,smmax,smmin,alpha=0.5,facecolor=color,label='B13')
    return


def plot_m18(ax,h=0.6781,edges=False,omega_0=0.25,**kwargs):
    """
    Moster (2018)
    """
    # Table 8, z=0.1
    moster_18_default = partial(moster_18_eff_func,M1=10**11.80,epsilon=0.14,beta=1.75,gamma=0.57)
    moster_18_sigma_default = partial(moster_18_eff_sigma_func,Msigma=10**10.80,sigma0=0.16,alpha=1)

    m18_h = 0.6781
    hubble_factor = np.log10(m18_h/h)
        
    lmhalo = np.arange(6,15,0.1)
    
    OMEGA_B = 0.04455
    baryon_mass_fraction = OMEGA_B/omega_0

    bmf = np.log10(baryon_mass_fraction)
    
    eff_moster = moster_18_default(10**lmhalo)
    log_sig_moster = moster_18_sigma_default(10**lmhalo)

    lmstar = lmhalo + np.log10(eff_moster*baryon_mass_fraction) + 2*hubble_factor
    
    xx = lmhalo + hubble_factor
    y1 = lmhalo + np.log10(eff_moster) + 2*log_sig_moster + bmf + 2*hubble_factor
    y2 = lmhalo + np.log10(eff_moster) - 2*log_sig_moster + bmf + 2*hubble_factor
    
    if edges:
        pl.plot(xx,y1,**kwargs)
        pl.plot(xx,y2,**kwargs)

    ax.fill_between(xx,y1,y2,color=kwargs.get('c','c'),
                zorder=kwargs.get('zorder',-20),
                alpha=kwargs.get('alpha',0.2),label='Moster+ (2018)')
    return

def plot_dw_2020_cmf(ax,label=None):
    """
    """
    Upsilon_V = 1.5  # stellar mass-to-light ratio
    Mv_sun    = 4.83   
    dw  = Table.read('/lfs/data/apcooper/projects/obsdata/dw2020/dw2020_mw_cor_lf.csv',
                    format='ascii')
    # Missing - in table
    M_V = -dw['MV']

    logM_star = np.log10(Upsilon_V) - 0.4 * (M_V - Mv_sun)
    ax.plot(logM_star, np.log10(dw['NMV']), drawstyle='steps-pre',c='k',label=label)
    return

def plot_saga_cmf(ax,label=None):
    """
    """
    # Table C5 from Mao 2024
    saga_sat_mf = np.array([
    (10.31 , 0.026),
    (9.94  , 0.185),
    (9.56  , 0.475),
    (9.19  , 0.475),
    (8.81  , 0.820),
    (8.44  , 1.041),
    (8.06  , 1.524),
    (7.69  , 2.304),
    (7.31  , 3.111),
    (6.94  , 5.950),
    (6.56  , 4.376)])

    saga_lmstar = saga_sat_mf[:,0]
    saga_count  = saga_sat_mf[:,1]

    # SAGA
    # The bin width is either 0.37 or 0.38

    # The bins are in log Mstar
    BIN_WIDTH  = 0.37
    BIN_HWIDTH = BIN_WIDTH/2.0
    BIN_EDGES  = np.concatenate([saga_lmstar - BIN_HWIDTH, [saga_lmstar[-1]+BIN_HWIDTH]])

    # We take the given masses as midpoints of bins
    # Reverse order to get low to high
    saga_lmstar = saga_sat_mf[:,0][::-1]

    # The counts are in dN/dlogM
    # Reverse order to get low to high
    saga_dNdlogM = saga_sat_mf[:,1][::-1]

    saga_N_bin   = saga_dNdlogM*BIN_WIDTH
    saga_N_bin_c = np.cumsum(saga_N_bin[::-1])[::-1]
    ax.plot(saga_lmstar,np.log10(saga_N_bin_c),label=label,c='g',lw=2)
    return

def plot_koposov_cmf(ax):
    """
    Thanks GPT
    """
    # Constants
    Upsilon_V = 1.5  # stellar mass-to-light ratio
    Mv_sun    = 4.83

    # Define Mv range from -13 (bright) to -2 (faint)
    Mv = np.linspace(-13, -2, 1000)

    # Koposov differential LF
    dNdMv = 10 * 10**(0.1 * (Mv + 5))

    # Convert Mv to stellar mass
    logM_star = np.log10(Upsilon_V) - 0.4 * (Mv - Mv_sun)
    M_star = 10**logM_star

    # Sort increasing mass
    sort_idx = np.argsort(M_star)
    M_star = M_star[sort_idx]
    dNdMv = dNdMv[sort_idx]

    # Compute dlogM/dMv to convert to dN/dlogM
    dlogM_dMv = -0.4 / np.log(10)
    dNdlogM = dNdMv / np.abs(dlogM_dMv)

    # Estimate bin widths
    logM = np.log10(M_star)
    dlogM = np.gradient(logM)
    N_bin = dNdlogM * dlogM

    # Cumulative number: number of satellites with M_* > X
    N_cumulative = np.cumsum(N_bin[::-1])[::-1]

    ax.plot(np.log10(M_star), np.log10(N_cumulative), label='Koposov+08 (converted to mass)')
    return

def plot_survive_frac(progs):
    """
    progenitor status = 0, intact
                      = 1, lost
    """
    x = progs.keys()
    total_prog = len(progs[list(x)[0]]['tree_idx'])

    survive_frac = []
    for xx in x:
        number_survivors = len(np.where(progs[xx]['status'][:,-1]==0)[0])
        survive_frac.append(number_survivors/total_prog)
    print('survive_frac: ',survive_frac)
    x_pos = np.arange(len(x))
    y = survive_frac
    pl.figure(figsize=(6,4))
   
    pl.bar(x_pos,y,color='cyan',alpha=0.5,width=0.5)
    pl.bar(x_pos,[1-yi for yi in y], bottom=y, color='red', alpha=0.5, width=0.5)

    pl.xticks(x_pos,x)
    pl.ylim(0,1)

def plot_subhalo_massfunction(progs,root_mass,radius='total',normalized=False):
    """
    Accumulated subhalo mass function

    Parameters: progs: All the progentiors from satgen
                root_mass: The root halo mass of ntrees
                radius: inner (<50kpc) or total
    """
    models = list(progs.keys())

    pl.figure(figsize=(8,6))
    for key,p in progs.items():
        if radius=='inner':
            final_radius = p['radii'][:,-1]
            inner_region = final_radius <50
            #p_inner = dict_comprehensive(p,inner_region,inhomogenious=True)
            p_inner = {}
            for k, v in p.items():
                if v.shape[0] == final_radius.shape[0]:
                    p_inner[k] = v[inner_region]
                else:
                    p_inner[k] = v
            hm_acc = compute_mass(p_inner,root_mass,task='hm_acc')
            pl.title('$\mathrm{<50kpc}$',fontsize=20)
        else:
            hm_acc = compute_mass(p,root_mass,task='hm_acc')
            pl.title('total',fontsize=20)
        if normalized:
            bins = np.arange(-4,0,0.2)
            h,_ = np.histogram(np.log10(hm_acc/root_mass),bins)
        else:
            bins = np.arange(10,12,0.1)
            h,_ = np.histogram(np.log10(hm_acc),bins)
        reverse_h = h[::-1]
        accumulate_h = np.cumsum(reverse_h)
        
        pl.plot(bins[:-1],np.log10(accumulate_h[::-1]),label=key)

    pl.legend(frameon=False,prop={'size':15})
    if normalized:
        pl.xlabel('$\mathrm{\log_{10}\,M_{sat,halo}/M_{host,halo}}$',fontsize=20)
        pl.ylabel('$\mathrm{\log_{10}\, N(>M_{sat,halo}/M_{host,halo})}$',fontsize=20)
    else:
        pl.xlabel('$\mathrm{\log_{10}\,M_{sat,halo}}$',fontsize=20)
        pl.ylabel('$\mathrm{\log_{10}\, N(>M_{sat,halo})}$',fontsize=20)
    larger_ticks()
    return

def plot_macc_function(progs,root_mass,reionization=False):
    """
    Volume weighted accreted stellar mass function
    """
    weights = volume_weight(root_mass)
    bins = np.arange(5,13,0.2)
    cs = ['r','k','b','cyan','purple']

    pl.figure(figsize=(7,7))
    for (k,p),c in zip(progs.items(),cs):
        m_acc = compute_mass(p,root_mass,task='m_acc',reionization=reionization)
        h,_ = np.histogram(np.log10(m_acc),bins,weights=weights,density=True)
        pl.plot(bins[:-1],np.log10(h),label=k,c=c,drawstyle='steps-post')
    
    pl.legend(frameon=False,prop={'size':15})
    pl.xlabel('$\log_{10}\,M_\mathrm{\star,acc}/M_{\odot}$',fontsize=20)
    pl.ylabel('$\log_{10}\,dN/dM_\mathrm{\star,acc}$',fontsize=20)
    larger_ticks()

    return

def plot_macc_mstream_function(progs,root_mass,task='m_acc'):
    """
    Volume weighted stellar stream mass function

    s is selected for a tree that has all progenitors not 
    having higher than 0.2 of the host mass

    Parameters: task: m_acc
                      m_stream
                      m_acc_stream
    """

    cs = ['r','k','b','cyan','purple']
    keys = list(progs.keys())

    # This value is the same for all models
    tree_max_prog_mass = compute_mass(progs[keys[0]],root_mass,task='max_prog')
    tree_max_prog_mass_frac = tree_max_prog_mass/root_mass
    s = tree_max_prog_mass_frac < 0.2    

    weights = volume_weight(root_mass)
    bins=np.arange(5,13,0.2)

    pl.figure(figsize=(12,5))
    pl.subplot(121)
    for (k,p),c in zip(progs.items(),cs):
        m_acc = compute_mass(p,root_mass,task='m_acc')
        m_stream = compute_mass(p,root_mass,task='m_stream')

        ha,_ = np.histogram(np.log10(m_acc),bins,weights=weights,density=True)
        hs,_ = np.histogram(np.log10(m_stream),bins,weights=weights,density=True)

        if task=='m_acc':
            pl.plot(bins[:-1],np.log10(ha),drawstyle='steps-post',c=c)
        if task=='m_stream':
            pl.plot(bins[:-1],np.log10(hs),drawstyle='steps-post',c=c,ls='dashed')
        if task=='m_acc_stream':
            pl.plot(bins[:-1],np.log10(ha),drawstyle='steps-post',c=c)
            pl.plot(bins[:-1],np.log10(hs),drawstyle='steps-post',c=c,ls='dashed')

    l1 = pl.Line2D([0,0],[0,0],ls='solid',c='r')
    l2 = pl.Line2D([0,0],[0,0],ls='dashed',c='r')
    handles = [l1,l2]
    labels = ['m_acc','m_stream']
    legend2 = pl.legend(handles=handles,labels=labels,frameon=False,prop={'size':15},loc='lower left')
    pl.xlim(7,11.5)
    pl.xlabel('$\log_{10}\,M_\mathrm{\star,acc}/M_{\odot}$',fontsize=20)
    pl.ylabel('$\log_{10}\,dN/dM_\mathrm{\star,acc}$',fontsize=20)
    larger_ticks()

    # Selected by the less massive ones (s<0.2)
    pl.subplot(122)

    for (k,p),c in zip(progs.items(),cs):
        m_acc = compute_mass(p,root_mass,task='m_acc')
        m_stream = compute_mass(p,root_mass,task='m_stream')

        ha,_ = np.histogram(np.log10(m_acc[s]),bins,weights=weights[s],density=True)
        hs,_ = np.histogram(np.log10(m_stream[s]),bins,weights=weights[s],density=True)

        if task=='m_acc':
            pl.plot(bins[:-1],np.log10(ha),label=k,drawstyle='steps-post',c=c)
        if task=='m_stream':
            pl.plot(bins[:-1],np.log10(hs),label=k,drawstyle='steps-post',c=c,ls='dashed')
        if task=='m_acc_stream':
            pl.plot(bins[:-1],np.log10(ha),label=k,drawstyle='steps-post',c=c)
            pl.plot(bins[:-1],np.log10(hs),drawstyle='steps-post',c=c,ls='dashed')

    legend1 = pl.legend(frameon=False,prop={'size':15},loc='upper right')
    pl.gca().add_artist(legend1)

    pl.xlim(7,11.5)
    pl.title('$\mathrm{s<0.2}$',fontsize=20)
    pl.xlabel('$\log_{10}\,M_\mathrm{\star,acc}/M_{\odot}$',fontsize=20)
    #pl.ylabel('$\log_{10}\,dN/dM_\mathrm{\star,acc}$',fontsize=20)
    larger_ticks()
    
    return

def plot_fianl_mhalo_compare_coded_infall_time(tree_prog,progs,models):
    """
    final progenitor halo mass for two models color coded with infall time.
    """

    zinfall = tree_prog['ProgenitorZred']
    prog1,prog2 = progs[0],progs[1]
    x = prog1['prog_masses'][:,-1]
    y = prog2['prog_masses'][:,-1]
    
    a = []
    for i in np.arange(0,1000):
        this_tree_1 = np.where(prog1['tree_idx']==i)[0]
        this_tree_2 = np.where(prog2['tree_idx']==i)[0]
        for i1,i2 in zip(this_tree_1,this_tree_2):
            a.append([x[i1],y[i2]])
    a = np.array(a)

    pl.figure(figsize=(10,8))
    pl.scatter(np.log10(a[:,0]),np.log10(a[:,1]),s=1,c=zinfall)
    pl.xlabel(models[0]+r'$_ \mathrm{disk\,\log_{10} M_{halo}} \, (M_\odot)$ ', fontsize=20)
    pl.ylabel(models[1]+r'$_ \mathrm{disk\,\log_{10} M_{halo}} \, (M_\odot)$ ', fontsize=20)
    cbar = pl.colorbar()
    cbar.set_label("infall z", fontsize=20)

def plot_orbits(prog_no,prog_disk,models,task='survive_with_disk',candidate=None):
    """
    Prameters: prog_no: the no disk model
               prog_disk: one of the disk model

    status = 0, intact
           = 1, lost
    """   

    # Find the indices of progenitors still intact with a disk but lost without a disk
    final_status_no = prog_no['status'][:,-1]
    final_status_disk = prog_disk['status'][:,-1]

    inolist = []
    idisklist = []
    survive_longer_with_disk = 0
    same = 0
    survive_shorter_with_disk = 0
    for i in np.arange(0,1000):
        this_tree_no   = np.where(prog_no['tree_idx']==i)[0]
        this_tree_disk = np.where(prog_disk['tree_idx']==i)[0]

        for ino,idisk in zip(this_tree_no,this_tree_disk):
            status_no = final_status_no[ino]
            status_disk = final_status_disk[idisk]
            if (status_no==1) & (status_disk==0):
                if task=='survive_with_disk':
                    inolist.append(ino)
                    idisklist.append(idisk)
                survive_longer_with_disk+=1
            elif status_no==status_disk:
                if task=='same':
                    inolist.append(ino)
                    idisklist.append(idisk)
                same+=1
            else:
                if task=='survive_without_disk':
                    inolist.append(ino)
                    idisklist.append(idisk)
                survive_shorter_with_disk+=1
    
    print("Progenitors survive longer with {} disk: ".format(models[1]),survive_longer_with_disk)
    print("Progenitors end the same: ",same)
    print("Progenitors survive shorter with {} disk: ".format(models[1]),survive_shorter_with_disk)

    if candidate is None:
        # Choose a random example
        pick = np.random.choice(np.arange(0,len(inolist)))
        ino_pick = inolist[pick]
        idisk_pick = idisklist[pick]
        print("viewing {},{}th progenitor of wo/w disk...".format(ino_pick,idisk_pick))
    else:
        ino_pick=candidate[0]
        idisk_pick=candidate[1]

    picks = [ino_pick,idisk_pick]
    progs = [prog_no,prog_disk]
    colors = ['k','r']
    prog_masses = np.repeat(prog_no['prog_masses'][ino_pick,0],2) # Infall halo masses are the same 

    pl.figure(figsize=(20,13))
    ax1 = pl.subplot(2,3,1)
    ax2 = pl.subplot(2,3,2)
    ax3 = pl.subplot(2,3,3)
    ax4 = pl.subplot(2,3,4)
    ax5 = pl.subplot(2,3,5)
    
    for pk,prog,color,l,pm in zip(picks,progs,colors,models,prog_masses):
        coors = prog['orbit'][pk]
        coors = valid_coors(coors)
        X,Y,Z = coors[:,0],coors[:,1],coors[:,2]
        
        ax = pl.sca(ax1)
        pl.plot(X,Y,c=color,ls='solid',label=l)
        pl.legend(frameon=False,prop={'size':20})
        
        ax = pl.sca(ax2)
        pl.plot(X,Z,c=color,ls='solid')

        ax = pl.sca(ax3)
        pl.plot(Y,Z,c=color,ls='solid')

        ax = pl.sca(ax4)
        pl.plot(prog['tsteps'][pk],prog['radii'][pk],c=color,ls='solid')

        ax = pl.sca(ax5)
        a1, = pl.plot(prog['tsteps'][pk],np.log10(prog['prog_masses'][pk]),c=color,ls='solid')
        a2, = pl.plot(prog['tsteps'][pk],np.log10(prog['prog_mstars'][pk]),c=color,ls='dashed')
        
        halo_mass_resolution_limit_rel = cfg.phi_res*pm
        pl.axhline(np.log10(halo_mass_resolution_limit_rel),c='k',alpha=0.1,lw=3,ls='--')
    
    legend1 = pl.legend([a1,a2],['Total','Stars'],frameon=False,prop={'size':20})    
    pl.sca(ax5)
    
    pl.gca().add_artist(legend1)
    # This "resolution limit" phi_res is built in to the satgen orbit integrator;
    # It can be adjusted by setting cfg.phi_res before the evolution calculation.
    # The orbit is not updated after the total mass reaches this limit.
    

    # There is another "resolution limit"  built in to the satgen orbit integrator,
    # cfg.Mres. In this case the limit is in absolute rather than relative mass.
    # Again, the orbit is not updated after the total mass reaches this limit. 
    # By default, cfg.Mres is None.
    if cfg.Mres is not None:
        halo_mass_resolution_limit_abs = cfg.Mres
        pl.axhline(np.log10(halo_mass_resolution_limit_abs),c='g',alpha=0.1,lw=3,ls='--')

    # Note that strange things happen if both limits are set and 
    # cfg.Mres < cfg.phi_res * (initial halo mass)
    
    
    for ax in [ax1,ax2,ax3]:
        pl.sca(ax)
        pl.scatter([0],[0],c='k',marker='x')
        pl.grid()
        pl.xlim(-200,200)
        pl.ylim(-200,200)


    return

def plot_history(mains,models,halo_mass=False):
    """
    Check the mainbranch histories that have step jumps for the step_forward method

    Parameters: halo_mass: If true, plot the sm to z and it's hm to z.
                           If false, plot the sm to z for the two different model.
    """

    z,_,_,hm = read_tree()
    ztrees = np.tile(z,(1000,1)) 
    colorsl = ['b','yellow']   
    colorss = ['cyan','orange']

    b13sm = np.array([gh.lgMs_B13(np.log10(hm_tree),z) for hm_tree in hm])
    b13sm50 = np.percentile(b13sm,50,axis=0)
    b13sm1sigma = np.percentile(b13sm,84.1,axis=0)
    b13sm_err = b13sm1sigma-b13sm50

    fig = pl.figure(figsize=(8,6))

    if halo_mass:
        main = mains[0]
        dm = main['main_branch_disk_mass']
        lgdm50 = np.percentile(np.log10(dm),50,axis=0)
        pl.plot(np.log10(1+ztrees).T,np.log10(dm).T,alpha=0.05,c=colorsl[0],zorder=1)
        pl.scatter(np.log10(1+ztrees)[0].T,lgdm50.T,c=colorss[0],zorder=2)

        lghm50 = np.percentile(np.log10(hm),50,axis=0)
        pl.plot(np.log10(1+ztrees).T,np.log10(hm).T,alpha=0.05,c=colorsl[1],zorder=1)
        pl.scatter(np.log10(1+ztrees)[0].T,lghm50.T,c=colorss[1],zorder=2)
    else:
        for main,model,colorl,colors in zip(mains,models,colorsl,colorss):
            dm = main['main_branch_disk_mass']
            lgdm50 = np.percentile(np.log10(dm),50,axis=0)
            pl.plot(np.log10(1+ztrees).T,np.log10(dm).T,alpha=0.05,c=colorl,zorder=1)
            pl.scatter(np.log10(1+ztrees)[0].T,lgdm50.T,c=colors,zorder=2)
            #print(np.log10(mid_dm))

    # Satgen built-in sm scatter is 0.2 in log
    pl.errorbar(np.log10(1+z),b13sm50,yerr=b13sm_err,c='k',alpha=0.5)
    for zz in z:
        pl.axvline(np.log10(1+zz),c='grey',alpha=0.2,lw=1,ls='--')
    return

def plot_history_smhm(mains,models):
    """
    """
    _,_,_,hm = read_tree()
    colorsl = ['b','yellow']
    colorss = ['cyan','orange']

    pl.figure(figsize=(8,6))

    for main,model,colorl,colors in zip(mains,models,colorsl,colorss):
        dm = main['main_branch_disk_mass']
        pl.plot(np.log10(hm).T,np.log10(dm).T,alpha=0.05,c=colorl,zorder=2,label=model)

    ax = pl.gca()
    plot_b13_satgen(ax)

    pl.xlabel('$\mathrm{\log_{10}\, halo \,mass\,(M_{\odot})}$',fontsize=15)
    pl.ylabel('$\mathrm{\log_{10}\, stellar(disk) \,mass\,(M_{\odot})}$',fontsize=15)
    pl.xlim(9,12.5)
    pl.ylim(5,11.5)
    pl.legend(frameon=False)
    return

def plot_coco(ax,style='individual',bin_method='stellar'):
    is_central = galaxies.is_dhalo_central

    MW_LOG_MSTAR_LO = np.log10(5.7e10 - 1.1e10)
    MW_LOG_MSTAR_HI = np.log10(5.7e10 + 1.5e10)

    is_mw_by_mstar = (np.log10(galaxies.mstar) >= MW_LOG_MSTAR_LO) & \
                    (np.log10(galaxies.mstar) <  MW_LOG_MSTAR_HI) & \
                    galaxies.is_dhalo_central

    galaxies_mw = galaxies.subset(is_mw_by_mstar)

    # For a single mass bin
    lmbins = np.array([11.8,12.2])

    if style=='individual':
        median_profiles_all      = coco.density.AverageDensityProfiles(density_profiles,galaxies,lmbins,minimum_mstar=1e5)
        median_profiles_mw_mstar = coco.density.AverageDensityProfiles(density_profiles,galaxies_mw,lmbins,minimum_mstar=1e5)

        lmbins_wide = np.array([0,15])
        median_profiles_mw_wide = coco.density.AverageDensityProfiles(density_profiles,galaxies_mw,lmbins_wide,minimum_mstar=1e5)

        for ibin in range(median_profiles_all.n_mass_bins):
            xmed,ymed = median_profiles_all.all.median[ibin]
            ax.plot(xmed, ymed, c='k',label='Halo mass only');

            xmed,ymed = median_profiles_mw_mstar.all.median[ibin]
            ax.plot(xmed, ymed, c='k',ls='--',label='Halo mass & MW mstar');

            xmed,ymed = median_profiles_mw_wide.all.median[ibin]
            ax.plot(xmed, ymed, c='k',ls=':',label='MW mstar only',lw=3);

            for idx in np.flatnonzero(is_mw_by_mstar):
                x = np.log10(density_profiles.dp_bins[idx][:-1]) + 3.0
                y = np.log10(density_profiles.all_all_2d[idx,0]) - 6.0
                ax.plot(x, y, c='grey', lw=0.5,alpha=1,zorder=20)

    elif style=='median':
        if bin_method=='stellar':
            lmbins = np.array([8.5,10.6])
            line_style = 'solid'
        else:
            lmbins = np.array([11.4,12.4])
            line_style = 'dotted'
        median_profiles = coco.density.AverageDensityProfiles(density_profiles,galaxies,lmbins,bin_by_mass=bin_method)

        for ibin in range(median_profiles.n_mass_bins):
            xmed,ymed = median_profiles.acc.median[ibin]
            xup,yup = median_profiles.acc.up[ibin]
            xlp,ylp = median_profiles.acc.lp[ibin]
            ax.plot(xmed, ymed, c='cyan', label='COCO accreted',lw=2,ls=line_style)
            ax.fill_between(xlp,ylp,yup,lw=0,color='cyan',alpha=0.2)

            xmed,ymed = median_profiles.ins.median[ibin]
            xup,yup = median_profiles.ins.up[ibin]
            xlp,ylp = median_profiles.ins.lp[ibin]
            ax.plot(xmed, ymed, c='red', label='COCO in-situ',lw=2,ls=line_style)
            ax.fill_between(xlp,ylp,yup,lw=0,color='red',alpha=0.2)
    else:
        raise Exception

    return

def plot_surface_density_profile_with_coco(star_coor_list,dmstar_list,itree_stream,
                                          disk_mass,disk_reff,
                                          r_bins=None,include_disk=False,disk='MN',
                                          task='plot_accreted',hmrange='MW'):
    """
    Remove later: include_disk is not used
    """
    nprofile = len(star_coor_list)

    if r_bins == None:
        # bin the radius within 1000 kpc
        r_bins = np.linspace(np.log10(0.1), np.log10(1000),30)
    else:
        r_bins = r_bins
    r_mid = 0.5 * (r_bins[1:] + r_bins[:-1])
    R_mid = 10**r_mid

    planes=['xy','xz','yz']
    sigma_total = []
    sigma_disk_only = []

    for itree in range(nprofile):

        if include_disk:
            tree_idx = np.where(itree_stream == itree)[0]
            z0_disk_mass_this_tree = disk_mass[tree_idx][0] # This [0] is weird coding, maybe I changed the data shape before.
            z0_disk_reff_this_tree = disk_reff[tree_idx][0]

            z0_disk_scale_radius, z0_disk_scale_height = disk_scale_radius_height(
                z0_disk_reff_this_tree
            )
            if disk=='MN':
                sigma_disk_total = np.array([
                    Sigma_faceon(r, z0_disk_mass_this_tree,
                                z0_disk_scale_radius,
                                z0_disk_scale_height)
                                for r in R_mid])
            elif disk=='exponential':
                sigma_disk_total = np.array([
                    sigma_exponential_mass_normalized(r, z0_disk_mass_this_tree,
                                                     z0_disk_scale_radius)
                                            for r in R_mid])
            sigma_disk_only.append(sigma_disk_total)

        sigma_proj_total = []

        for i in range(3):
            _, sigma, _ = surface_density_profile(
                star_coor_list[itree],
                dmstar_list[itree],
                r_bins,
                axis=planes[i]
            )
            sigma_proj_total.append(np.asarray(sigma))

        # This should not return nan
        assert(not np.any(np.isnan(sigma_proj_total)))

        sigma_proj_mean = np.mean(sigma_proj_total, axis=0) #just mean
        sigma_total.append(sigma_proj_mean)

    sigma_total = np.asarray(sigma_total)
    sigma_disk_only = np.asarray(sigma_disk_only)

    #sigma_total[sigma_total <= 0] = np.nan # not needed

    median = np.nanpercentile(sigma_total, 50, axis=0)
    y1     = np.nanpercentile(sigma_total, 80, axis=0)
    y2     = np.nanpercentile(sigma_total, 20, axis=0)

    median_log = np.where(median > 0, np.log10(median), np.nan)
    y1_log     = np.where(y1 > 0, np.log10(y1), np.nan)
    y2_log     = np.where(y2 > 0, np.log10(y2), np.nan)

    median_disk = np.nanpercentile(sigma_disk_only, 50, axis=0)
    y1_disk     = np.nanpercentile(sigma_disk_only, 100, axis=0)
    y2_disk     = np.nanpercentile(sigma_disk_only, 0, axis=0)

    median_disk_log = np.where(median_disk > 0, np.log10(median_disk), np.nan)
    y1_disk_log     = np.where(y1_disk > 0, np.log10(y1_disk), np.nan)
    y2_disk_log     = np.where(y2_disk > 0, np.log10(y2_disk), np.nan)

    lmbins_mstar = np.array([8.5,10.6])
    median_profiles_mstar = coco.density.AverageDensityProfiles(density_profiles,galaxies,lmbins_mstar,bin_by_mass='stellar')
    if hmrange == 'MW':
        lmbins_mhalo = np.array([11.4,12.4])
    elif hmrange == 'lower':
        lmbins_mhalo = np.array([10.7,11.7])
    median_profiles_mhalo = coco.density.AverageDensityProfiles(density_profiles,galaxies,lmbins_mhalo,bin_by_mass='halo')

    fig,ax = pl.subplots(1,1, figsize=(5,6))

    if task=='plot_accreted':
        ax.plot(r_mid,median_log,c='b',ls='solid',lw=2, zorder=2,label='This work')
        ax.fill_between(r_mid,y1_log,y2_log,facecolor='blue', zorder=2,alpha=0.2)

        for ibin in range(median_profiles_mstar.n_mass_bins):
            xmed,ymed = median_profiles_mstar.acc.median[ibin]
            xup,yup = median_profiles_mstar.acc.up[ibin]
            xlp,ylp = median_profiles_mstar.acc.lp[ibin]
            #ax.plot(xmed, ymed, c='green', label='COCO Mstar selected',lw=2,ls='dashed')
            #ax.fill_between(xlp,ylp,yup,lw=0,color='green',alpha=0.2)

        for ibin in range(median_profiles_mhalo.n_mass_bins):
            xmed,ymed = median_profiles_mhalo.acc.median[ibin]
            xup,yup = median_profiles_mhalo.acc.up[ibin]
            xlp,ylp = median_profiles_mhalo.acc.lp[ibin]
            ax.plot(xmed, ymed, c='orange', label='COCO Mhalo selected',lw=2,ls='solid')
            ax.fill_between(xlp,ylp,yup,lw=0,color='orange',alpha=0.2)

        #pl.title('accreted stellar halo',fontsize=12)
    elif task=='plot_insitu':
        ax.plot(r_mid,median_disk_log,c='blue',ls='solid',lw=2,zorder=2,label='face-on disk')
        ax.fill_between(r_mid,y1_disk_log,y2_disk_log,facecolor='blue', zorder=2,alpha=0.2)

        for ibin in range(median_profiles_mstar.n_mass_bins):
            xmed,ymed = median_profiles_mstar.ins.median[ibin]
            xup,yup = median_profiles_mstar.ins.up[ibin]
            xlp,ylp = median_profiles_mstar.ins.lp[ibin]
            ax.plot(xmed, ymed, c='green', label='COCO Mstar selected',lw=2,ls='dashed')
            ax.fill_between(xlp,ylp,yup,lw=0,color='green',alpha=0.2)

        for ibin in range(median_profiles_mhalo.n_mass_bins):
            xmed,ymed = median_profiles_mhalo.ins.median[ibin]
            xup,yup = median_profiles_mhalo.ins.up[ibin]
            xlp,ylp = median_profiles_mhalo.ins.lp[ibin]
            ax.plot(xmed, ymed, c='orange', label='COCO Mhalo selected',lw=2,ls='solid')
            ax.fill_between(xlp,ylp,yup,lw=0,color='orange',alpha=0.2)

        #pl.title('in-situ stellar halo',fontsize=12)

    larger_ticks(ax=ax)
    ax.set_ylim(2,10)
    ax.set_xlim(-1,2.5)
    ax.set_xlabel(r'$\log_{10}\,r/\mathrm{kpc}$',fontsize=15)
    ax.set_ylabel(r'$\log_{10}\,\Sigma_{\star}/\mathrm{M_{\odot}\,kpc^{-2}}$',fontsize=15)
    ax.legend(frameon=False, fontsize=10)
    pl.savefig('This_work_vs_coco.pdf',format='pdf')
    return

def plot_surface_density_profile_with_coco_merger_ratio(star_coor_list,dmstar_list,itree_stream,
                                                        disk_mass,halo_mass,disk_reff,
                                                        prog_mstar_list,prog_mhalo_list,prog_tage_list,
                                                        tree_lbk,r_bins=None,include_disk=False,
                                                        ratio_type='stellar'):
    """
    Split to major/intermediate/minor mergers
    How if massive progenitors just came in and not destroyed? (like LMC)
    merger ratio is at the z=infall or z=0? or z=recent?
    Maybe select recent=5Gyr?8Gyr?
    The definition of recent major merger within 5Gyr is M(zinfall<5Gyr)/Mhost>major_ratio or
    M(z=5Gyr)/Mhost>major_ratio? So for zinfall>5Gyr, halos still have M/Mhost>major_ratio also counts?

    The stream data and prog data were recorded together when doing multi-thread,
    so the itree order is the same.
    """

    nprofile = len(star_coor_list)

    if r_bins == None:
        # bin the radius within 1000 kpc
        r_bins = np.linspace(np.log10(0.1), np.log10(1000),30)
    else:
        r_bins = r_bins
    r_mid = 0.5 * (r_bins[1:] + r_bins[:-1])
    R_mid = 10**r_mid

    planes=['xy','xz','yz']

    major_itree  = []
    intermediate_itree = []
    minor_itree  = []
    sigma_total  = []
    sigma_disk_only = []

    for itree in range(nprofile):
        tree_idx = int(np.where(itree_stream == itree)[0])
        z0_disk_mass_this_tree = disk_mass[tree_idx][0]
        z0_disk_reff_this_tree = disk_reff[tree_idx][0]
        # save itree based on the merger ratio
        prog_mstar_this_tree = prog_mstar_list[tree_idx]
        prog_tage_this_tree  = prog_tage_list[tree_idx]
        # select infall < 8Gyr
        infall_lbk = prog_tage_this_tree[:,-1]-prog_tage_this_tree[:,0]
        infall_prog_mstar_this_tree = prog_mstar_this_tree[:,0][infall_lbk<=8]
        infall_prog_mhalo_this_tree = prog_mhalo_this_tree[:,0][infall_lbk<=8]
        infall_lbk = infall_lbk[infall_lbk<=8]
        # find the closest disk,halo mass at infall
        infall_tree_indices = np.searchsorted(tree_lbk,infall_lbk)
        infall_disk_mass_this_tree = disk_mass[tree_idx][infall_tree_indices]
        infall_halo_mass_this_tree = halo_mass[tree_idx][infall_tree_indices]

        if ratio_type=='stellar':
            merger_ratio = infall_prog_mstar_this_tree/infall_disk_mass_this_tree
        else:
            merger_ratio = infall_prog_mhalo_this_tree/infall_halo_mass_this_tree
        max_ratio = np.max(merger_ratio)
        if max_ratio >= 0.25:
            major_itree.append(itree)
        elif max_ratio >= 0.1:
            intermediate_itree.append(itree)
        else:
            minor_itree.append(itree)

        if include_disk:

            z0_disk_scale_radius, z0_disk_scale_height = disk_scale_radius_height(
                z0_disk_reff_this_tree
            )
            sigma_disk_total = np.array([
                sigma_exponential_mass_normalized(r, z0_disk_mass_this_tree,
                                                z0_disk_scale_radius)
                                                for r in R_mid])
            sigma_disk_only.append(sigma_disk_total)

        sigma_proj_total = []

        for i in range(3):
            _, sigma, _ = surface_density_profile(
                star_coor_list[itree],
                dmstar_list[itree],
                r_bins,
                axis=planes[i]
            )
            sigma_proj_total.append(np.asarray(sigma))

        sigma_proj_mean = np.nanmean(sigma_proj_total, axis=0)
        sigma_total.append(sigma_proj_mean)

    print('number of major/intermediate/minor mergers: ',len(major_itree),len(intermediate_itree),len(minor_itree))
    sigma_total = np.asarray(sigma_total)
    sigma_total[sigma_total <= 0] = np.nan

    merger_ratio = [major_itree,intermediate_itree,minor_itree]
    name_ratio = ['major','intermediate','minor']

    fig,ax = pl.subplots(1,1, figsize=(4,5))
    for i,mri in enumerate(merger_ratio):
        median = np.nanpercentile(sigma_total[mri], 50, axis=0)
        y1     = np.nanpercentile(sigma_total[mri], 80, axis=0)
        y2     = np.nanpercentile(sigma_total[mri], 20, axis=0)

        median_log = np.where(median > 0, np.log10(median), np.nan)
        y2_log     = np.where(y2 > 0, np.log10(y2), np.nan)
        y2_log     = np.where(y2 > 0, np.log10(y2), np.nan)
        pl.plot(r_mid,median_log,label=name_ratio[i])

    lmbins_mhalo = np.array([11.4,12.4])
    median_profiles_mhalo = coco.density.AverageDensityProfiles(density_profiles,galaxies,lmbins_mhalo,bin_by_mass='halo')
    for ibin in range(median_profiles_mhalo.n_mass_bins):
        xmed,ymed = median_profiles_mhalo.acc.median[ibin]
        xup,yup = median_profiles_mhalo.acc.up[ibin]
        xlp,ylp = median_profiles_mhalo.acc.lp[ibin]
        pl.plot(xmed, ymed, c='orange', label='COCO Mhalo selected',lw=2,ls='solid')

    larger_ticks(ax=ax)
    pl.ylim(2,10)
    pl.xlim(-1,2.5)
    pl.xlabel(r'$\log_{10}\,r/\mathrm{kpc}$',fontsize=15)
    pl.ylabel(r'$\log_{10}\,\Sigma_{\star}/\mathrm{M_{\odot}\,kpc^{-2}}$',fontsize=15)
    pl.legend(frameon=False, fontsize=10)
    return

def plot_surface_density_profile_individual_tree(star_coor_list,dmstar_list,r_bins=None):
    """
    This uses the testing stripping function to represent the stripped mass distribution
    """
    m20gals = ['n1042','n1084','n2903','n3351','n1084','n3368','n4220','n4258']

    if r_bins == None:
        # bin the radius within 1000 kpc
        r_bins = np.linspace(-1,5,30)
    else:
        r_bins = r_bins


    pl.figure(figsize=(10,4))
    axis=['xy','xz','yz']
    for i in range(0,3):
        pl.subplot(1,3,i+1)
        r_mid,sigma,sigma_each = surface_density_profile(star_coor_list,dmstar_list,r_bins,axis=axis[i])

        pl.plot(r_mid,np.log10(sigma),c='k',label='total')
        first = True
        for m20gal in m20gals:
            m20 = Table.read('/data/apcooper/coco/obs/merritt20/m20_{}_mstar_kpc.csv'.format(m20gal))
            pl.plot(np.log10(m20['x']),m20['lsmsd'],c='purple',lw=1.5,alpha=0.5,zorder=10,
                    label='Merritt et al. 2020' if first else None)
            first = False
        for s in sigma_each:
            pl.plot(r_mid,np.log10(s),c='grey',alpha=0.5)
        pl.title(axis[i])
        pl.xlim(-1,3)
        pl.ylim(0,10)
        if i==0:
            pl.xlabel(r'$\log_{10}\,R/\mathrm{kpc}$',fontsize=10)
            pl.ylabel(r'$\log_{10}\,\Sigma_{\star}/\mathrm{M_{\odot}\,kpc^{-2}}$')
    pl.legend(frameon=False, fontsize=8)
    pl.tight_layout()
    return

def plot_surface_density_profile_all_tree(star_coor_list,dmstar_list,r_bins=None):
    """
    """
    nprofile = len(star_coor_list)

    m20gals = ['n1042','n1084','n2903','n3351','n1084','n3368','n4220','n4258']

    if r_bins == None:
        # bin the radius within 1000 kpc
        r_bins = np.linspace(-1,5,30)
    else:
        r_bins = r_bins

    fig,ax = pl.subplots(1,3, figsize=(10,4))
    planes=['xy','xz','yz']
    for i in range(0,3):
        sigma_total = []
        for itree in range(0,nprofile):
            r_mid,sigma,_ = surface_density_profile(star_coor_list[itree],dmstar_list[itree],r_bins,axis=planes[i])
            sigma_total.append(sigma)
        sigma_total = np.asarray(sigma_total)
        sigma_total[sigma_total <= 0] = np.nan
        nvalid = np.sum(np.isfinite(sigma_total), axis=0)
        median = np.nanpercentile(sigma_total,50,axis=0)
        y1     = np.nanpercentile(sigma_total,80,axis=0)
        y2     = np.nanpercentile(sigma_total,20,axis=0)
        median_log = np.where(median > 0, np.log10(median), np.nan)
        y1_log     = np.where(y1 > 0, np.log10(y1), np.nan)
        y2_log     = np.where(y2 > 0, np.log10(y2), np.nan)
        ax[i].plot(r_mid,median_log,c='b',ls='dashed',label='This Work')
        ax[i].fill_between(r_mid,y1_log,y2_log,facecolor='cyan', zorder=1,alpha=0.5)
        first = True
        for m20gal in m20gals:
            m20 = Table.read('/data/apcooper/coco/obs/merritt20/m20_{}_mstar_kpc.csv'.format(m20gal))
            ax[i].plot(np.log10(m20['x']),m20['lsmsd'],c='purple',lw=1.5,alpha=0.5,zorder=10,
                    label='Merritt et al. 2020' if first else None)
            first = False
        ax[i].set_ylim(2,10)
        ax[i].set_xlim(-1,2.5)
        ax[i].set_title(planes[i])
    ax[0].set_xlabel(r'$\log_{10}\,r/\mathrm{kpc}$')
    ax[0].set_ylabel(r'$\log_{10}\,\Sigma_{\star}/\mathrm{M_{\odot}\,kpc^{-2}}$')
    ax[0].legend(frameon=False, fontsize=8)
    return

def plot_surface_density_profile_all_tree_average(star_coor_list,dmstar_list,itree_stream,
                                                 disk_mass,disk_reff,
                                                 r_bins=None,include_disk=False,disk='MN',
                                                 plot_disk='together',coco_style='individual',
                                                 bin_method='stellar'):
    """

    Parameters
    ----------
    include_disk: assuming a face-on MN disk

    Note
    ----
    Maybe there is a projection effect in satgen when creating initial infall condition. (need to check)
    So it is better to average the three projeciton (or more) if showing only one projection.
    """
    nprofile = len(star_coor_list)

    # Merritt 2020
    m20gals = ['n1042','n1084','n2903','n3351','n1084','n3368','n4220','n4258']


    if r_bins == None:
        # bin the radius within 1000 kpc
        r_bins = np.linspace(np.log10(0.1), np.log10(1000),30)
    else:
        r_bins = r_bins
    r_mid = 0.5 * (r_bins[1:] + r_bins[:-1])
    R_mid = 10**r_mid

    planes=['xy','xz','yz']

    sigma_total = []
    sigma_disk_only = []

    for itree in range(nprofile):

        if include_disk:
            tree_idx = np.where(itree_stream == itree)[0]
            z0_disk_mass_this_tree = disk_mass[tree_idx][0]
            z0_disk_reff_this_tree = disk_reff[tree_idx][0]

            z0_disk_scale_radius, z0_disk_scale_height = disk_scale_radius_height(
                z0_disk_reff_this_tree
            )
            if disk=='MN':
                sigma_disk_total = np.array([
                    Sigma_faceon(r, z0_disk_mass_this_tree,
                                z0_disk_scale_radius,
                                z0_disk_scale_height)
                                for r in R_mid])
            elif disk=='exponential':
                sigma_disk_total = np.array([
                    sigma_exponential_mass_normalized(r, z0_disk_mass_this_tree,
                                                     z0_disk_scale_radius)
                                            for r in R_mid])
            sigma_disk_only.append(sigma_disk_total)

        sigma_proj_total = []

        for i in range(3):
            _, sigma, _ = surface_density_profile(
                star_coor_list[itree],
                dmstar_list[itree],
                r_bins,
                axis=planes[i]
            )
            sigma_proj_total.append(np.asarray(sigma))

        sigma_proj_mean = np.nanmean(sigma_proj_total, axis=0)

        if include_disk:
            if plot_disk=='together':
                sigma_total.append(sigma_proj_mean + sigma_disk_total)
            elif plot_disk=='separate':
                sigma_total.append(sigma_proj_mean)
        else:
            sigma_total.append(sigma_proj_mean)

    sigma_total = np.asarray(sigma_total)
    sigma_disk_only = np.asarray(sigma_disk_only)

    sigma_total[sigma_total <= 0] = np.nan

    median = np.nanpercentile(sigma_total, 50, axis=0)
    y1     = np.nanpercentile(sigma_total, 80, axis=0)
    y2     = np.nanpercentile(sigma_total, 20, axis=0)

    median_log = np.where(median > 0, np.log10(median), np.nan)
    y1_log     = np.where(y1 > 0, np.log10(y1), np.nan)
    y2_log     = np.where(y2 > 0, np.log10(y2), np.nan)

    median_disk = np.nanpercentile(sigma_disk_only, 50, axis=0)
    y1_disk     = np.nanpercentile(sigma_disk_only, 80, axis=0)
    y2_disk     = np.nanpercentile(sigma_disk_only, 20, axis=0)

    median_disk_log = np.where(median_disk > 0, np.log10(median_disk), np.nan)
    y1_disk_log     = np.where(y1_disk > 0, np.log10(y1_disk), np.nan)
    y2_disk_log     = np.where(y2_disk > 0, np.log10(y2_disk), np.nan)

    fig,ax = pl.subplots(1,1, figsize=(5,6))

    if include_disk:
        if plot_disk=='together':
            label1='This Work+face-on disk'
        elif plot_disk=='separate':
            label1='This Work'
        ax.plot(r_mid,median_log,c='b',ls='dashed',lw=2, zorder=2,label=label1)
        ax.plot(r_mid,median_disk_log,c='green',ls='dashed',lw=2,zorder=2,label='face-on disk')
    else:
        ax.plot(r_mid,median_log,c='b',ls='dashed',lw=2, zorder=2,label='This Work')
    ax.fill_between(r_mid,y1_log,y2_log,facecolor='blue', zorder=2,alpha=0.2)
    ax.fill_between(r_mid,y1_disk_log,y2_disk_log,facecolor='green', zorder=2,alpha=0.2)

    first = True
    for m20gal in m20gals:
        m20 = Table.read('/data/apcooper/coco/obs/merritt20/m20_{}_mstar_kpc.csv'.format(m20gal))
        ax.plot(np.log10(m20['x']),m20['lsmsd'],c='purple',lw=1.5,alpha=0.5,zorder=10,
                label='Merritt et al. 2020' if first else None)
        first = False

    plot_coco(ax,style=coco_style)
    larger_ticks(ax=ax)
    ax.set_ylim(2,10)
    ax.set_xlim(-1,2.5)
    ax.set_xlabel(r'$\log_{10}\,r/\mathrm{kpc}$',fontsize=15)
    ax.set_ylabel(r'$\log_{10}\,\Sigma_{\star}/\mathrm{M_{\odot}\,kpc^{-2}}$',fontsize=15)
    ax.legend(frameon=False, fontsize=10) #loc=(1.1,0)

    return

def plot_surface_density_profile_all_tree_average_splitting(star_coor_list,dmstar_list,itree_stream,
                                                            disk_mass,disk_reff,
                                                            r_bins=None,include_disk=False,disk='MN',
                                                            bin_method='macc'):
    """
    """

    nprofile = len(star_coor_list)

    # Merritt 2020
    m20gals = ['n1042','n1084','n2903','n3351','n1084','n3368','n4220','n4258']

    _,root_mass,_,_ = read_tree()
    data_path = '/lfs/data/chungwen/SatgenOutput/pchtree_mass_2e11_to_2e12/ludlow_concentration/emerge_disk/cut_both1e7_09_atleast09/'
    prog_dataset_names = ['orbit','itree','levels_at_tsteps','nprog','prog_masses','prog_mstars','radii','status','t_proc','tage','tsteps']
    prog = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_EMERGE_disk.hdf5',prog_dataset_names,group='/Results')
    m_acc    = compute_mass(prog,root_mass,task='m_acc')
    m_stream = compute_mass(prog,root_mass,task='m_stream')
    m_both = m_acc+m_stream

    # Create color map
    #cmap = cm.Blues
    #norm = mcolors.Normalize(vmin=np.log10(median_profiles.centroids.min()*0.9), 
    #                     vmax=np.log10(mediif first else Nonean_profiles.centroids.max()*1.1))

    if r_bins == None:
        # bin the radius within 1000 kpc
        r_bins = np.linspace(np.log10(0.1), np.log10(1000),30)
    else:
        r_bins = r_bins
    r_mid = 0.5 * (r_bins[1:] + r_bins[:-1])
    R_mid = 10**r_mid

    fig,ax = pl.subplots(1,4, figsize=(20,6))
    planes=['xy','xz','yz']

    sigma_total = []
    sigma_proj = []
    sigma_disk_only = []

    for itree in range(nprofile):

        if include_disk:
            tree_idx = np.where(itree_stream == itree)[0]
            z0_disk_mass_this_tree = disk_mass[tree_idx][0]
            z0_disk_reff_this_tree = disk_reff[tree_idx][0]

            z0_disk_scale_radius, z0_disk_scale_height = disk_scale_radius_height(
                z0_disk_reff_this_tree
            )
            if disk=='MN':
                sigma_disk_total = np.array([
                    Sigma_faceon(r, z0_disk_mass_this_tree,
                                z0_disk_scale_radius,
                                z0_disk_scale_height)
                                for r in R_mid])
            elif disk=='exponential':
                sigma_disk_total = np.array([
                    sigma_exponential_mass_normalized(r, z0_disk_mass_this_tree,
                                                     z0_disk_scale_radius)
                                                     for r in R_mid])
            sigma_disk_only.append(sigma_disk_total)

        sigma_proj_total = []

        for i in range(3):
            _, sigma, _ = surface_density_profile(
                star_coor_list[itree],
                dmstar_list[itree],
                r_bins,
                axis=planes[i]
            )
            sigma_proj_total.append(np.asarray(sigma))

        sigma_proj_mean = np.nanmean(sigma_proj_total, axis=0)
        sigma_proj.append(sigma_proj_mean)

        sigma_total.append(sigma_proj_mean + sigma_disk_total)

    sigma_total = np.asarray(sigma_total)
    sigma_proj  = np.asarray(sigma_proj)
    sigma_disk_only = np.asarray(sigma_disk_only)

    sigma_total[sigma_total <= 0] = np.nan
    sigma_proj[sigma_proj <= 0] = np.nan

    m_both_split_ranges = [(m_both<=1e8),
                           (m_both>1e8)&(m_both<=1e9),
                           (m_both>1e9)&(m_both<=1e10),
                           (m_both>1e10)]
    titles = ['[1e8]','[1e8,1e9]','[1e9,1e10]','[1e10]']

    for i,split_range in enumerate(m_both_split_ranges):
        median = np.nanpercentile(sigma_total[split_range], 50, axis=0)
        y1     = np.nanpercentile(sigma_total[split_range], 80, axis=0)
        y2     = np.nanpercentile(sigma_total[split_range], 20, axis=0)

        median_log = np.where(median > 0, np.log10(median), np.nan)
        y1_log     = np.where(y1 > 0, np.log10(y1), np.nan)
        y2_log     = np.where(y2 > 0, np.log10(y2), np.nan)

        median_proj = np.nanpercentile(sigma_proj[split_range], 50, axis=0)
        y1_proj     = np.nanpercentile(sigma_proj[split_range], 80, axis=0)
        y2_proj     = np.nanpercentile(sigma_proj[split_range], 20, axis=0)

        median_proj_log = np.where(median_proj > 0, np.log10(median_proj), np.nan)
        y1_proj_log     = np.where(y1_proj > 0, np.log10(y1_proj), np.nan)
        y2_proj_log     = np.where(y2_proj > 0, np.log10(y2_proj), np.nan)

        median_disk = np.nanpercentile(sigma_disk_only[split_range], 50, axis=0)
        y1_disk     = np.nanpercentile(sigma_disk_only[split_range], 80, axis=0)
        y2_disk     = np.nanpercentile(sigma_disk_only[split_range], 20, axis=0)

        median_disk_log = np.where(median_disk > 0, np.log10(median_disk), np.nan)
        y1_disk_log     = np.where(y1_disk > 0, np.log10(y1_disk), np.nan)
        y2_disk_log     = np.where(y2_disk > 0, np.log10(y2_disk), np.nan)

        ax[i].plot(r_mid,median_proj_log,c='b',ls='dashed',lw=2, zorder=2,label='This work')
        ax[i].plot(r_mid,median_log,c='r',ls='dashed',lw=5, zorder=3,label='This work+ disk')
        ax[i].plot(r_mid,median_disk_log,c='green',ls='dashed',lw=2,zorder=2,label='face-on disk')
        ax[i].fill_between(r_mid,y1_log,y2_log,facecolor='r', zorder=2,alpha=0.2)

        first = True
        for m20gal in m20gals:
            m20 = Table.read('/data/apcooper/coco/obs/merritt20/m20_{}_mstar_kpc.csv'.format(m20gal))
            ax[i].plot(np.log10(m20['x']),m20['lsmsd'],c='purple',lw=1.5,alpha=0.5,zorder=10,
                    label='Merritt et al. 2020' if first else None)
            first = False

        # COCO data
        plot_coco(ax[i],style='individual')

        ax[i].set_ylim(2,10)
        ax[i].set_xlim(-1,2.5)
        ax[i].set_title(titles[i],fontsize=15)
        larger_ticks(ax=ax[i])
    ax[0].set_xlabel(r'$\log_{10}\,r/\mathrm{kpc}$',fontsize=15)
    ax[0].set_ylabel(r'$\log_{10}\,\Sigma_{\star}/\mathrm{M_{\odot}\,kpc^{-2}}$',fontsize=15)
    ax[0].legend(frameon=False, fontsize=10)
    pl.tight_layout()
    return
#########################################################
### check
## This place check assumptions or models.

def plot_fitting(age,mass,conc,zred):
    """
    A check for polyfit and interp1d on the mass and virial radius
    """

    halo_profile = NFW(mass,conc,Delta=200.,z=zred,sf=1.)
    Rv = halo_profile.rh

    # Interpolate in log mass
    lgmass = np.log10(mass)
    fmi = interp1d(age,lgmass,fill_value='extrapolate')
    fri = interp1d(age,Rv,fill_value='extrapolate')

    pmp = np.polyfit(age,lgmass,deg=4)
    fmp = np.poly1d(pmp)
    prp = np.polyfit(age,Rv,deg=4)
    frp = np.poly1d(prp)
    
    x_new = np.arange(13,0,-0.1)
    pl.subplot(121)
    pl.plot(x_new,fmi(x_new),label='interp')
    pl.plot(x_new,fmp(x_new),label='polyfit')
    pl.title('Halo mass')
    pl.xlabel('age')
    pl.ylabel('$\log_{10}\,\mathrm{M_{200}}$')
    pl.legend(frameon=False)

    pl.subplot(122)
    pl.plot(x_new,fri(x_new))
    pl.plot(x_new,frp(x_new))
    pl.xlabel('age')
    pl.ylabel('Rv')
    pl.title('Virial radius')
    return

def plot_mdyn_Rvdyn(age,mass,conc,zred):
    """
    A check for the m_dot_dyn and Rv_dot_dyn
    """

    lgm_dot_dyn, Rv_dot_dyn,_,_,_ = sga.calculate_tdyn(age,mass,conc,zred,return_lgmass=True)
    

    pl.subplot(121)
    pl.plot(age,lgm_dot_dyn)
    pl.xlabel('age (Gyr)')
    pl.ylabel(r'$\mathrm{\dot{M_{tdyn}} \, (\log_{10}\,M_{\odot}})$')

    pl.subplot(122)
    pl.plot(age,Rv_dot_dyn)    
    pl.ylabel(r'$\dot{Rv_{tdyn}} \, (\mathrm{kpc})$')
    return

def plot_EMERGE(itree,hm,z,cosmology,choice='history',result=None):
    '''
    If reuslt is given, itree is not used.
    '''
    b13sm = np.array([gh.lgMs_B13(np.log10(hm_tree),z) for hm_tree in hm])
    b13sm50 = np.percentile(b13sm,50,axis=0)
    b13sm1sigma = np.percentile(b13sm,84.1,axis=0)
    b13sm_err = b13sm1sigma-b13sm50

    if result == None:
        hosts = [
            sga.Host(hm[i], z, cosmology,
                     fd=0.1, flattening=25.,
                     disk_method='EMERGE',
                    walk_tree='forward',
                    cooling_threshold=False,
                    z0_smhm=False)
            for i in itree
        ]
    
        z      = np.array([h._tree_zred      for h in hosts])
        dm     = np.log10([h.disk_mass       for h in hosts])
        hm     = np.log10([h.mass            for h in hosts])
        b13sm  = np.array([gh.lgMs_B13(np.log10(h.mass), h._tree_zred) for h in hosts])
        z1     = np.array([h._tree_zred+1      for h in hosts])
    else:
        z  = np.tile(z,(1000,1))
        dm = np.log10(result['main_branch_disk_mass'])
        hm = np.log10(result['main_branch_halo_mass'])
        
    
    if choice=='history':
        pl.plot(z.T,dm.T,alpha=0.2,c='b')
        pl.plot(z.T,hm.T,alpha=0.2,c='k')
        pl.xlabel('z')
    elif choice=='smhm':
        pl.plot(hm.T,dm.T,alpha=0.1)
        ax = pl.gca()
        #plot_b13_satgen(ax)
        plot_m18(ax)
        pl.xlabel('$\mathrm{log_{10}\,halo\,mass}$')

    pl.ylabel('$\mathrm{log_{10}\,disk\,mass}$')

    return

def plot_tdyn(host):
    """
    Compare tdyn from profile and calculation

    tdyn_pf = dens_profile.tdyn(Rv)
    tdyn_cal = (Rv^3 / G / M)^0.5
    """

    pl.plot(host.Rv,host.tdyn_pf,label='profile')
    pl.plot(host.Rv,host.tdyn_cal,label='calculate')
    pl.xlabel('Rv (kpc)')
    pl.ylabel('tdyn (Gyr)')
    pl.legend(frameon=False)
    return

def plot_m18smhm(progs):
    """
    This is testing M18 integrated baryon conversion efficiency
    making a smhm relation
    """
    hm = []
    sm = []
    for prog in progs:
        hm.append(prog.mass)
        sm.append(prog.mstar)

    ax = pl.gca()
    plot_m18(ax)

    pl.scatter(np.log10(hm),np.log10(sm),s=1,label='M18')
    pl.xlabel('$\mathrm{log_{10}\,halo\,mass}$')
    pl.ylabel('$\mathrm{log_{10}\,stellar\,mass}$')
    pl.legend(frameon=False,loc='lower right')
    return

def plot_integrate_instant(main,root_mass):
    """
    This checks the instantanious baryonic conversion efficiency can grow 
    a disk that matches the integrated smhm relation
    """

    disk_mass = main['emerge']['main_branch_disk_mass']
    halo_mass = main['emerge']['main_branch_halo_mass']

    root_lgmass    = np.log10(halo_mass[:,0])
    z0_disk_lgmass = np.log10(disk_mass[:,0])

    fb = 0.156
    #root_lgmass = np.log10(root_mass)

    # redshift 0 m18 relation with 1 sigma
    hm_range = np.arange(10,14,0.2)
    upper_lgsm = []
    lower_lgsm = []
    for hm in hm_range:
        hm = 10**hm
        e,sigma = sga.integrate_b_conversion(0,hm)
        mean_sm = fb*(hm)*e
        upper_lgsm.append(np.log10(mean_sm)+sigma)
        lower_lgsm.append(np.log10(mean_sm)-sigma)

    ax = pl.gca()
    #plot_m18(ax)

    ax.fill_between(hm_range,upper_lgsm,lower_lgsm,alpha=0.5)
    pl.scatter(root_lgmass,z0_disk_lgmass,s=1)
    pl.xlim(11,13)
    pl.ylim(7,12)
    return

def plot_instb_coeff(redshift_index=None):
    """
    M18 fig2
    redshift_index 13 ~ 0.1
    """

    tree_redshifts,root_mass,_,tree_main_branch_masses = read_tree()

    pl.figure()
    if redshift_index is None:
        for i,z in enumerate(tree_redshifts):
            ez = np.array(sga.inst_b_conversion(tree_main_branch_masses[:,i],z,h=0.7))
        
            pl.scatter(np.log10(tree_main_branch_masses[:,i]),ez,s=1)
    else:
        ez = np.array(sga.inst_b_conversion(tree_main_branch_masses[:,redshift_index],
                      tree_redshifts[redshift_index]))
        pl.scatter(np.log10(tree_main_branch_masses[:,redshift_index]),ez,s=1)
    return

def plot_inteb_coeff():
    """
    """
    zred = np.arange(0,14,2)
    lghmrange = np.arange(8,13,0.2)
    hmrange = 10**lghmrange

    for z in zred:
        ez,sigma = np.array(sga.integrate_b_conversion(z,hmrange,h=0.7))
        pl.errorbar(lghmrange,ez,yerr=sigma,label=f'{z}')

    pl.legend(frameon=False,prop={'size':8},ncols=2)
    pl.xlabel('halo mass',fontsize=15)
    pl.ylabel('e',fontsize=15)
    return

def plot_mstar_mb():
    """
    index 13 ~ z=0.1
    m18 fig12
    """
    tree_redshifts,root_mass,progs,tree_main_branch_masses = read_tree(tree_setup='extend')
    fb = 0.156

    mass  = tree_main_branch_masses[:,13]
    mstar = sga.mod_Mstar(mass,h=0.7,z=tree_redshifts[13],choice='m18',task='Mstar')
    mb    = mass * fb
    y1    = mstar/mb

    prog_z01_index  = np.where(progs['ProgenitorZred']==tree_redshifts[13])[0]
    print('total number of progenitros: ',len(progs['ProgenitorZred']),'number of progenitors at z=0.1: ',len(prog_z01_index))
    prog_z01_masses = progs['ProgenitorMass'][prog_z01_index]
    mstar_prog = sga.mod_Mstar(prog_z01_masses,h=0.7,z=tree_redshifts[13],choice='m18',task='Mstar')
    mb_prog    = prog_z01_masses * fb
    print(mstar_prog.shape,mb_prog.shape)
    y2 = mstar_prog/mb_prog
    print(np.argmax(y2))
    print('halo mass: ',prog_z01_masses[np.argmax(y2)])
    print(mstar_prog[np.argmax(y2)],mb_prog[np.argmax(y2)])

    # more checks
    frac = mstar_prog/prog_z01_masses
    print('The highest mstar to mhalo fraction: ',max(frac),'mstar: ',mstar_prog[np.argmax(frac)],'mhalo: ',prog_z01_masses[np.argmax(frac)])
    print('This prog index: ',prog_z01_index[np.argmax(frac)])
    
    y = np.concatenate((y1,y2))
    lg_main_mass = np.log10(mass)
    lg_prog_mass = np.log10(prog_z01_masses)
    x = np.concatenate((lg_main_mass,lg_prog_mass))

    totmass = 10**x
    e,sigma = np.array(sga.integrate_b_conversion(tree_redshifts[13],totmass,h=0.7))

    moster_18_default = partial(moster_18_eff_func,M1=10**11.80,epsilon=0.14,beta=1.75,gamma=0.57)
    e1 = moster_18_default(totmass)

    lghmrange = np.arange(10,14,0.1)
    e2,_ = np.array(sga.integrate_b_conversion(tree_redshifts[13],10**lghmrange,h=0.7))

    pl.figure()
    pl.scatter(x,y,s=1,c='r')
    pl.scatter(x,e,s=1,c='k')
    pl.scatter(x,e1,s=1,c='b')
    pl.plot(lghmrange,e2,c='green')
    return

def plot_conc_to_hm_smooth_zhao():
    """
    Check smooth concentraion vs zhao concentration by redshifts (Ludluw 2016. fig.4)
    z = 1,2,3 ~ index = 77,102,123

    Note: The concentration function reads the tage history, not the instantanious age.
    The current eps tree has only main branch history, so it can not reproduce the figure 4 
    due to the real abundance.
    Unless there is a need for running the whole tree history, this will not be checked.
    Also the concentration for subhalos used DM14 in Satgen.
    """

    return

def plot_conc_and_rs_to_tage_smooth_zhao(itree):
    """
    """

    tree_redshift,root_mass,progenitors,mainbranch_mass = read_tree()

    tage = cosmology.age(tree_redshift).value

    # Concentration
    smooth_c = sga.smooth_c(mainbranch_mass[itree],tage,version='zhao')
    zhao_c   = sga.halo_mah_to_zhao_c_nfw(mainbranch_mass[itree],tage)
    ludlow_c = llc.ludlow_concentration(mainbranch_mass[itree],tree_redshift,cosmology)
    print('Ludlow c[-10:-1]: ',ludlow_c[-10:-1])
    print('Zhao c[-10:-1]: ',zhao_c[-10:-1])
    print('Redshift[-10:-1]: ',tree_redshift[-10:-1])
    lgsmooth_c = np.log10(smooth_c)
    lgzhao_c   = np.log10(zhao_c)
    lgludlow_c = np.log10(ludlow_c)

    # Scale radius
    nlev = len(tree_redshift)
    idx_iter = range(nlev)

    # halo mass
    mhalo   = mainbranch_mass[itree]
    lgmhalo = np.log10(mhalo)

    srs = []
    zrs = []
    grs = []
    lrs = []
    srvir = []
    zrvir = []
    for i in idx_iter:

        mass_i = mhalo[i]
        sc_i = smooth_c[i]
        zc_i = zhao_c[i]
        lc_i = ludlow_c[i]
        z_i = tree_redshift[i]

        shalo_profile = NFW(mass_i,sc_i,Delta=200.,z=z_i,sf=1.)
        zhalo_profile = NFW(mass_i,zc_i,Delta=200.,z=z_i,sf=1.)
        lhalo_profile = NFW(mass_i,lc_i,Delta=200.,z=z_i,sf=1.)

        srs.append(shalo_profile.rs)
        zrs.append(zhalo_profile.rs)
        lrs.append(lhalo_profile.rs)

        # self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        # self.rhoh = self.Deltah * self.rhoc
        # self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        srvir.append(shalo_profile.rh)
        zrvir.append(zhalo_profile.rh)

        if i<4:
            print('Mhalo: ',mass_i)
            print('M200: ',shalo_profile.M(shalo_profile.rh))

        # The virial radius is the same
        rs_gao = scale_radius_gao(mass_i,z_i) * shalo_profile.rh
        grs.append(rs_gao)

    # It says the mass can be M200 or Mvir depending on the model choice.
    # But there is only one choice (DM14), also the difference between M200 and
    # Mvir is small.
    dmc = init.concentration(mhalo,tree_redshift)
    lgdmc = np.log10(dmc)

    pl.figure(figsize=(16,6))
    pl.subplot(121)

    pl.plot(tage,srs,c='k',label='smooth')
    pl.plot(tage,zrs,c='r',label='Zhao')
    pl.plot(tage,grs,c='green',label='Gao')
    pl.plot(tage,lrs,c='blue',label='Ludlow')

    pl.xlabel('age (Gyr)',fontsize=15)
    pl.ylabel('$\mathrm{r_{s} \, (kpc)}$',fontsize=15)
    pl.legend(loc='upper left',frameon=False,prop={'size':10})

    pl.subplot(122)

    pl.plot(tage,lgsmooth_c,c='k',label='smooth')
    pl.plot(tage,lgzhao_c,c='r',label='Zhao')
    pl.plot(tage,lgdmc,c='purple',label='DM14')
    pl.plot(tage,lgludlow_c,c='blue',label='Ludlow')

    pl.xlabel('age (Gyr)',fontsize=15)
    pl.ylabel('$\mathrm{log_{10}\, c}$',fontsize=15)

    pl.legend(loc='lower right',frameon=False,prop={'size':10})

    return

def plot_scale_radius_tage(itree):
    """
    check halo scale radius and disk scale radius
    """
    tree_redshift,root_mass,progenitors,mainbranch_mass = read_tree()

    mhalo = mainbranch_mass[itree]

    host = sga.Host(mhalo,tree_redshift,cosmology,fd=0.1,flattening=25.,disk_method='EMERGE',
            walk_tree='forward',cooling_threshold=False,z0_smhm=False,smhm='m18')

    rs_disk = host.scale_radius
    rs_halo = [hhdp.rs for hhdp in host.halo_dens_profile]
    conc = host.concentration
    tage = host.t_age

    ax1 = pl.subplot(111)

    pl.plot(tage,rs_disk,c='k',label='disk')
    pl.plot(tage,rs_halo,c='r',label='halo')
    pl.ylabel('$r_{s}$',fontsize=15)
    pl.legend(loc='upper left',frameon=False,prop={'size':10})
    
    ax2 = ax1.twinx()
    pl.plot(tage,conc,c='b')
    pl.ylabel('c',fontsize=15)

    ax1.set_xlabel('age (Gyr)',fontsize=15)
    
    return

def plot_flattening(itree):
    """
    Check various flattening (disk scale radius/disk scale height)
    The default satgen uses flattening=25

    disk_profile = MN(disk_mass,scale_radius,scale_height)
    """
    
    tree_redshift,root_mass,progenitors,mainbranch_mass = read_tree()

    mhalo = mainbranch_mass[itree]

    host1 = sga.Host(mhalo,tree_redshift,cosmology,fd=0.1,flattening=25.,disk_method='EMERGE',
            walk_tree='forward',cooling_threshold=False,z0_smhm=False,smhm='m18')

    host2 = sga.Host(mhalo,tree_redshift,cosmology,fd=0.1,flattening=5.,disk_method='EMERGE',
            walk_tree='forward',cooling_threshold=False,z0_smhm=False,smhm='m18')

    host3 = sga.Host(mhalo,tree_redshift,cosmology,fd=0.1,flattening=45.,disk_method='EMERGE',
            walk_tree='forward',cooling_threshold=False,z0_smhm=False,smhm='m18')


    scale_radius1 = host1.scale_radius
    scale_radius2 = host2.scale_radius
    scale_radius3 = host3.scale_radius
    print(scale_radius1[0],scale_radius2[0],scale_radius3[0])
    lg_disk_mass1 = np.log10(host1.disk_mass)
    lg_disk_mass2 = np.log10(host2.disk_mass)
    lg_disk_mass3 = np.log10(host3.disk_mass)

    pl.figure()
    pl.plot(lg_disk_mass1,scale_radius1,label='flattening=25')
    pl.plot(lg_disk_mass2,scale_radius1,label='flattening=5')
    pl.plot(lg_disk_mass3,scale_radius3,label='flattening=45')

    pl.xlabel('$\mathrm{log_{10}\,disk\,mass\,M_{\odot}}$')
    pl.ylabel('scale radius')
    pl.legend(frameon=False,prop={'size':10})
    return

def plot_reff_sm(main,sample=1):
    """
    The disk data from the Legacy survey was provided by Li-Wen

    Parameters
    ------
    sample:
    1: spectral-z R50
    2: spectral-z Rmajor-axis
    3: photo-z    R50
    4: photo-z    Rmajor-axis
    """

    Lsurvey_data = fits.open('/data/chungwen/cwt/mass_size_relation/size_mass_pub.fits')
    reff = main['main_branch_disk_reff']
    mass = main['main_branch_disk_mass']

    mask = Lsurvey_data[sample].data['err_all'] > 0

    fig,ax = pl.subplots(1,1, figsize=(5,5))
    pl.scatter(np.log10(mass),np.log10(reff),label='This Work',c='grey',alpha=0.5,s=2)
    ax.errorbar(Lsurvey_data[sample].data['sm'][mask],
            Lsurvey_data[sample].data['mu_all'][mask],
            yerr=Lsurvey_data[sample].data['err_all'][mask],label='Liao 2026')
    pl.xlabel('$\mathrm{log_{10}\,M_{\star}/\,M_{\odot}}$',fontsize=15)
    pl.ylabel('$\mathrm{log_{10}\,R/\,kpc}$',fontsize=15)
    pl.legend(frameon=False,prop={'size':10})
    return

def check_disk_mean_density(itree_stream,disk_mass,disk_reff):
    """
    z0 disk mean density within radius
    """

    tree_idx = np.where(itree_stream==0)[0]
    z0_disk_mass_this_tree = disk_mass[tree_idx][0]
    print('disk mass: ',z0_disk_mass_this_tree)
    z0_disk_reff_this_tree = disk_reff[tree_idx][0]
    a,b = disk_scale_radius_height(z0_disk_reff_this_tree)
    print('scale_radius: ',a,'scale_height: ',b)
    disk_profile = MN(z0_disk_mass_this_tree,a,b)

    rbins = np.arange(0,30)
    m = disk_profile.M(rbins)

    pl.plot(np.log10(rbins),np.log10(m))
    pl.xlabel('$\mathrm{log_{10}\,r/\,kpc}$')
    pl.ylabel('$\mathrm{log_{10}\,m/\,\mathrm{M_{\odot}}}$')
    return

def plot_macc_histogram():
    """
    """
    return
#########################################################
### debug
def debug_prog_matching(data,models,r=10,f=100):
    """

    Conclusion: The tree orders are different for different hdf5
                due to multithread processing. However the progenitors
                for each tree are the same order.
    """

    prog1 = data[models[0]]
    prog1_treeidx = prog1['tree_idx']
    prog1_initial_radii = prog1['radii'][:,0]
    prog1_initial_masses  = prog1['prog_masses'][:,0]

    prog2 = data[models[1]]
    prog2_treeidx = prog2['tree_idx']
    prog2_initial_radii = prog2['radii'][:,0]
    prog2_initial_masses  = prog2['prog_masses'][:,0]

    rcounter = 0
    mcounter = 0
    for i in np.arange(0,1000):
        p1irs = prog1_initial_radii[np.where(prog1_treeidx==i)[0]]
        p2irs = prog2_initial_radii[np.where(prog2_treeidx==i)[0]]
        for p1ir,p2ir in zip(p1irs,p2irs):
            if abs(p1ir-p2ir) > r:
                rcounter+=1

        p1ims = prog1_initial_masses[np.where(prog1_treeidx==i)[0]]
        p2ims = prog2_initial_masses[np.where(prog2_treeidx==i)[0]]
        for p1im,p2im in zip(p1ims,p2ims):
            if ((p1im/p2im) > f) | ((p1im/p2im) < (1/f)):
                mcounter+=1
 
    #i_rdiff_far=[]
    #for i,(p1ir,p2ir) in enumerate(zip(prog1_initial_radii,prog2_initial_radii)):
    #    if abs(p1ir-p2ir) > r:
    #        i_rdiff_far.append(i)

    #i_hmdiff_large=[]
    #for i,(p1im,p2im) in enumerate(zip(prog1_initial_masses,prog2_initial_masses)):
    #    if ((p1im/p2im) > f) | ((p1im/p2im) < (1/f)):
    #        i_hmdiff_large.append(i)

    print("Progenitors have far initial radius difference ({}kpc): ".format(r),rcounter)
    print("Progenitors have large initial mass difference ({} times): ".format(f),mcounter)
    return #i_rdiff_far,i_hmdiff_large

def debug_smhm_range(nrepeat,zrange=None,hmrange=None,choice='B13',plot_figure=True):
    """
    """
    if zrange is None:
        zrange  = np.arange(0,24,0.2)
    else:
        zrange==zrange

    if hmrange is None:
        hmrange = 10**np.arange(9,12.7,0.2)    
        lghmrange = np.log10(hmrange)
    else:
        hmrange==hmrange
        lghmrange = np.log10(hmrange)

    maxsm = 0

    for hm in hmrange:
        for _ in np.arange(nrepeat):
            sm = np.array([sga.mod_Mstar(hm,z=z,choice=choice,task='Mstar') for z in zrange])
            maxsm_this_range = max(sm)
            if maxsm_this_range>maxsm:
                maxsm = maxsm_this_range
                hm_this_sm = hm
                z_this_sm  = zrange[np.argmax(sm)]
            if plot_figure:
                pl.scatter(np.repeat(np.log10(hm),len(sm)),np.log10(sm),s=2)

    print('max sm: ',maxsm,'hm for this sm: ',hm_this_sm,'z for this sm: ',z_this_sm)

    if plot_figure:
        #pl.figure(figsize=(8,6))
        if choice=='B13':
            ax = pl.gca()
            plot_b13_satgen(ax)

        pl.xlabel('$\mathrm{\log_{10}\, halo \,mass\,(M_{\odot})}$',fontsize=15)
        pl.ylabel('$\mathrm{\log_{10}\, stellar(disk) \,mass\,(M_{\odot})}$',fontsize=15)
        if hmrange is None:
            pl.ylim(9,13)
        else:
            pl.xlim(min(lghmrange),max(lghmrange))

    return

def check_sm_growth_logic(n_repeat=10,sigma=1):
    """
    Given two slop, one is steep and one is shallow, check if shallow slop makes 
    sm growth more easily to stay outside one sigma once fell outside one sigma.

    Maybe a larger time step also has an effect? (maybe not) 
    """

    x = np.arange(0, 20, 0.5)
    
    pl.figure(figsize=(8, 6))
    
    for rep in range(n_repeat):
        prevy = 0
        y, ymax, ymin = [], [], []
        ymax1, ymin1 = [], []

        for xx in x:
            if xx <= 10:
                slope = 1
                yy = np.random.normal(xx * slope, 0.5)
                ymax.append(xx * slope + 0.5)
                ymin.append(xx * slope - 0.5)
                ymax1.append(xx * slope + 0.5*sigma)
                ymin1.append(xx * slope - 0.5*sigma)
            else:
                slope = 0.2
                yy = np.random.normal(xx * slope + 10*(1-slope), 0.5)
                ymax.append(xx * slope + 10*(1-slope) +0.5)
                ymin.append(xx * slope + 10*(1-slope) -0.5)
                ymax1.append(xx * slope + 10*(1-slope) + 0.5*sigma)
                ymin1.append(xx * slope + 10*(1-slope) - 0.5*sigma)
            if yy > prevy:
                y.append(yy)
                prevy = yy
            else:
                y.append(prevy)
    
        # Plot this realization
        pl.scatter(x, y, s=8, zorder=2, alpha=0.6, label=f"Run {rep+1}" if rep==0 else None)
    
    # Grey shaded region showing ±1σ band
    ax = pl.gca()
    ax.fill_between(x, ymax, ymin, alpha=0.4, facecolor='grey', zorder=1)
    ax.fill_between(x, ymax1, ymin1, alpha=0.3, facecolor='cyan', zorder=1)
    return
#########################################################
