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

    
import numpy as np
import os
import time

import hmf
from hmf import MassFunction

import astropy.cosmology as cosmo

from astropy.table import Table
from functools import partial
import tables as tb
from scipy.interpolate import interp1d

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

def read_tree():
    """
    pchtrees output files
    """

    # Millennium
    hubble_parameter = 0.73
    OMEGA_B = 0.04455

    cosmology = cosmo.FlatLambdaCDM(hubble_parameter*100,0.25)
    tree_file = '/data/chungwen/sgarrak/pchtrees/runs/1000_mixed_logmass/output_satgen_1000_mixed_logmass.hdf5'

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

    Data sets: fddisk: Use the fd=0.1 to get the disk mass
               nodisk: No disk
               interpdisk: Use the z=0 B13 SMHM relation to get the disk mass,
                           and rescale to the earlier halo mass
    
    ver: lessoutput: Contains only initial and final outputs
         ver2: 
         ver3: Adds main branch infomation

    Notes: host_disk_mass arrays not fixed in ver2 and ver3. 
           The disk masses of hosts are recorded in MainBranches instead.

    TODO: Is there an way to control which file will load first 
          when using os.listdir or glob.glob?
    """

    dir_path = '/data/chungwen/sgarrak/runs/1000_mixed_logmass/'
    data = dict()
    if ver==0:
        data_path = dir_path+'lessoutput/'
        
        output_dataset_names = ['fd01_disk','fd002_disk','interp_disk','no_disk']
        satgen_dataset_names = ['initial_mass','initial_mstar','final_mass',
                                'final_mstar','final_radius','final_status','tree_idx']

        for odn in output_dataset_names:
            data[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_{}_lessoutput.hdf5'.format(odn),
                                  satgen_dataset_names,group='/Progenitors')
    
    elif ver==1:
        data_path = dir_path+'reionoutput/'

        output_dataset_names = ['fd01','no','step_forward','step_backward']#,'interp_sm','interp']
        satgen_dataset_names = ['coors','has_galaxy','itree','levels_at_tsteps','nprog',
                                'prog_masses','prog_mstars','radii','status','t_proc','tage',
                                'tree_idx','tsteps']

        for odn in output_dataset_names:
            print('Building data: ',odn)
            data[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_{}_disk.hdf5'.format(odn),
                                  satgen_dataset_names,group='/Progenitors')

    elif ver==2:
        data_path = dir_path+'reionoutput/evolve/'

        output_dataset_names = ['fd01','no','step_forward','step_forward_shift15','step_forward_shift05']#,'interp_sm','interp']
        satgen_dataset_names = ['coors','has_galaxy','itree','levels_at_tsteps','nprog',
                                'prog_masses','prog_mstars','radii','status','t_proc','tage',
                                'tree_idx','tsteps']

        for odn in output_dataset_names:
            print('Building data: ',odn)
            data[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_{}_disk.hdf5'.format(odn),
                                  satgen_dataset_names,group='/Progenitors')

    elif ver==3:
        data_path = dir_path+'reionoutput/evolve_ver2/'

        output_dataset_names = ['no','fd01','step_backward','step_forward','step_forward_threshold_off','step_forward_threshold_off_z0smhm_on']
        prog_dataset_names = ['coors','has_galaxy','itree','levels_at_tsteps','nprog',
                                'prog_masses','prog_mstars','radii','status','t_proc','tage',
                                'tree_idx','tsteps']
        main_dataset_names = ['main_branch_disk_mass']
        main = dict()
        for odn in output_dataset_names:
            print('Building data: ',odn)
            data[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_{}_disk.hdf5'.format(odn),
                                  prog_dataset_names,group='/Progenitors')
            if odn != 'no':
                main[odn] = read_hdf5(data_path+'prog_evo_1000_mixed_logmass_{}_disk.hdf5'.format(odn),
                                      main_dataset_names,group='/MainBranches')

    elif ver==-1:
        data_path = dir_path

        prog_dataset_names = ['levels_at_tsteps','host_disk_masses']
        main_dataset_names = ['main_branch_disk_mass']

        data = read_hdf5(data_path+'test.hdf5',prog_dataset_names,group='/Progenitors')
        main = read_hdf5(data_path+'test.hdf5',main_dataset_names,group='/MainBranches')
        
    if ver==3 or ver==-1:
        return data,main
    else:
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
    tree_idx = prog['tree_idx']
    has_galaxy = prog['has_galaxy'] == 1

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

def mod_Mstar(hm,h=0.73,z=0.,choice='B13',task='Mstar'):
    """
    Customized Mstar 
    The halo mass from our eps tree already dealt with h
    
    B13 assumed parameters: Omega_M = 0.27
                            Omega_lambda = 0.73
                            h = 0.7
                            ns = 0.95
                            sigma8 = 0.82

    B19 assumed parameters: h = 0.678

    Parameters: h: Hubble parameter assumed in the merger tree
                hm: linear halo mass

    Return: linear stellar mass

    """

    if choice=='B13':
        h_model = 0.7
        hm *= (h/h_model)
        if task=='Mstar':
            sm = init.Mstar(hm,z,choice=choice)
        elif task=='lgMs_B13':
            sm = 10**gh.lgMs_B13(np.log10(hm),z)
        else:
            raise ValueError(f"Invalid task '{task}' for choice 'B13'")
        
    elif choice=='RP17':
        h_model = 0.7 # Not used, needs to check
        hm *= (h/h_model)
        if task=='Mstar':
            sm = init.Mstar(hm,z,choice=choice)
        elif task=='lgMs_RP17':
            sm = 10**gh.lgMs_RP17(np.log10(hm),z)
        else:
            raise ValueError(f"Invalid task '{task}' for choice 'RP17'")

    elif choice=='B19':
        h_model = 0.678
        hm *= (h/h_model)
        raise NotImplementedError('B19: Not provide now')
        
    sm *= (h_model/h)
    return sm


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
        smlist = [mod_Mstar(hm,z=z, choice='B13',task='Mstar') for z in Nz]
        avgsm.append(np.median(smlist))
        maxsm.append(max(smlist))
        minsm.append(min(smlist))

    f = interp1d(hmrange,avgsm)

    if return_edge:
        return np.array(maxsm),np.array(minsm),hmrange

    return f


# ChatGPT
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
            sm = np.log10([mod_Mstar(hmr,z=0,task='lgMs_B13') for zr in zrange])
        else:
            sm = np.log10([mod_Mstar(hmr,z=zr,task='lgMs_B13') for zr in zrange])
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

def plot_macc_mstream_funtion(progs,root_mass,task='m_acc'):
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

    pl.figure(figsize=(22,10))
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
    pl.ylabel('$\log_{10}\,dN/dM_\mathrm{\star,acc}$',fontsize=20)
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
        coors = prog['coors'][pk]
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

def plot_EMERGE(itree,hm,z,cosmology,choice='history'):

    b13sm = np.array([gh.lgMs_B13(np.log10(hm_tree),z) for hm_tree in hm])
    b13sm50 = np.percentile(b13sm,50,axis=0)
    b13sm1sigma = np.percentile(b13sm,84.1,axis=0)
    b13sm_err = b13sm1sigma-b13sm50

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
    
    print('init redshift check: ',z[:,-1])
    print('init disk mass check: ',dm[:,-1])     
    if choice=='history':
        pl.plot(z.T,dm.T,alpha=0.2,c='b')
        pl.plot(z.T,hm.T,alpha=0.2,c='k')
        pl.xlabel('z')
    elif choice=='smhm':
        pl.plot(hm.T,dm.T)
        ax = pl.gca()
        plot_b13_satgen(ax)
        pl.ylabel('$\mathrm{log_{10}\,halo\,mass}$')

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

def debug_smhm_range(nrepeat):
    """
    """

    zrange  = np.arange(0,24,0.2)
    hmrange = 10**np.arange(9,12.7,0.2)    

    pl.figure(figsize=(8,6))

    for hm in hmrange:
        for _ in np.arange(nrepeat):
            sm = np.array([mod_Mstar(hm,z=z,choice='B13',task='Mstar') for z in zrange])
            pl.scatter(np.repeat(np.log10(hm),len(sm)),np.log10(sm),s=2)
    ax = pl.gca()
    plot_b13_satgen(ax)

    pl.xlabel('$\mathrm{\log_{10}\, halo \,mass\,(M_{\odot})}$',fontsize=15)
    pl.ylabel('$\mathrm{\log_{10}\, stellar(disk) \,mass\,(M_{\odot})}$',fontsize=15)
    pl.xlim(9,12.5)
    pl.ylim(5,11.5)

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
