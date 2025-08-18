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
    
import numpy as np
import os
import time

import hmf
from hmf import MassFunction

import astropy.cosmology as cosmo

from astropy.table import Table

import tables as tb

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

    return tree_redshifts,root_mass,progenitors


def read_satgen():
    """
    Satgen output files

    Data sets: fddisk: Use the fd=0.1 to get the disk mass
               nodisk: No disk
               interpdisk: Use the z=0 B13 SMHM relation to get the disk mass,
                           and rescale to the earlier halo mass
    """
    fn1 = '/data/chungwen/sgarrak/runs/1000_mixed_logmass/prog_evo_1000_mixed_logmass_lessoutput.hdf5'
    fn2 = '/data/chungwen/sgarrak/runs/1000_mixed_logmass/prog_evo_1000_mixed_logmass_no_disk_lessoutput.hdf5'
    fn3 = '/data/chungwen/sgarrak/runs/1000_mixed_logmass/prog_evo_1000_mixed_logmass_interp_disk_lessoutput.hdf5'
    fn4 = '/data/chungwen/sgarrak/runs/1000_mixed_logmass/prog_evo_1000_mixed_logmass_fd002_disk_lessoutput.hdf5'    

    satgen_dataset_names = ['initial_mass','initial_mstar','final_mass','final_mstar','final_radius','final_status','tree_idx']
    
    progenitor_fd01disk  = read_hdf5(fn1, satgen_dataset_names, group='/Progenitors')
    progenitor_nodisk = read_hdf5(fn2, satgen_dataset_names, group='/Progenitors')
    progenitor_interpdisk  = read_hdf5(fn3, satgen_dataset_names, group='/Progenitors')
    progenitor_fd002disk  = read_hdf5(fn4, satgen_dataset_names, group='/Progenitors')
    
    return progenitor_fd01disk,progenitor_nodisk,progenitor_interpdisk,progenitor_fd002disk

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

def compute_mass(prog,root_mass,task='m_acc',ntrees=1000):
    """
    Compute different mass accretion type
    
    Parameters: prog: progenitor_wdisk: progenitor dict with a disk potential
                      progenitor_wodisk: without a disk potential
                task: m_acc: total destroyed satellite mstar
                      m_stream: stripped mstar, removed the intact part
                      m_total: total satellite mstar
                      
    """
    
    survives = prog['final_status'] == 0
    tree_idx = prog['tree_idx']
    
    mass_arr = np.zeros(ntrees)

    for itree in range(0,ntrees):
        this_tree    = tree_idx == itree
        imax = np.argmax(prog['initial_mass'][this_tree])
        
        if task=='m_acc':
            mass_arr[itree] = np.sum(prog['initial_mstar'][this_tree & (~survives)],dtype=np.float64)
            
        if task=='m_stream':
            mass_arr[itree] = np.sum(prog['initial_mstar'][this_tree & (survives)] - prog['final_mstar'][this_tree & (survives)],dtype=np.float64)
            
        if task=='m_total':
            mass_arr[itree] = np.sum(prog['initial_mstar'][this_tree],dtype=np.float64)       
        
        if task=='hm_smooth':
            mass_arr[itree] = root_mass[itree]-np.sum(prog['initial_mass'][this_tree])

        if task=='hm_acc':
            mass_arr[itree] = np.sum(prog['initial_mass'][this_tree & (survives)] - prog['final_mass'][this_tree & (survives)],dtype=np.float64)
            
        if task=='max_prog':
            mass_arr[itree] = prog['initial_mass'][this_tree][imax]
            
        if task=='hm_total':
            mass_arr[itree] = np.sum(prog['initial_mass'][this_tree],dtype=np.float64)
        
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

    w = (y < 0) & (x > 10) #& (np.abs(dyp) < 1) & (np.abs(dym) < 1)
    
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

def plot_m18(ax,h=0.6781,edges=False,omega_0=0.25,**kwargs):
    """
    Moster (2018)
    """
    m18_h = 0.6781
    hubble_factor = np.log10(m18_h/h)
        
    lmhalo = np.arange(10,15,0.1)
    
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

def plot_subhalo_massfunction(progs,root_mass,radius='total'):
    """
    Accumulated subhalo mass function

    Parameters: progs: All the progentiors from satgen
                root_mass: The root halo mass of ntrees
                radius: inner (<50kpc) or total
    """

    models = ['fd01','no','interp','fd002']
    
    pl.figure(figsize=(8,6))
    for i,p in enumerate(progs):
        bins = np.arange(10,12,0.1)
        
        if radius=='inner':
            inner_region = p['final_radius'] <50
            p_inner = {k:v[inner_region] for k,v in p.items()}
            hm_acc = compute_mass(p_inner,root_mass,task='hm_acc')
            pl.title('$\mathrm{<50kpc}$',fontsize=20)
        else:
            hm_acc = compute_mass(p,root_mass,task='hm_acc')
            pl.title('total',fontsize=20)
        h,_ = np.histogram(np.log10(hm_acc),bins)
        reverse_h = h[::-1]
        accumulate_h = np.cumsum(reverse_h)
        
        pl.plot(bins[:-1],np.log10(accumulate_h[::-1]),label=models[i])

    pl.legend(frameon=False,prop={'size':15})
    pl.xlabel('$\mathrm{\log_{10}\,halo\,mass}$',fontsize=20)
    pl.ylabel('$\mathrm{\log_{10}\, N(>halo\,mass)}$',fontsize=20)
    larger_ticks()
    return

def plot_macc_function(progs,root_mass):
    """
    Volume weighted accreted stellar mass function
    """
    weights = volume_weight(root_mass)
    models = ['fd01','no','interp','fd002']
    bins = np.arange(5,13,0.2)

    pl.figure(figsize=(7,7))
    for i,p in enumerate(progs):
        m_acc = compute_mass(p,root_mass,task='m_acc')
        h,_ = np.histogram(np.log10(m_acc),bins,weights=weights,density=True)
        pl.plot(bins[:-1],np.log10(h),label=models[i],drawstyle='steps-post')
    
    pl.legend(frameon=False,prop={'size':15})
    pl.xlabel('$\log_{10}\,M_\mathrm{\star,acc}/M_{\odot}$',fontsize=20)
    pl.ylabel('$\log_{10}\,dN/dM_\mathrm{\star,acc}$',fontsize=20)
    larger_ticks()

    return

def plot_macc_mstream_funtion(progs,root_mass,task='m_acc'):
    """
    Volume weighted stellar stream mass function

    Parameters: task: m_acc
                      m_stream
                      m_acc_stream
    """

    models = ['fd01','no','interp','fd002']
    cs = ['r','k','b','cyan']

    # This value is the same for all models
    tree_max_prog_mass = compute_mass(progs[0],root_mass,task='max_prog')
    tree_max_prog_mass_frac = tree_max_prog_mass/root_mass
    s = tree_max_prog_mass_frac < 0.2    

    weights = volume_weight(root_mass)
    bins=np.arange(5,13,0.2)

    pl.figure(figsize=(18,7))
    pl.subplot(121)
    for i,p in enumerate(progs):
        m_acc = compute_mass(p,root_mass,task='m_acc')
        m_stream = compute_mass(p,root_mass,task='m_stream')

        ha,_ = np.histogram(np.log10(m_acc),bins,weights=weights,density=True)
        hs,_ = np.histogram(np.log10(m_stream),bins,weights=weights,density=True)

        if task=='m_acc':
            pl.plot(bins[:-1],np.log10(ha),drawstyle='steps-post',c=cs[i])
        if task=='m_stream':
            pl.plot(bins[:-1],np.log10(hs),drawstyle='steps-post',c=cs[i],ls='dashed')
        if task=='m_acc_stream':
            pl.plot(bins[:-1],np.log10(ha),drawstyle='steps-post',c=cs[i])
            pl.plot(bins[:-1],np.log10(hs),drawstyle='steps-post',c=cs[i],ls='dashed')

    pl.xlim(7,11.5)
    pl.xlabel('$\log_{10}\,M_\mathrm{\star,acc}/M_{\odot}$',fontsize=20)
    pl.ylabel('$\log_{10}\,dN/dM_\mathrm{\star,acc}$',fontsize=20)
    larger_ticks()

    # Selected by the less massive ones (s<0.2)
    pl.subplot(122)

    for i,p in enumerate(progs):
        m_acc = compute_mass(p,root_mass,task='m_acc')
        m_stream = compute_mass(p,root_mass,task='m_stream')

        ha,_ = np.histogram(np.log10(m_acc[s]),bins,weights=weights[s],density=True)
        hs,_ = np.histogram(np.log10(m_stream[s]),bins,weights=weights[s],density=True)

        if task=='m_acc':
            pl.plot(bins[:-1],np.log10(ha),label=models[i],drawstyle='steps-post',c=cs[i])
        if task=='m_stream':
            pl.plot(bins[:-1],np.log10(hs),label=models[i],drawstyle='steps-post',c=cs[i],ls='dashed')
        if task=='m_acc_stream':
            pl.plot(bins[:-1],np.log10(ha),label=models[i],drawstyle='steps-post',c=cs[i])
            pl.plot(bins[:-1],np.log10(hs),label=models[i],drawstyle='steps-post',c=cs[i],ls='dashed')

    legend1 = pl.legend(frameon=False,prop={'size':15},loc='upper right')
    l1 = pl.Line2D([0,0],[0,0],ls='solid',c='r')
    l2 = pl.Line2D([0,0],[0,0],ls='dashed',c='r')
    handles = [l1,l2]
    labels = ['m_acc','m_stream']
    legend2 = pl.legend(frameon=False,handles,prop={'size':15},labels,loc='lower left')
    pl.gca().add_artist(legend1)

    pl.xlim(7,11.5)
    pl.title('$\mathrm{s<0.2}$',fontsize=20)
    pl.xlabel('$\log_{10}\,M_\mathrm{\star,acc}/M_{\odot}$',fontsize=20)
    pl.ylabel('$\log_{10}\,dN/dM_\mathrm{\star,acc}$',fontsize=20)
    larger_ticks()
    
    return


#########################################################
