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

def post_integrate_stripped_time_varying_point_mass(o,host_dp,levels_at_tstep,tstep,tage,m,sm):
    """
    The orbit of stripped particles is only the Satgen o.integrate without the dynamical friction?
    But without a prog density profile and tidal stripping.

    Return: a list of arrays of coordinates

    Note: Maybe the satgen orbit integration consider more parameters.
          The host halo potential does not instantanious add up the stripped mass.

    """

    dsm = sm[:-1] - sm[1:]
    dm  = m[:-1] - m[1:]

    # no dynamical friction
    cfg.lnL_type = 0.
    cfg.lnL_pref = 0.

    # record the coordinates of stripped mass.
    strip_star_coors_list = []
    strip_halo_coors_list = []
    strip_tage_list = []
    strip_mstar = []
    strip_mhalo = []
    strip_index = [] # indices that passes the cut

    nsteps = len(tstep)
    # mass loss at each step
    for i_strip in range(1,nsteps):
        # Only integrate dm more than 10^5 to speed up.
        # dsm is the difference, so the index is n-1.
        if dsm[i_strip-1]>0:#10**5:

            o_star_i = copy.deepcopy(o[i_strip])
            o_halo_i = copy.deepcopy(o[i_strip])
            # tage index uses i_strip, so the total time is 
            # infall time + tage[istep].
            tage_at_strip = tage[0]
            dm_i = dm[i_strip-1]
            dsm_i = dsm[i_strip-1]
            strip_mstar.append(dsm_i)
            strip_mhalo.append(dm_i)
            strip_index.append(i_strip)

            # give a new variable for this stripped mass
            istep = i_strip

            strip_star_coors = []
            strip_halo_coors = []
            strip_tage = []
            while istep <= nsteps-1:
                start_step_level = levels_at_tstep[istep] -1
                host_dp_istep = host_dp[start_step_level]

                o_star_i.integrate(tstep[istep], host_dp_istep, dsm_i)
                o_halo_i.integrate(tstep[istep], host_dp_istep, dm_i)

                xv_star  = o_star_i.xv
                xv_halo  = o_halo_i.xv

                strip_star_coors.append(compute_coordinates(xv_star))
                strip_halo_coors.append(compute_coordinates(xv_halo))
                strip_tage.append(tage_at_strip + tstep[istep])

                istep = istep+1
            strip_star_coors_list.append(strip_star_coors)
            strip_halo_coors_list.append(strip_halo_coors)
            strip_tage_list.append(strip_tage)


    return strip_star_coors_list,strip_halo_coors_list,strip_tage_list,strip_mstar,strip_mhalo,strip_index

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

######################################################
### Density profile

def stellar_halo_contributer_frac(prog_mstars,mass_contribution_cut,cut_method='fraction',both_method='atleast'):
    """
    Define a cut for significant stellar halo contributers.

    Parameters
    ----------
    both_method: 'atleast': If the mass threshold excludes all 90% contributers of a tree, it will at least
                            return top 90% indices.
    Return
    ------
    The indices of those contributers
    """
    prog_mstars = np.array(prog_mstars)
    # dumped mass per progenitor
    sm_dump = prog_mstars[:, 0] - prog_mstars[:, -1]
    cut_values = np.atleast_1d(mass_contribution_cut).astype(float)
    ncut = len(cut_values)

    if cut_method=='fraction':
        if ncut!=1:
            raise ValueError("This method takes only one cut value")

        frac_cut = cut_values[0]

        # sort by contribution (largest first)
        idx_sort = np.argsort(sm_dump)[::-1]
        sm_sorted = sm_dump[idx_sort]

        total = np.sum(sm_dump)
        if total <= 0:
            return np.array([], dtype=int)

        cum_frac = np.cumsum(sm_sorted) / total
        cut = np.searchsorted(cum_frac, frac_cut)

        top_indices = idx_sort[:cut + 1]

    elif cut_method=='mass':
        if ncut!=1:
            raise ValueError("This method takes only one cut value")
        mass_cut = cut_values[0]
        top_indices = np.where(sm_dump>=mass_cut)[0]
    elif cut_method=='both':
        if ncut != 2:
            raise ValueError("cut_method='both' takes exactly two cut values: [frac_cut, mass_cut]")

        frac_cut, mass_cut = cut_values

        idx_sort = np.argsort(sm_dump)[::-1]
        sm_sorted = sm_dump[idx_sort]

        total = np.sum(sm_dump)
        if total <= 0:
            return np.array([], dtype=int)

        cum_frac = np.cumsum(sm_sorted) / total
        cut = np.searchsorted(cum_frac, frac_cut)

        top_indices_frac = idx_sort[:cut + 1]

        # preserve mass-sorted order
        if both_method=='atleast':
            if len(sm_dump[sm_dump >= mass_cut])>len(top_indices_frac):
                top_indices = np.where(sm_dump>=mass_cut)[0]
            else:
                top_indices = top_indices_frac
        else:
            top_indices = top_indices_frac[sm_dump[top_indices_frac] >= mass_cut]

    else:
        raise ValueError(f"This cut method '{cut_method}' not support.")


    print('Indices of {} stellar halo mass contributers: ',top_indices,' ,mass donates: ',sm_dump[top_indices])
    return top_indices

def store_strip_info(orbit_object,host_dp,level_at_tsteps,tsteps,tage,
                     prog_masses,prog_mstars,mass_contribution_cut,cut_method,itree):
    """
    strip_start_index is the starting index of each dm of a progenitor.
    """
    top_indices = stellar_halo_contributer_frac(prog_mstars,mass_contribution_cut,cut_method=cut_method)

    # for "atleast" method, there always have top_indices.
    get_stream = {}
    if len(top_indices)!=0:
        print('number of matched contributers: ',len(top_indices))
        get_stream['itree'] = np.asarray([itree], dtype=np.int64)
    else:
        print('no matched contributers')
        get_stream['itree'] = np.empty(0, dtype=np.int64)
    get_stream['top_iprog']  = top_indices
    # The store coordinates and time have N(>cut)xN_each_strip(dm>0)xtsteps
    get_stream['star_coor'] = []
    get_stream['dm_coor'] = []
    get_stream['time'] = []
    get_stream['strip_start_index'] = []
    get_stream['strip_each_length'] = []
    # The masses have the shape N(>cut)xN_each_strip(dm>0)
    get_stream['dmstar'] = []
    get_stream['dmhalo'] = []
    get_stream['sindex'] = []
    get_stream['prog_each_length'] = []

    for i in top_indices:
        print('Processing the {} prog: '.format(i))
        star_coor,dm_coor,time,dmstar,dmhalo,sindex = post_integrate_stripped_time_varying_point_mass(orbit_object[i],host_dp[i],level_at_tsteps[i],tsteps[i],
                                                                                                    tage[i],prog_masses[i],prog_mstars[i])

        # append an array
        get_stream['dmstar'].append(dmstar)
        get_stream['dmhalo'].append(dmhalo)
        get_stream['sindex'].append(sindex)
        # append an inhomogenious list of arrays
        get_stream['star_coor'].append(np.concatenate(star_coor))
        get_stream['dm_coor'].append(np.concatenate(dm_coor))
        get_stream['time'].append(np.concatenate(time))

        # This is for dm
        get_stream['prog_each_length'].append(len(dmstar))
        # This is for coor
        strip_each_length = np.array([len(each_star_coor) for each_star_coor in star_coor])
        get_stream['strip_each_length'].append(strip_each_length)
        get_stream['strip_start_index'].append(np.concatenate(([0], np.cumsum(strip_each_length)[:-1])))

    # Convert per-progenitor integer fields
    get_stream['top_iprog'] = np.asarray(top_indices, dtype=np.int64)
    get_stream['prog_each_length'] = np.asarray(get_stream['prog_each_length'], dtype=np.int64)

    if len(get_stream['prog_each_length']) > 0:
        get_stream['prog_start_index'] = np.concatenate(
            ([0], np.cumsum(get_stream['prog_each_length'])[:-1])
        ).astype(np.int64)
    else:
        get_stream['prog_start_index'] = np.empty(0, dtype=np.int64)

    int_keys = {
        'sindex',
        'strip_start_index',
        'strip_each_length',
    }

    # Concatenate list-valued fields
    for k, v in get_stream.items():
        # These keys are 1-D arrays or integers, no need to concatenate.
        if k not in ('itree', 'top_iprog', 'prog_each_length', 'prog_start_index'):
            if len(v) > 0:
                get_stream[k] = np.concatenate(v)
                if k in int_keys:
                    get_stream[k] = get_stream[k].astype(np.int64)
            else:
                dtype = np.int64 if k in int_keys else np.float64
                if k in ('star_coor', 'dm_coor'):
                    get_stream[k] = np.empty((0, 3), dtype=dtype)
                else:
                    get_stream[k] = np.empty(0, dtype=dtype)

    return get_stream

# Rres default is 0.001 kpc
def mass_distribution(star_coor,dmstar):
    """
    Distribute the stripped mass on the integrated orbit to get·
    the density profile.
    The amount of mass that deposit on the orbit is proportional·
    to the time it spend at the location.

    Parameters: mass_diff: the array of stripped mass at each step(already dm>0)
                star_coor: the coordinates of each stripped mass at each step(aleady dm>0)

    Return: A list of arrays that assigns equal masses at all the positions
    """

    equal_dm = []
    for i in np.arange(0,len(dmstar)):

        # equal mass at each saved coordinates
        dm = dmstar[i]/len(star_coor[i])
        equal_dm.append(np.repeat(dm,len(star_coor[i])))

    return np.concatenate(equal_dm)

def surface_density_profile(star_coor_list,dmstar_list,sindex_list, r_bins, axis='xy'):
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

    mass_in_bin_each = []
    for i in range(nprog):
        star_coor = star_coor_list[i]
        dmstar    = dmstar_list[i]
        sindex    = sindex_list[i]
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
    print(sum(mass_in_bin_tot))
    # Area of each annulus
    area = np.pi * ((10**r_bins[1:])**2 - (10**r_bins[:-1])**2)

    # Surface density
    Sigma = mass_in_bin_tot / area
    Sigma_each = mass_in_bin_each / area
    print(Sigma,sum(Sigma))
    # Midpoint radius
    R_mid = 0.5 * (r_bins[1:] + r_bins[:-1])

    return R_mid, Sigma, Sigma_each

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







