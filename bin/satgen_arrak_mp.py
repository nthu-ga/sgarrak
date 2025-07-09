import sys

SATGEN_PATH = '/data/chungwen/SatGen'
if not SATGEN_PATH in sys.path:
    sys.path.append(SATGEN_PATH)

SATGEN_ETC_PATH = '/data/chungwen/SatGen/etc'
if not SATGEN_ETC_PATH in sys.path:
    sys.path.append(SATGEN_ETC_PATH)
    
import numpy as np
import os
from time import time

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
from   profiles import NFW,Green
from   orbit import orbit
import galhalo as gh
import init

import argparse
import h5py

############################################################
def read_pchtrees_main_branch_mass(path):
    """
    Reads the main branch masses of pchtrees in the satgen branch.
    
    Return: 
        Halo masses of each tree with -1 representing no node in the tree at
        the corresponding redshift.
    """
    with tb.open_file(path, 'r') as f:
        data = f.get_node('/Mainbranch/MainbranchMass').read()
    return np.array(data)

############################################################
def read_pchtrees_redshifts(path):
    """
    Reads the main branch masses of pchtrees in the satgen branch.
    
    Return: 
        Halo masses of each tree with -1 representing no node in the tree at
        the corresponding redshift.
    """
    with tb.open_file(path, 'r') as f:
        data = f.get_node('/OutputTimes/Redshift').read()
    return np.array(data)

############################################################
def read_pchtrees_progenitors(path):
    """
    return: HostMass: The main branch halo mass before mergers
            ProgenitorZred: The first order minor branch redshifts that merge to the main branch
            ProgenitorMass: Same as above but mass
    """
    dataset_names = ["HostMass","ProgenitorZred","ProgenitorMass","TreeID"]
    with tb.open_file(path, 'r') as f:
        data = dict([ (name,f.get_node(f'/Progenitors/{name}').read()) for name in dataset_names])
        
    return data

############################################################
def halo_mah_to_zhao_c_nfw(tree_main_branch_masses, tree_t_age_gyr):
    """
    """
    h_c_nfw = list()
    
    nlev = tree_main_branch_masses.shape[0]
    for i in range(0,nlev):
        h_c_nfw.append(init.c2_fromMAH(tree_main_branch_masses[i:],tree_t_age_gyr[i:]))
    
    return np.array(h_c_nfw)

############################################################
def compute_orbit_radius(host_mass, prog_z, prog_mass, 
                         eps=1.0, xc=1.0,
                         orbit_init_method=None,
                         host_mass_z=None,
                         cosmology=None,
                         prog_mstar_init=None,
                         n_substeps=None,
                         evolve_prog_mass=False):
    """
    host_mass: if an array[0:N], assume ordering from present day [0] to 
               earliest time [N].
               
    orbit_init_method: 
        None     : use given eps and xc
        'li2020' : use Li et al. 2020 distributions (ZZLi in SatGen)
    """     
    if cosmology is None:
        cosmology = cosmo.Planck18
        
    evolve_host_mass = isinstance(host_mass,np.ndarray)
    
    if evolve_host_mass:
        #print('Evolving host mass')
        evolve_host_mass = True
        n_lev = len(host_mass)
        
        assert(host_mass_z is not None)
        assert(len(host_mass_z) == n_lev)
            
        # We need the ages at each redshift to compute the concentration
        # evolution with Zhao's method
        host_mass_t_age = cosmology.age(host_mass_z).value
        
        # We need the lookback times to compute the timesteps
        host_mass_t_lbk = cosmology.lookback_time(host_mass_z).value
    
    # Set up satellite progenitor
    if prog_mass is not None:
        prog_mass_init = prog_mass
        pc = init.concentration(prog_mass_init,prog_z,choice='DM14')
        
        # The halo potential is a "Green" profile; an NFW with additional
        # methods to adjust for the effects of tidal stripping.
        pp = Green(prog_mass,pc,Delta=200,z=prog_z)
        
        # Draw stellar mass from mstar-mhalo releation
        if prog_mstar_init is None:
            prog_mstar_init = init.Mstar(prog_mass,prog_z,choice='B13')
        
        # The mass within rmax is used in the stripping calculations
        prog_m_max_init = pp.M(pp.rmax)
    else:
        prog_mstar_init = None
        
    # Initial density profiles
    if evolve_host_mass:
        # Evolving halo profile
        
        if host_mass_z is not None:
            # Compute concentrations along the main branch using the 
            # Zhao formula
            hc_list = halo_mah_to_zhao_c_nfw(host_mass, host_mass_t_age)
        else:
            # Fix the concentration at the final time
            hc = init.concentration(host_mass[0],prog_z,choice='DM14')
            hc_list = np.repeat(hc, n_lev)
            
        # Make a profile for each timestep
        hp_list = list()
        for i in range(n_lev):
             hp_list.append(NFW(host_mass[i],hc_list[i],Delta=200.,z=host_mass_z[i],sf=1.))
    else:
        # Fixed halo profile
        hc = init.concentration(host_mass,prog_z,choice='DM14')
        hp = NFW(host_mass,hc,Delta=200.,z=prog_z,sf=1.) # should use z0 host concentration
    
    # Analytical orbits

    # Starting time for progenitor
    prog_release_t_lbk = cosmology.lookback_time(prog_z).value
            
    if evolve_host_mass:
        # In this case we have to start the evolution from the time of accretion
        # APC: would be better to work in terms of levels from the outset...
        host_node_idx_init = np.flatnonzero(host_mass_z >= prog_z)[0]
        
        #print(f'Host init idx = {host_node_idx_init:d}')
        #print(f'Host init z   = {host_mass_z[host_node_idx_init]:4.2f}')
        #print(f'Prog init z   = {prog_z:4.2f}')
        
        # A bit of juggling between time ordering conventions is required.
        # The input z/time grids are ordered such that [0] is z=0
        # Hence we select the range of tree times from 0 up to the 
        # required index. 
        #
        # However, the time coordinate in the integration measures time 
        # *since infall*.
        #
        # Doing everything in terms of lookback time doesn't make this
        # easier to understand :)
        #
        tstep  = (prog_release_t_lbk - host_mass_t_lbk[:host_node_idx_init+1])
        nsteps = len(tstep)
        dt     = (tstep[:-1]-tstep[1:])
        
        # Which host potential to use for each step, stored as an index into the
        # list of host potentials/concentrations. Make this a list to deal with
        # sub-stepping. +1 so that the first timestep uses the initial conditions
        # for the host.
        host_node_index_step = np.arange(host_node_idx_init+1, 0, -1, dtype=int)
        
        # We iterate forwards in time (i.e. from the smallest tstep to the largest),
        # so we need to flip the tstep and dt arrays (do this after computing dt!)
        
        # Remember that the integration time coordinate of the initial conditions is
        # tstep = 0; dt indicates the time ellapsed between the last step and the 
        # current step.
        tstep  = tstep[::-1]
        dtstep = np.concatenate([np.atleast_1d(tstep[0]),dt[::-1]])

        #print(f'Initial conditions lookback time: {host_mass_t_lbk[host_node_idx_init]:2.3f}')
        #print(f'{nsteps:d} steps')
        
        # The substeps run *up to* the current substep, i.e. the first substeps
        # run from the intial conditions up to the first tstep
        if n_substeps is not None:
            dt_substeps  = list()
            t_substeps   = list()
            idx_substeps = list()
            for istep in range(1,nsteps):
                t_this_substep  = np.linspace(tstep[istep-1],tstep[istep],n_substeps)
                dt_this_substep = t_this_substep[1]-t_this_substep[0],
                
                t_substeps.append(t_this_substep)
                dt_substeps.append(np.repeat(dt_this_substep,n_substeps))
                idx_substeps.append(np.repeat(host_node_index_step[istep],n_substeps))
            
            tstep  = np.concatenate(t_substeps)
            dtstep = np.concatenate(dt_substeps)
            host_node_index_step = np.concatenate(idx_substeps)
            nsteps = len(tstep)
            #print(f'With substeps, have {nsteps:d} timesteps')
            
        hp = hp_list[host_node_idx_init]
        hc = hc_list[host_node_idx_init]
    else:
        dtstep = np.repeat(0.01,nsteps) # Gyr
        nsteps = int(prog_release_t_lbk/dt)
        tstep  = np.linspace(0,prog_release_t_lbk,nsteps)
        #print(f'{nsteps:d} steps')
    
    # eps = 1./np.pi*np.arccos(1.-2.*np.random.random()) # From Treegen.py
    
    if orbit_init_method is None:
        # xv in cylindrical coordinates: np.array([R,phi,z,VR,Vphi,Vz])  
        xv  = init.orbit(hp, xc=xc, eps=eps) 
    elif orbit_init_method == 'li2020':
        vel_ratio, gamma = init.ZZLi2020(hp, prog_mass_init, prog_z)
        xv = init.orbit_from_Li2020(hp, vel_ratio, gamma)
    else:
        raise Exception

    r_init = np.sqrt(xv[0]**2+xv[2]**2)
    # Construct the orbit
    o = orbit(xv)
            
    # These arrays accumulate the output over timesteps
    # By convention, the output arrays store the state at the *end* of each timestep
    
    # The first entry in the output arrays is the initial conditions
    radii       = [r_init]
    prog_masses = [prog_mass_init]
    prog_mstars = [prog_mstar_init]
    
    prog_mass  = prog_mass_init
    prog_mstar = prog_mstar_init
    
    # istep = 0 corresponds to evolution up to the end of the first step (i.e. up to
    # t = 0 + dt)
    for istep in range(1,nsteps):    
        t  = tstep[istep]
        dt = dtstep[istep]
        host_node_index = host_node_index_step[istep]
        hp = hp_list[host_node_index]
        
        #print(f'Host init idx = {host_node_index:d}')
        #print(f'Host init z   = {host_mass_z[host_node_index]:4.2f}')
        
        # Check resolution to save time
        if (prog_mass < cfg.Mres) or ((prog_mass/prog_mstar_init) < cfg.psi_res) or (radii[-1] < cfg.Rres):
            radii.append(radii[-1])
            prog_masses.append(prog_masses[-1])
            prog_mstars.append(prog_mstars[-1])
            continue
            
        # Evolve the progenitor orbit based on the current mass
        # and host halo profile (i.e. at the start of the step)
        o.integrate(t,hp,prog_mass)

        # Note that the coordinates are updated 
        # internally in the orbit instance "o" when calling
        # the ".integrate" method, here we assign them to 
        # a new variable "xv" only for bookkeeping
        xv  = o.xv 
        r   = np.sqrt(xv[0]**2+xv[2]**2)
        radii.append(r)

        # Section 2.3 of Green 2021 describes a procedure to evolve the density profile of the subhalo
        # based on multiplying initial density profile at accretion with a "transfer function". 
        
        # The transfer function is H(l|f_bound, c_acc) where f_bound is mass still bound to the subhalo,
        # and c_acc is the concentration of the subhalo at accretion. The variable "l" is radius in the
        # subhalo in the nomenclature of SatGen.
        
        if evolve_prog_mass:
            # Evolve the progenitor mass for dt in the current potential
            # Following SatGen (SatEvo), msub takes the initial potentials
            # and orbit at the start of the step.
            # dt is the length of the step (right? APC)
            alpha_strip = ev.alpha_from_c2(hc,pc)
            prog_evolved_mass, prog_tidal_raidus = ev.msub(pp,hp,xv,dt,
                                                           choice='King62',
                                                           alpha=alpha_strip)
            
            # Now update the potential of the satellite to the end of the step, after
            # mass loss. This update function claims to handle the resolution limit.
            pp.update_mass(prog_evolved_mass)

            # Evolve baryonic properties
            
            # This is done in terms of the ratio of mass within r_max *now* to the
            # mass within r_max *at infall*.
            prog_m_max = pp.M(pp.rmax)
            # Alpha and leff/lmax here are a little subtle...
            g_le, g_ms = ev.g_EPW18(prog_m_max/prog_m_max_init, 
                                    alpha=1., lefflmax=0.1) 
            
            # APC: g_le and g_ms are arrays, don't know why...
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
            
    prog_masses = np.array(prog_masses)   
    prog_mstars = np.array(prog_mstars)        
    radii       = np.array(radii)
    
    return tstep, radii, o, prog_masses, prog_mstars

############################################################
def process_tree(itree,
                 progenitors=None,
                 tree_main_branch_masses=None,
                 tree_redshifts=None,
                 cosmology=None,
                 n_substeps=None):
    """
    """
    from time import sleep
    
    sleep(3)
    t_start = time()
    
    # SATGEN.cfg implicitly sets resolution limits on mass (absolute and relative) and radius
    # Set them explicitly here; these are the degaults
    cfg.Mres    = 100.0
    cfg.Rres    = 0.001
    cfg.psi_res = 1.0e-5

    np.random.seed(42)
    
    progs_this_tree   = np.flatnonzero(progenitors['TreeID'] == itree)
    nprogs_this_tree  = len(progs_this_tree)
    host_mass_history = tree_main_branch_masses[itree]
    
    results = dict()
    
    results['initial_mass']  = list()
    results['initial_mstar'] = list()
    
    results['final_mass']   = list()
    results['final_mstar']  = list()
    results['final_radius'] = list()
    
    for iprog in range(0,nprogs_this_tree):
        prog_z    = progenitors['ProgenitorZred'][progs_this_tree][iprog]
        prog_mass = progenitors['ProgenitorMass'][progs_this_tree][iprog]
        t, r, o, pm, pmstar = compute_orbit_radius(host_mass_history,prog_z,prog_mass,
                                                   host_mass_z = tree_redshifts,
                                                   orbit_init_method = 'li2020',
                                                   cosmology = cosmology,
                                                   n_substeps = n_substeps,
                                                   evolve_prog_mass = True)
        
        results['initial_mass'].append(pm[0])
        results['final_mass'].append(pm[-1])
        
        results['initial_mstar'].append(pmstar[0])
        results['final_mstar'].append(pmstar[-1])
        results['final_radius'].append(r[-1])

    for k in results.keys():
        results[k] = np.array(results[k])
        
    results['nprog'] = len(results['initial_mass'])
    
    t_end = time()
    results['t_proc'] = t_end - t_start
    results['itree']  = itree
    return results

############################################################
def write_results(results, filename):
    """
    """
    ntrees    = len(results)
    data_keys = results[0].keys()

    total_results = dict()
    for k in data_keys:
        total_results[k] = list()
    total_results['tree_idx'] = list()

    for itree in range(0,ntrees):
        for k in data_keys:
            total_results[k].append(np.atleast_1d(results[itree][k]))
        total_results['tree_idx'].append(np.repeat(results[itree]['itree'],results[itree]['nprog']))

    for k in total_results.keys():
        print(k)
        total_results[k] = np.concatenate(total_results[k])

    total_nprog = len(total_results['tree_idx'])

    with h5py.File(filename, "w") as f:
        f["/"].create_group('Progenitors')
        for k, v in total_results.items():
            f["/Progenitors"].create_dataset(k, data=v, compression=6) 
            
    print('Wrote {:s}'.format(filename))
    return

############################################################
def main(args,client=None):
    """
    """
    import multiprocessing
    from multiprocessing import Pool
    from functools import partial
    from time import sleep
    
    multiprocessing.set_start_method('fork')
    
    if 'SLURM_CPUS_ON_NODE' in os.environ:
        ncpus = int(os.environ['SLURM_CPUS_ON_NODE'])
    else:
        ncpus = 1 
    print('Available cores: {:d}'.format(ncpus))
    
    sleep(5)
    
    # Millennium
    hubble_parameter = 0.73
    cosmology = cosmo.FlatLambdaCDM(hubble_parameter*100,0.25)

    print('Cosmology:', cosmology)
    
    tree_file = args.tree_file
    print('Reading {:s}'.format(tree_file))
        
    tree_main_branch_masses = read_pchtrees_main_branch_mass(tree_file)
    tree_main_branch_masses = tree_main_branch_masses/hubble_parameter

    ntrees, nlev = tree_main_branch_masses.shape

    print('{:d} trees, {:d} levels'.format(ntrees,nlev))
    
    progenitors = read_pchtrees_progenitors(tree_file)
    progenitors['ProgenitorMass'] = progenitors['ProgenitorMass']/hubble_parameter
    progenitors['HostMass']       = progenitors['HostMass']/hubble_parameter

    tree_redshifts = read_pchtrees_redshifts(tree_file)
    tree_t_lbk_gyr = cosmology.lookback_time(tree_redshifts).value
    tree_t_age_gyr = cosmology.age(tree_redshifts).value
        
    partial_process_tree = partial(process_tree, 
                                   n_substeps = args.substeps,
                                   progenitors=progenitors,
                                   tree_main_branch_masses=tree_main_branch_masses,
                                   tree_redshifts=tree_redshifts,
                                   cosmology=cosmology)    
    print('Processing...')
    t_start = time()
    
    pool      = Pool(processes=ncpus)
    results   = list()
    chunksize = 2
    NMAX      = ntrees
    print('Running {:d} trees'.format(NMAX))
    print("{:10s} | {:10s} | {:6s}".format("IDX", "ITREE", "TIME"))
    for i, _ in enumerate(pool.imap_unordered(partial_process_tree, range(NMAX), chunksize)):
        print("{:10d} | {:10d} | {:6.2f}s".format(i, _['itree'], _['t_proc']))
        sys.stdout.flush()
        results.append(_)

    print('Total time: {:g}'.format(time() - t_start))
    
    # results = pool.map(partial_process_tree, range(0,8))
      
    #for itree in range(0,ntrees):
    #    print(itree)
    #    R = process_tree(itree,
    #                progenitors=progenitors,
    #                tree_main_branch_masses=tree_main_branch_masses,
    #                tree_redshifts=tree_redshifts,
    #                cosmology=cosmology)
    #    results.append(R)
     
    write_results(results, args.output)
    print('Done!')
          
############################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Satgen for Arrakihs")
    parser.add_argument("tree_file", help="Input PCHTrees file (PFOP run)")
    parser.add_argument("--ncores","-n", help="Number of cores", default=1, type=int)
    parser.add_argument("--substeps","-s", help="Number of substeps", default=None, type=int)
    parser.add_argument("--output","-o", help="Output filename", default='all_progenitors.hdf5')
    return parser.parse_args()

############################################################
if __name__ == '__main__':
    args = parse_args()
    main(args)
