import sys
import os
from time import time

import numpy as np
import tables as tb
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import astropy.cosmology as cosmo

from astropy.table import Table

import copy

SATGEN_PATH = '/data/chungwen/SatGen'
if not SATGEN_PATH in sys.path:
    sys.path.append(SATGEN_PATH)

SATGEN_ETC_PATH = '/data/chungwen/SatGen/etc'
if not SATGEN_ETC_PATH in sys.path:
    sys.path.append(SATGEN_ETC_PATH)

# SatGen Imports
import config as cfg
import cosmo as co
import evolve as ev
import profiles as pf
from   profiles import NFW,Dekel,MN,Einasto,Green
from   orbit import orbit
import galhalo as gh
import aux
import init

PY_PATH = '/data/chungwen/sgarrak/py/sgarrak'
if not PY_PATH in sys.path:
    sys.path.append(PY_PATH)

import sgarrak as sga

import argparse
import h5py

import matplotlib.pyplot as pl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'

###########################################################
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

    total_results['tree_idx'] = np.concatenate(total_results['tree_idx'])

    total_nprog = len(total_results['tree_idx'])

    with h5py.File(filename, "w") as f:
        f["/"].create_group('Progenitors')
        for k, v in total_results.items():
            f["/Progenitors"].create_dataset(k, data=v, compression=6) 
            
    print('Wrote {:s}'.format(filename))
    return
###########################################################
def process_tree(itree ,fd=0.1 ,flattening=25.,
                 output_zred=None,
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
    
    nsteps = 200

    np.random.seed(42)
    
    progs_this_tree   = np.flatnonzero(progenitors['TreeID'] == itree)
    nprogs_this_tree  = len(progs_this_tree)
    host_mass_history = tree_main_branch_masses[itree]
    
    # Call the host object
    host = sga.Host(host_mass_history, tree_redshifts, cosmology,
                    fd=fd, flattening=flattening, output_zred=output_zred)
    
    # Define result keys once
    result_keys = [
        'prog_masses', 'prog_mstars', 'status', 'radii', 'tsteps', 'tage',
        'prog_dp', 'levels_at_tsteps', 'host_times_starting_from_initial_level', 'orbit'
    ]

    # Initialize results dict with empty lists
    results = {key: [] for key in result_keys}
    
    for iprog in range(0,nprogs_this_tree):

        prog_mass = progenitors['ProgenitorMass'][progs_this_tree][iprog]
        prog_ilev = progenitors['ProgenitorIlev'][progs_this_tree][iprog]
        # Call the progenitor object
        prog = sga.Progenitor(prog_mass, host, level=prog_ilev)
        
        # Define a time step to evolve
        total_time_gyr = prog.infall_t_lbk
        # tsteps grid and output array sizes will be nsteps+1,
        # because the initial conditions are included at index zero (t=0)
        tsteps = np.linspace(0, total_time_gyr, nsteps+1)

        # Call the evolve orbit object
        solution = sga.evolve_orbit(host, prog, tsteps, evolve_prog_mass=True)     

        for key in result_keys:
            results[key].append(solution[key])

    for k in results.keys():
        results[k] = np.array(results[k])
        
    results['nprog'] = len(results['prog_masses'])
    
    t_end = time()
    results['t_proc'] = t_end - t_start
    results['itree']  = itree
    return results

###########################################################
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
    
    # Read main branch mass histories (immediately deal with little h)    
    tree_main_branch_masses = sga.read_hdf5(tree_file,'/Mainbranch/MainbranchMass')/hubble_parameter

    # Number of treees and tree levels
    ntrees, nlev = tree_main_branch_masses.shape

    print('{:d} trees, {:d} levels'.format(ntrees,nlev))
    
    
    # Read progenitor data (immediately deal with little h on masses)
    progenitor_dataset_names = ["HostMass","ProgenitorZred","ProgenitorMass","ProgenitorIlev","TreeID"]
    
    progenitors = sga.read_hdf5(tree_file, progenitor_dataset_names, group='/Progenitors')
    progenitors['ProgenitorMass'] = progenitors['ProgenitorMass']/hubble_parameter
    progenitors['HostMass']       = progenitors['HostMass']/hubble_parameter

    tree_redshifts = sga.read_hdf5(tree_file,'Redshift',group='/OutputTimes')
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



###########################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Satgen for Arrakihs")
    parser.add_argument("tree_file", help="Input PCHTrees file (PFOP run)")
    parser.add_argument("--ncores","-n", help="Number of cores", default=1, type=int)
    parser.add_argument("--substeps","-s", help="Number of substeps", default=None, type=int)
    parser.add_argument("--output","-o", help="Output filename", default='all_progenitors.hdf5')
    return parser.parse_args()


###########################################################
if __name__ == '__main__':
    args = parse_args()
    main(args)
