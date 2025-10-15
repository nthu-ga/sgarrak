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

sys.path.append(os.path.abspath('/data/chungwen/sgarrak/py'))
import sgarrak.sgarrak as sga
import argparse
import h5py

from functools import partial
from time import sleep
 
import matplotlib.pyplot as pl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'

###########################################################
def write_results(results, tree_data, params, output_times, filename,reorder=None):
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
            print('processing data: ',k)
            total_results[k].append(np.atleast_1d(results[itree][k]))
        total_results['tree_idx'].append(np.repeat(results[itree]['itree'],results[itree]['nprog']))

    for k in data_keys:
        total_results[k] = np.concatenate(total_results[k],axis=0)
    total_results['tree_idx'] = np.concatenate(total_results['tree_idx'])

    total_nprog = len(total_results['tree_idx'])

    if 'main_branch_disk_mass' in tree_data[0].keys():
        tree_array_2d_properties = [
                'main_branch_halo_mass', 
                'main_branch_halo_c',
                'main_branch_disk_mass', 
                'main_branch_disk_reff']
    else:
        tree_array_2d_properties = [
                'main_branch_halo_mass',
                'main_branch_halo_c']

    sort_order = reorder if reorder is not None else None

    # Reshape main branch arrays
    nlev = params['nlev']

    tree_results = dict()
    for k in tree_array_2d_properties:
        temp = np.zeros((ntrees,nlev),dtype=np.float32)
        for itree,_tree_data in enumerate(tree_data):
            temp[itree,:] = _tree_data[k]
        if sort_order is not None:
            temp = temp[sort_order,:]
        tree_results[k] = temp
    
    for k in tree_data[0].keys():
        if not k in tree_array_2d_properties:
            a = np.array([_[k] for _ in tree_data])
            if sort_order is not None:
                a = a[sort_order]
            tree_results[k] = a
 
    with h5py.File(filename, "w") as f:
        f["/"].create_group('Progenitors')
        for k, v in total_results.items():
            print("Writing {} data...".format(k))
            f["/Progenitors"].create_dataset(k, data=v, compression=6) 
  
        f["/"].create_group('MainBranches')
        for k, v in tree_results.items():
            print("Writing {} data...".format(k))
            f["/MainBranches"].create_dataset(k, data=v, compression=6) 
        # Can improve
        f["/MainBranches"].create_dataset('TreeIdx',data=np.array(reorder)[np.array(reorder)], compression=6) 

        f["/"].create_group('Parameters')
        for k, v in params.items():
            print("Writing parameter {}...".format(k))
            f["/Parameters"].create_dataset(k, data=np.atleast_1d(v)) 
         
        f["/"].create_group('OutputTimes')
        for k, v in output_times.items():
            print("Writing {}...".format(k))
            f["/OutputTimes"].create_dataset(k, data=np.atleast_1d(v)) 


    print('Wrote {:s}'.format(filename))
    return

###########################################################
def process_tree(itree ,fd=0.0 ,flattening=25.,
                 disk_method='fd',
                 walk_tree='backward',
                 mstar_shift=None,
                 output_zred=None,
                 progenitors=None,
                 tree_main_branch_masses=None,
                 tree_redshifts=None,
                 cosmology=None,
                 nprogs_max=None,
                 n_substeps=None,
                 verbose=False,
                 cooling_threshold=True,
                 z0_smhm=False):
    """
    """
    #sleep(3)
    t_start = time.perf_counter()

    # These should be set in the sgarrak.py module
    # SATGEN.cfg implicitly sets resolution limits on mass (absolute and relative) and radius
    # Set them explicitly here; these are the degaults
    #cfg.Mres    = 100.0
    #cfg.Rres    = 0.001
    #cfg.psi_res = 1.0e-5

    nsteps = 200

    np.random.seed(42) 
    progs_this_tree   = np.flatnonzero(progenitors['TreeID'] == itree)
    nprogs_this_tree  = len(progs_this_tree)
    host_mass_history = tree_main_branch_masses[itree]
    # Call the host object
    host = sga.Host(host_mass_history, tree_redshifts, cosmology,
                    fd=fd, flattening=flattening, output_zred=output_zred,
                    disk_method=disk_method,walk_tree=walk_tree,
                    cooling_threshold=cooling_threshold,z0_smhm=z0_smhm)

    # Define result keys once
    result_keys = [
        'prog_masses', 'prog_mstars', 'status', 'radii', 'tsteps', 'tage',
        'levels_at_tsteps', 'coors', 'has_galaxy'
    ]

    if host.has_disk:
        result_keys.append('host_disk_masses')

    # Initialize results dict with empty lists
    results = {key: [] for key in result_keys}
    
    if nprogs_max is None:
        nprogs_max = nprogs_this_tree
    
    for iprog in range(0,nprogs_max): 
        start_time = time.perf_counter()

        prog_mass = progenitors['ProgenitorMass'][progs_this_tree][iprog]
        prog_ilev = progenitors['ProgenitorIlev'][progs_this_tree][iprog]
        # Call the progenitor object
        prog = sga.Progenitor(prog_mass, host, level=prog_ilev, mstar_shift=mstar_shift)
        
        # Define a time step to evolve
        total_time_gyr = prog.infall_t_lbk
        # tsteps grid and output array sizes will be nsteps+1,
        # because the initial conditions are included at index zero (t=0)
        tsteps = np.linspace(0, total_time_gyr, nsteps+1)

        # Call the evolve orbit object
        solution = sga.evolve_orbit(host, prog, tsteps, evolve_prog_mass=True)     

        for key in result_keys:
            results[key].append(solution[key])

    t_end = time.perf_counter()
    
    results['nprog']  = len(results['prog_masses'])
    results['t_proc'] = t_end - t_start
    results['itree']  = itree

    tree_data = dict()
    tree_data['main_branch_halo_mass'] = host_mass_history
    tree_data['main_branch_halo_c']    = host.concentration
    tree_data['t_proc'] = t_end - t_start
    if host.has_disk:
        tree_data['main_branch_disk_mass'] = host.disk_mass
        tree_data['main_branch_disk_reff'] = host.disk_reff

    results['_PID'] = os.getpid()

    return results, tree_data

###########################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Satgen for Arrakihs")
    parser.add_argument("tree_file", help="Input PCHTrees file (PFOP run)")
    parser.add_argument("--ncores","-n", help="Number of cores", default=None, type=int)
    parser.add_argument("--substeps","-s", help="Number of substeps", default=None, type=int)
    parser.add_argument("--fd",help="Disk mass ratio",default=0.1, type=float)
    parser.add_argument("--flattening",help="disk scale length/scale height",default=25.,type=float)
    parser.add_argument("--disk_method",help="fd, no, interp_sm, interp, step, interp_zavg",default="fd", type=str)
    parser.add_argument("--walk_tree",help="backward or forward",default="backward", type=str)
    parser.add_argument("--mstar_shift","-msh",help="+-50 percent or so",default=None,type=float)
    parser.add_argument("--output","-o", help="Output filename", default='test_all_progenitors.hdf5')
    parser.add_argument("--nprogs", help="Only process a fixed number of progenitors per tree", default=None, type=int)
    parser.add_argument("--ntrees", help="Only process a fixed number of trees", default=None, type=int)
    parser.add_argument("--hubble","-H", help="H0 hubble constant", default=0.73,type=float)
    parser.add_argument("--serial", help="Execute in serial, no multithreading",action="store_true")
    parser.add_argument("--z0_smhm", help="Whether uses z0 smhm relation for disk masses",default=False)
    parser.add_argument("--cooling_threshold", help="Whether turns on a cooling threshold check for disk growth",default=True)
    return parser.parse_args()

###########################################################
if __name__ == '__main__':
    args = parse_args()
    # multiprocessing.set_start_method("forkserver")
    # multiprocessing.set_start_method('fork')
   
    if 'SLURM_CPUS_ON_NODE' in os.environ:
        ncpus = int(os.environ['SLURM_CPUS_ON_NODE'])
    else:
        if args.ncores is None:
            ncpus = 1 
            args.ncores=1
        else:
            ncpus = args.ncores
    print('Available cores: {:d}'.format(ncpus))
    
    if args.ncores > 1:
        print(f'Number of cores requested: {args.ncores:d}')
        #sleep(3)
    
    # Millennium
    cosmology = cosmo.FlatLambdaCDM(args.hubble*100,0.25)
    print('Cosmology:', cosmology)
    
    tree_file = args.tree_file
    print('Reading {:s}'.format(tree_file))
    
    # Read main branch mass histories (immediately deal with little h)    
    tree_main_branch_masses = sga.read_hdf5(tree_file,'/Mainbranch/MainbranchMass')/args.hubble

    # Number of treees and tree levels
    ntrees, nlev = tree_main_branch_masses.shape

    print('{:d} trees, {:d} levels'.format(ntrees,nlev))
    
    
    # Read progenitor data (immediately deal with little h on masses)
    progenitor_dataset_names = ["HostMass","ProgenitorZred","ProgenitorMass","ProgenitorIlev","TreeID"]
    
    progenitors = sga.read_hdf5(tree_file, progenitor_dataset_names, group='/Progenitors')
    progenitors['ProgenitorMass'] = progenitors['ProgenitorMass']/args.hubble
    progenitors['HostMass']       = progenitors['HostMass']/args.hubble

    tree_redshifts = sga.read_hdf5(tree_file,'Redshift',group='/OutputTimes')
    tree_t_lbk_gyr = cosmology.lookback_time(tree_redshifts).value
    tree_t_age_gyr = cosmology.age(tree_redshifts).value

    partial_process_tree = partial(process_tree, 
                                   fd = args.fd,
                                   flattening = args.flattening,
                                   disk_method = args.disk_method,
                                   walk_tree = args.walk_tree,
                                   mstar_shift = args.mstar_shift,
                                   n_substeps = args.substeps,
                                   progenitors=progenitors,
                                   tree_main_branch_masses=tree_main_branch_masses,
                                   tree_redshifts=tree_redshifts,
                                   cosmology=cosmology,
                                   nprogs_max=args.nprogs,
                                   cooling_threshold=args.cooling_threshold,
                                   z0_smhm=args.z0_smhm,
                                   verbose=True)    
 
    print('Processing...')
    t_start = time.perf_counter()
    
    if args.ntrees is not None:
        ntrees_max = args.ntrees
    else:
        ntrees_max = ntrees
    print('Running {:d} trees'.format(ntrees_max))

    results    = list()
    tree_data  = list()
    tree_order = list()

    if args.serial:
        for itree in range(0,ntrees_max):
            print()
            print('Tree {:d} of {:d}'.format(itree+1,ntrees_max))
            print()
            results_this_tree, tree_data_this_tree = partial_process_tree(itree)
            results.append(results_this_tree)
            tree_data.append(tree_data_this_tree)
            tree_order.append(itree)
    else:
        import multiprocessing
        with multiprocessing.get_context("forkserver").Pool(processes=ncpus) as pool:
            chunksize = 2
            print("{:10s} | {:10s} | {:10s} | {:6s}".format("IDX", "ITREE", "PID", "TIME"))
            for itree, _ in enumerate(pool.imap_unordered(partial_process_tree, range(ntrees_max), chunksize)):
                results_this_tree, tree_data_this_tree = _
                print("{:10d} | {:10d} | {:10d} | {:6.2f}s".format(itree, 
                    results_this_tree['itree'], 
                    results_this_tree['_PID'],
                    results_this_tree['t_proc']))
                sys.stdout.flush()
                results.append(results_this_tree)
                tree_data.append(tree_data_this_tree)
                tree_order.append(itree)

    t_end = time.perf_counter()
    print('Total processing time {:f}s'.format(t_end-t_start))
    print('             Per tree {:f}s'.format((t_end-t_start)/ntrees_max))

    # results = pool.map(partial_process_tree, range(0,8))
      
    #for itree in range(0,ntrees):
    #    print(itree)
    #    R = process_tree(itree,
    #                progenitors=progenitors,
    #                tree_main_branch_masses=tree_main_branch_masses,
    #                tree_redshifts=tree_redshifts,
    #                cosmology=cosmology)
    #    results.append(R)
    
    params = dict()
    params['nlev'] = nlev
    #params['treefile'] = str(args.tree_file)
    params['hubble'] = float(args.hubble)

    if args.substeps is not None:
        params['substeps'] = int(args.substeps)
    else:
        params['substeps'] = 0

    output_times = dict()
    output_times['Redshift'] = tree_redshifts
    output_times['LookBackTime'] = tree_t_lbk_gyr
    output_times['AgeOfUniverse'] = tree_t_age_gyr 

    write_results(results, tree_data, params, output_times, args.output,
            reorder=tree_order)
    print('Done!')
