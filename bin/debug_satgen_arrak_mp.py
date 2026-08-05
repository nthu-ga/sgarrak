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
import sgarrak.strip as strip

import argparse
import h5py

from functools import partial
from time import sleep
 
import matplotlib.pyplot as pl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'


###########################################################
def process_tree(itree ,fd=0.0 ,flattening=25.,
                 disk_method='fd',
                 walk_tree='backward',
                 smhm='m18',
                 concentration_method='ludlow',
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
                 z0_smhm=False,
                 orbit_init_method='li2020'):
    """
    """
    #sleep(3)
    t_start = time.perf_counter()

    nsteps = 200

    np.random.seed(42) 
    progs_this_tree   = np.flatnonzero(progenitors['TreeID'] == itree)
    nprogs_this_tree  = len(progs_this_tree)
    host_mass_history = tree_main_branch_masses[itree]
    if nprogs_max is None:
        nprogs_max = nprogs_this_tree
    # Call the host object
    host = sga.Host(host_mass_history, tree_redshifts, cosmology,
                    fd=fd, flattening=flattening, output_zred=output_zred,
                    disk_method=disk_method,walk_tree=walk_tree, 
                    concentration_method=concentration_method,
                    cooling_threshold=cooling_threshold,z0_smhm=z0_smhm,smhm=smhm)
    # Define result keys once
    result_keys = [
    'prog_masses', 'prog_mstars', 'status', 'radii', 'tsteps', 'tage',
        'levels_at_tsteps', 'orbit', 'host_dp', 'orbit_object'
    ]

    if host.has_disk:
        result_keys.append('host_disk_masses')

    # Initialize results dict with empty lists
    results = {key: [] for key in result_keys}
    
    
    for iprog in range(0,nprogs_max): 
        start_time = time.perf_counter()

        prog_mass = progenitors['ProgenitorMass'][progs_this_tree][iprog]
        prog_ilev = progenitors['ProgenitorIlev'][progs_this_tree][iprog]
        # Call the progenitor object
        prog = sga.Progenitor(prog_mass, host, level=prog_ilev, mstar_shift=mstar_shift,smhm=smhm,
                              orbit_init_method=orbit_init_method)
        
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
def process_stream(result,mass_contribution_cut,cut_method):
    """
    """
    # Unpack require info
    orbit_object    = result['orbit_object']
    host_dp         = result['host_dp']
    level_at_tsteps = result['levels_at_tsteps']
    tsteps          = result['tsteps']
    tage            = result['tage']
    prog_masses     = result['prog_masses']
    prog_mstars     = result['prog_mstars']
    itree           = result['itree']

    get_stream = strip.store_strip_info(orbit_object,host_dp,level_at_tsteps,tsteps,tage,
                                                prog_masses,prog_mstars,mass_contribution_cut,cut_method,itree)
    return get_stream
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
    parser.add_argument("--output","-o", help="Output filename", default="test_all_progenitors.hdf5")
    parser.add_argument("--nprogs", help="Only process a fixed number of progenitors per tree", default=None, type=int)
    parser.add_argument("--ntrees", help="Only process a fixed number of trees", default=None, type=int)
    parser.add_argument("--hubble","-H", help="H0 hubble constant", default=0.7,type=float)
    parser.add_argument("--serial", help="Execute in serial, no multithreading",action="store_true")
    parser.add_argument("--z0_smhm", help="Whether uses z0 smhm relation for disk masses",default=False)
    parser.add_argument("--cooling_threshold", help="Whether turns on a cooling threshold check for disk growth",default=True)
    parser.add_argument("--smhm", help="stellar mass halo mass relation",default="m18")
    parser.add_argument("--orbit_init_method", help="orbit initial method for progenitors",default='li2020')
    parser.add_argument("--concentration_method", help="choose host concentration model: zhao, smooth, ludlow",default='ludlow')
    parser.add_argument("--mass_contribution_cut",nargs="+",type=float,default=[0.9])
    parser.add_argument("--cut_method", help="cut by the stellar mass or total fraction of mass contributed",default='fraction')
    return parser.parse_args()

###########################################################
if __name__ == '__main__':
    args = parse_args()
    # multiprocessing.set_start_method("forkserver")
    # multiprocessing.set_start_method('fork')
   
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])
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
                                   smhm = args.smhm,
                                   concentration_method = args.concentration_method,
                                   orbit_init_method = args.orbit_init_method,
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
        processing_ranges = [np.arange(0,ntrees_max)]
    else:
        ntrees_max = ntrees
        processing_ranges = np.array([np.arange(0,20)+20*i for i in range(0,49)])
    print('Running {:d} trees'.format(ntrees_max))

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

    # Define data types so empty arrays does not turn int arrays into floats.
    stream_dtypes = {
    "itree": np.int64,
    "top_iprog": np.int64,
    "prog_each_length": np.int64,
    "prog_start_index": np.int64,
    #"strip_each_length": np.int64,
    #"strip_start_index": np.int64,
    "sindex": np.int64,
    "star_coor": np.float64,
    "dm_coor": np.float64,
    }

    # Create sizable arrays
    with h5py.File(args.output, "w") as f:
        grp_tree         = f.create_group("Results")
        grp_tree_main    = f.create_group("ResultsMain")
        grp_stream       = f.create_group("Streams")
        grp_param        = f.create_group("Parameters")
        grp_output_times = f.create_group("OutputTimes")

        dsets_tree     = {}
        dsets_tree_main= {}
        dsets_stream   = {}

        example1,example2 = partial_process_tree(0)
        example3 = process_stream(example1,args.mass_contribution_cut,args.cut_method)

        for k, v in example1.items():
            if k not in ('host_dp', 'orbit_object'):
                v = np.atleast_1d(v)
                shape = (0,) + v.shape[1:]
                maxshape = (None,) + v.shape[1:]
                dsets_tree[k] = grp_tree.create_dataset(
                    k,
                    shape=shape,
                    maxshape=maxshape,
                    dtype=v.dtype,
                    chunks=True
                )

        for k, v in example2.items():
            v = np.atleast_1d(v)
            shape = (0,) + v.shape[1:]
            maxshape = (None,) + v.shape[1:]
            dsets_tree_main[k] = grp_tree_main.create_dataset(
                    k,
                    shape=shape,
                    maxshape=maxshape,
                    dtype=v.dtype,
                    chunks=True
                )

        for k, v in example3.items():
            v = np.asarray(v)
            if k in ("star_coor", "dm_coor"):
                tail_shape = (3,)
            else:
                if v.ndim == 0:
                    v = np.atleast_1d(v)
                tail_shape = v.shape[1:]

            dtype = stream_dtypes.get(k, v.dtype if v.size > 0 else np.float64)

            shape = (0,) + tail_shape
            maxshape = (None,) + tail_shape

            dsets_stream[k] = grp_stream.create_dataset(
                k,
                shape=shape,
                maxshape=maxshape,
                dtype=dtype,
                chunks=True
            )

        for k, v in params.items():
            grp_param.create_dataset(k, data=np.atleast_1d(v))

        for k, v in output_times.items():
            grp_output_times.create_dataset(k, data=np.atleast_1d(v))


        if args.serial:
            # It should be safe for series to write files individually
            for itree in range(0,ntrees_max):
                print()
                print('Tree {:d} of {:d}'.format(itree+1,ntrees_max))
                print()
                results_this_tree, tree_data_this_tree = partial_process_tree(itree)
                stream = process_stream(results_this_tree,args.mass_contribution_cut,args.cut_method)
                results_this_tree = {k: v for k, v in results_this_tree.items() if k not in ('host_dp', 'orbit_object')}

                for k,v in results_this_tree.items():
                    if k not in ('host_dp', 'orbit_object'):
                        v = np.atleast_1d(v)
                        dset = dsets_tree[k]
                        n_old = dset.shape[0]
                        n_new = n_old + v.shape[0]
                        dset.resize((n_new,) + dset.shape[1:])
                        dset[n_old:n_new] = v

                for k, v in tree_data_this_tree.items():
                    v = np.atleast_1d(v)
                    dset = dsets_tree_main[k]
                    n_old = dset.shape[0]
                    n_new = n_old + v.shape[0]
                    dset.resize((n_new,) + dset.shape[1:])
                    dset[n_old:n_new] = v

                for k, v in stream.items():
                    v = np.atleast_1d(v)
                    dset = dsets_stream[k]
                    n_old = dset.shape[0]
                    n_new = n_old + v.shape[0]
                    dset.resize((n_new,) + dset.shape[1:])
                    dset[n_old:n_new] = v
        else:
            import multiprocessing
            with multiprocessing.Manager() as manager:
                results      = manager.list()
                results_main = manager.list()
                streams      = manager.list()
                for processing_range in processing_ranges:
                    print('Doing tree range: ',processing_range)
                    with multiprocessing.get_context("forkserver").Pool(processes=ncpus) as pool:
                        chunksize = 2
                        print("{:10s} | {:10s} | {:10s} | {:6s}".format("IDX", "ITREE", "PID", "TIME"))
                        # for r in a_range_of_tree:
                        for itree, _ in enumerate(pool.imap_unordered(partial_process_tree, processing_range, chunksize)):
                            results_this_tree, tree_data_this_tree = _
                            print("{:10d} | {:10d} | {:10d} | {:6.2f}s".format(itree, 
                                results_this_tree['itree'], 
                                results_this_tree['_PID'],
                                results_this_tree['t_proc']))
                            sys.stdout.flush()
                            results_main.append(tree_data_this_tree)
                            stream = process_stream(results_this_tree,args.mass_contribution_cut,args.cut_method)
                            streams.append(stream)
                            # Remove unnecessary data before writing the result
                            results_this_tree = {k: v for k, v in results_this_tree.items() if k not in ('host_dp', 'orbit_object')}
                            results.append(results_this_tree)
                            #tree_data.append(tree_data_this_tree)
                            #tree_order.append(itree)

                    pool.join()

                    # save the data for a set of tree range
                    for result in results:
                        for k, v in result.items():
                            if k not in ('host_dp', 'orbit_object'):
                                v = np.atleast_1d(v)
                                dset = dsets_tree[k]
                                n_old = dset.shape[0]
                                n_new = n_old + v.shape[0]
                                dset.resize((n_new,) + dset.shape[1:])
                                dset[n_old:n_new] = v

                    for result_main in results_main:
                        for k, v in result_main.items():
                            v = np.atleast_1d(v)
                            dset = dsets_tree_main[k]
                            n_old = dset.shape[0]
                            n_new = n_old + v.shape[0]
                            dset.resize((n_new,) + dset.shape[1:])
                            dset[n_old:n_new] = v

                    for stream in streams:
                        for k, v in stream.items():
                            v = np.atleast_1d(v)
                            dset = dsets_stream[k]
                            n_old = dset.shape[0]
                            n_new = n_old + v.shape[0]
                            dset.resize((n_new,) + dset.shape[1:])
                            #print(k, v.shape, dset.shape)
                            dset[n_old:n_new] = v

                    # clear the occupied lists
                    results      = manager.list()
                    results_main = manager.list()
                    streams      = manager.list()


    t_end = time.perf_counter()
    print('Total processing time {:f}s'.format(t_end-t_start))
    print('             Per tree {:f}s'.format((t_end-t_start)/ntrees_max))

    print('Done!')
