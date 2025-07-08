## Steps to create progenitor lists

1. Edit `pchtrees_pfop_100_mw.toml` to set `data_path` to the data directory of
   your copy of pchtrees.

2. Make sure the `run_pchtrees_pfop_100_mw.sh` script can call the `pchtrees`
   binary. This might involve one of the following: 

	* Copy the `pchtrees` binary here;
	* ... or make a symbolic link to it;
	* ... or put the full path to the binary in the script;
	* ... or put the path to the binary on the system PATH (not
	  reccomended).

With the default settings in the script/parameter file, this should be quite
quick to run (about 30s).

The output is a file `output_progenitors_100_mw.hdf5` with the progenitor lists
for 100 random roughly MW-mass merger trees in the Millennium Simulation
cosmology. By default the script doesn't write the full trees (see the
options).

If you're using SLURM (or another batch queue system) you can start from
`slurm_run_pchtrees_pfop_100_mw.sh` to create a suitable batch script.
