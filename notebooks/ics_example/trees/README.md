## Steps to create trees

1. Edit `pchtrees_pfop_100_mw.toml` to set `data_path` to the data directory of
   your copy of pchtrees.

2. Make sure the `run_pchtrees_pfop_100_mw.sh` script can call the `pchtrees`
   binary. This might involve one of the following: 

	* Copy the `pchtrees` binary here;
	* ... or make a symbolic link to it;
	* ... or put the full path to the binary in the script;
	* ... or put the path to the binary on the system PATH (not
	  reccomended).

If you're not running under SLURM, edit out the bits of the script you don't
think you'll need.
