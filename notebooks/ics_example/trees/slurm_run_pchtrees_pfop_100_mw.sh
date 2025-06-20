#!/bin/bash
#SBATCH --export=NONE
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --time 5:00:00
#SBATCH --job-name pchtrees
#SBATCH --output ./slurm_%j.log

# Clear modules from the submitting environment
module purge
# Load the cluster python module
module load gcc/9.4.0 hdf5

#'2.308463e+12 is 10**12.5 * 0.73

# All paths relative to executable
# We use the pchtrees excutable in CWD; could be a symlink

srun ./pchtrees --ntrees 1000 --mphalo 2.308463e11 --mmax 2.308463e12 --params ./pfop_satgen_1000_mixed_logmass.toml --no-output-trees --process-first-order-progenitors --loguniform
