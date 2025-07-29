#!/bin/bash
#SBATCH --export=NONE
#SBATCH --partition cpugpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 36
#SBATCH --mem 20G
#SBATCH --time 48:00:00
#SBATCH --job-name satgen
#SBATCH --output ./slurm_%j.log
#SBATCH --exclusive

# Clear modules from the submitting environment
module purge
# Load the cluster python module
module load python/3.10_miniforge
# Limit the number of python threads
module load slurm_limit_threads
# Load hdf5
module load gcc/9.4.0
module load hdf5

# Load environment
source activate cw


# Path to satgen_arrak script
EXEC=/data/chungwen/sgarrak/bin/satgen_arrak_mp.py 

# Parameters
PCHTREES_RUN=1000_mixed_logmass

INPUT=/data/chungwen/sgarrak/pchtrees/runs/${PCHTREES_RUN}/output_satgen_${PCHTREES_RUN}.hdf5
OUTPUT=./prog_evo_${PCHTREES_RUN}_${SUBSTEP}ss.hdf5

srun --export=ALL python ${EXEC} -n ${SLURM_CPUS_PER_TASK} -o ${OUTPUT} ${INPUT}
