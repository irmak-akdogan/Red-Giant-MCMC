#!/bin/bash
#SBATCH --job-name=red_giant_mcmc
#SBATCH --output=logs/mcmc_%j.out
#SBATCH --error=logs/mcmc_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load miniconda module
module load miniconda

# Activate the astro environment
source activate astro

# Create logs directory if it doesn't exist
mkdir -p logs

# Processing all 96 stars (indices 0-96)
START_IDX=0
END_IDX=96
RUN_ID=0

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Processing stars ${START_IDX} to ${END_IDX}"
echo "Run ID: ${RUN_ID}"
echo "CPUs available: ${SLURM_CPUS_PER_TASK}"
echo "Started at: $(date)"
echo "=========================================="

# Set the number of threads to match allocated CPUs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run the MCMC script
python mcmc_by_star.py ${START_IDX} ${END_IDX} ${RUN_ID}

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
