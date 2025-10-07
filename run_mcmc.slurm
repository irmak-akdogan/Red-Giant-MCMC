#!/bin/bash
#SBATCH --job-name=red_giant_mcmc
#SBATCH --output=logs/mcmc_%A_%a.out
#SBATCH --error=logs/mcmc_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-11

# Load miniconda module
module load miniconda

# Activate the astro environment
source activate astro

# Create logs directory if it doesn't exist
mkdir -p logs

# Calculate star indices for this array task
# Processing 96 stars total, divided into 12 jobs (8 stars per job)
STARS_PER_JOB=8
START_IDX=$((SLURM_ARRAY_TASK_ID * STARS_PER_JOB))
END_IDX=$((START_IDX + STARS_PER_JOB))

# Run ID is the array task ID
RUN_ID=${SLURM_ARRAY_TASK_ID}

echo "=========================================="
echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
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
