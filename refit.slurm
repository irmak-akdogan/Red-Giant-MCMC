#!/bin/bash
#SBATCH --job-name=refit_stars
#SBATCH --output=logs/refit_%A_%a.out
#SBATCH --error=logs/refit_%A_%a.err
#SBATCH --array=0-124
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Load miniconda module
module load miniconda

# Activate the astro environment
source activate astro

# Create logs and output directories if they don't exist
mkdir -p logs
mkdir -p refit_output

# Total rows: 12466
# Chunk size: 100 rows per task
# Number of tasks: 125 (0-124)
CHUNK_SIZE=100
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$((START_IDX + CHUNK_SIZE))

# For the last task, make sure we don't exceed total rows
if [ $END_IDX -gt 12466 ]; then
    END_IDX=12466
fi

echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing rows ${START_IDX} to ${END_IDX}"
echo "Started at: $(date)"
echo "=========================================="

# Run the refit script
python refit_stars.py --start-id ${START_IDX} --end-id ${END_IDX} -o refit_output/refit_${SLURM_ARRAY_TASK_ID}.csv

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="