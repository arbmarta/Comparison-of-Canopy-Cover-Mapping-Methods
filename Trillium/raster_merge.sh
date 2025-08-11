#!/bin/bash
#SBATCH --job-name=merge_eth
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/arbmarta/merge_eth_%j.out
#SBATCH --error=/scratch/arbmarta/merge_eth_%j.err

# Limit implicit threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate your env
source /home/arbmarta/.virtualenvs/myenv/bin/activate

# Run your script
python /scratch/arbmarta/raster_merge.py
