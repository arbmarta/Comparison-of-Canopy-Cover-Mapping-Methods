#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:45:00
#SBATCH --job-name=canopy_models
#SBATCH --output=/scratch/arbmarta/Outputs/Out/canopy_models.out
#SBATCH --error=/scratch/arbmarta/Outputs/Err/canopy_models.err

# Activate virtual environment
source /home/arbmarta/.virtualenvs/myenv/bin/activate

# Run canopy modeling script
python /scratch/arbmarta/Code/canopy_models.py
