#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=03:30:00
#SBATCH --job-name=canopy_models
#SBATCH --output=/scratch/arbmarta/canopy_models.out
#SBATCH --error=/scratch/arbmarta/canopy_models.err

# Run your Python script
source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/canopy_models.py
