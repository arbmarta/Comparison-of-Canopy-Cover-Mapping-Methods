#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:30:00
#SBATCH --job-name=buildings
#SBATCH --output=/scratch/arbmarta/buildings.out
#SBATCH --error=/scratch/arbmarta/buildings.err

# Activate virtual environment
source /home/arbmarta/.virtualenvs/myenv/bin/activate

# Run canopy modeling script
python /scratch/arbmarta/buildings.py
