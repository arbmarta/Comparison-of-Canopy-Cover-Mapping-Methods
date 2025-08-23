#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=03:30:00
#SBATCH --job-name=buildings
#SBATCH --output=/scratch/arbmarta/Outputs/Out/buildings.out
#SBATCH --error=/scratch/arbmarta/Outputs/Err/buildings.err

# Activate virtual environment
source /home/arbmarta/.virtualenvs/myenv/bin/activate

# Run canopy modeling script
python /scratch/arbmarta/Code/buildings.py
