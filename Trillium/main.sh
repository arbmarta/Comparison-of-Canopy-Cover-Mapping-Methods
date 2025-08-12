#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:30:00
#SBATCH --job-name=canopy_poly_split
#SBATCH --output=/scratch/arbmarta/canopy_poly_split%j.out
#SBATCH --error=/scratch/arbmarta/canopy_poly_split%j.err

# Run your Python script
source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/main.py
