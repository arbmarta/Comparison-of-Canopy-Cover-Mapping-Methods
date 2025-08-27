#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:15:00
#SBATCH --job-name=main
#SBATCH --output=/scratch/arbmarta/Outputs/Out/create_binary_rasters.out
#SBATCH --error=/scratch/arbmarta/Outputs/Err/create_binary_rasters.err

# Run your Python script
source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/Code/create_binary_rasters.py
