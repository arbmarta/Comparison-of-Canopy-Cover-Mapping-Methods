#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:30:00
#SBATCH --job-name=main
#SBATCH --output=/scratch/arbmarta/Outputs/Out/main.out
#SBATCH --error=/scratch/arbmarta/Outputs/Err/main.err

# Run your Python script
source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/Code/main.py
