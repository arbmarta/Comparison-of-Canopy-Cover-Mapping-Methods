#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --job-name=main
#SBATCH --output=/scratch/arbmarta/main_%j.out
#SBATCH --error=/scratch/arbmarta/main_%j.err

# Run your Python script
source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/main.py
