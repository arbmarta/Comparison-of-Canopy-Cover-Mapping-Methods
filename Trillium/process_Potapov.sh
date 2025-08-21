#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --job-name=process_Potapov
#SBATCH --error=/scratch/arbmarta/process_Potapov.err
#SBATCH --output=/scratch/arbmarta/process_Potapov.out

source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/process_Potapov.py
