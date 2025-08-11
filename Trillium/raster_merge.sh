#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:15:00
#SBATCH --job-name=merge_eth
#SBATCH --output=/scratch/arbmarta/merge_eth_%j.out

source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/raster_merge.py
