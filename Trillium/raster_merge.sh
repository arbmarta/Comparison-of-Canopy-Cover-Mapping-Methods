#!/bin/bash
#SBATCH --job-name=merge_eth
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/raster_merge.py
