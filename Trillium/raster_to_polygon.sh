#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:15:00
#SBATCH --job-name=raster_to_polygon
#SBATCH --output=/scratch/arbmarta/merge_eth_%j.out

source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/raster_to_polygon.py
