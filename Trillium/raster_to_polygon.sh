#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:30:00
#SBATCH --job-name=raster_to_polygon
#SBATCH --output=/scratch/arbmarta/raster_to_polygon_%j.out

source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/raster_to_polygon.py
