#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=canopy_polys
#SBATCH --output=/scratch/arbmarta/canopy_polys%j.out
#SBATCH --error=/scratch/arbmarta/canopy_polys%j.err

# Run your Python script
source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/raster_to_polygon.py
