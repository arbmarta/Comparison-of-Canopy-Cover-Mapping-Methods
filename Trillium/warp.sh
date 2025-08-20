#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=warp
#SBATCH --output=/scratch/arbmarta/warp.out
#SBATCH --error=/scratch/arbmarta/warp.err

module load StdEnv/2023
module load gcc/12.3
module load gdal/3.9.1

# Ottawa (EPSG:32618)
gdalwarp -t_srs EPSG:32618 -r near -tr 30 30 "/scratch/arbmarta/Potapov/Ottawa Potapov.tif" "/scratch/arbmarta/Potapov/Ottawa_Potapov_32618.tif"

# Winnipeg (EPSG:32614)
gdalwarp -t_srs EPSG:32614 -r near -tr 30 30 "/scratch/arbmarta/Potapov/Winnipeg Potapov.tif" "/scratch/arbmarta/Potapov/Winnipeg_Potapov_32614.tif"

# Vancouver (EPSG:32610)
gdalwarp -t_srs EPSG:32610 -r near -tr 30 30 "/scratch/arbmarta/Potapov/Vancouver Potapov.tif" "/scratch/arbmarta/Potapov/Vancouver_Potapov_32610.tif"
