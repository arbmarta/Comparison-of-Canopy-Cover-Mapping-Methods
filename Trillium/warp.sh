#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=warp_potapov
#SBATCH --output=/scratch/arbmarta/warp_potapov.out
#SBATCH --error=/scratch/arbmarta/warp_potapov.err

module load gdal/3.9.1

# Ottawa (EPSG:32618)
gdalwarp -t_srs EPSG:32618 -r near -tr 30 30 "/scratch/arbmarta/Potapov/Ottawa Potapov.tif" "/scratch/>

# Winnipeg (EPSG:32614)
gdalwarp -t_srs EPSG:32614 -r near -tr 30 30 "/scratch/arbmarta/Potapov/Winnipeg Potapov.tif" "/scratc>

# Vancouver (EPSG:32610)
gdalwarp -t_srs EPSG:32610 -r near -tr 30 30 "/scratch/arbmarta/Potapov/Vancouver Potapov.tif" "/scrat>


