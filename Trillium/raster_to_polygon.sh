#!/bin/bash
#SBATCH --job-name=canopy_polys
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6           # matches your Pool(processes=6)
#SBATCH --mem=24G
#SBATCH --output=/scratch/arbmarta/canopy_%j.out
#SBATCH --error=/scratch/arbmarta/canopy_%j.err

set -euo pipefail

echo "Node: $(hostname)  |  CPUs: ${SLURM_CPUS_PER_TASK}"; date

# Use your env (has rasterio/geopandas)
module load python/3.11
source /home/arbmarta/.virtualenvs/myenv/bin/activate

# Keep libraries from oversubscribing cores; give GDAL a bigger cache
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GDAL_CACHEMAX=1024   # MB
export PYTHONUNBUFFERED=1

# Run your Python script (update the filename if needed)
python /scratch/arbmarta/your_script.py
