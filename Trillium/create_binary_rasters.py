import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from multiprocessing import Pool
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.io
from rasterio.windows import from_bounds
from tqdm import tqdm

# Constants
OUT_DIR = "/scratch/arbmarta/Binary Rasters"
os.makedirs(OUT_DIR, exist_ok=True)

## ------------------------------------------- INPUT DATASETS -------------------------------------------

datasets = {
    "Vancouver": {
        "epsg": 32610,
        "ETH": "/scratch/arbmarta/CHMs/ETH/Vancouver ETH.tif",
        "Meta": "/scratch/arbmarta/CHMs/Meta/Vancouver Meta.tif",
        "Potapov": "/scratch/arbmarta/CHMs/Potapov/Vancouver Potapov.tif",
        "DW_10m": "/scratch/arbmarta/Land Cover/DW_2020/Vancouver DW.tif", # Trees indicated by Value 1
        "ESRI": "/scratch/arbmarta/Land Cover/ESRI/Vancouver ESRI.tif", # Trees indicated by Value 2
        "Terrascope 2020": "/scratch/arbmarta/Land Cover/Terrascope/Vancouver 2020 Terrascope.tif", # Trees indicated by Value 10
        "Terrascope 2021": "/scratch/arbmarta/Land Cover/Terrascope/Vancouver 2021 Terrascope.tif"
    },
    "Winnipeg": {
        "epsg": 32614,
        "ETH": "/scratch/arbmarta/CHMs/ETH/Winnipeg ETH.tif",
        "Meta": "/scratch/arbmarta/CHMs/Meta/Winnipeg Meta.tif",
        "Potapov": "/scratch/arbmarta/CHMs/Potapov/Winnipeg Potapov.tif",
        "DW_10m": "/scratch/arbmarta/Land Cover/DW_2020/Winnipeg DW.tif",
        "ESRI": "/scratch/arbmarta/Land Cover/ESRI/Winnipeg ESRI.tif",
        "Terrascope 2020": "/scratch/arbmarta/Land Cover/Terrascope/Winnipeg 2020 Terrascope.tif",
        "Terrascope 2021": "/scratch/arbmarta/Land Cover/Terrascope/Winnipeg 2021 Terrascope.tif"
    },
    "Ottawa": {
        "epsg": 32618,
        "ETH": "/scratch/arbmarta/CHMs/ETH/Ottawa ETH.tif",
        "Meta": "/scratch/arbmarta/CHMs/Meta/Ottawa Meta.tif",
        "Potapov": "/scratch/arbmarta/CHMs/Potapov/Ottawa Potapov.tif",
        "DW_10m": "/scratch/arbmarta/Land Cover/DW_2020/Ottawa DW.tif",
        "ESRI": "/scratch/arbmarta/Land Cover/ESRI/Ottawa ESRI.tif",
        "Terrascope 2020": "/scratch/arbmarta/Land Cover/Terrascope/Ottawa 2020 Terrascope.tif",
        "Terrascope 2021": "/scratch/arbmarta/Land Cover/Terrascope/Ottawa 2021 Terrascope.tif"
    }
}

chms = ["ETH", "Meta", "Potapov"]
lcs = ["DW_10m", "ESRI", "Terrascope 2020", "Terrascope 2021"]

# Dictionary mapping each land cover type to its specific canopy value
canopy_values = {
    "DW_10m": 1,
    "ESRI": 2,
    "Terrascope 2020": 10,
    "Terrascope 2021": 10
}

## ------------------------------------------- FUNCTIONS TO CHECK PROJECTION OF RASTERS AND SHAPEFILES -------------------------------------------

def get_epsg_int(crs):
    if crs is None:
        return None
    return int(crs.to_string().split(":")[1])

for city, info in datasets.items():
    target_epsg = info["epsg"]
    
    # Shapefile EPSG
    gdf = gpd.read_file(info["shp"])
    shp_epsg = get_epsg_int(gdf.crs)
    if shp_epsg is None:
        print(f"[Error] {city} shapefile has no CRS defined")
    elif shp_epsg != target_epsg:
        print(f"[Mismatch] {city} shapefile EPSG: {shp_epsg} != {target_epsg}")
    else:
        print(f"[OK] {city} shapefile EPSG matches {target_epsg}")

    # Raster EPSG
    all_match = True
    for key in raster_keys:
        if key in info:
            raster_path = info[key]
            try:
                with rasterio.open(raster_path) as src:
                    raster_epsg = get_epsg_int(src.crs)
                    if raster_epsg is None:
                        print(f"[Mismatch] {city} raster '{key}' has no CRS defined")
                        all_match = False
                    elif raster_epsg != target_epsg:
                        print(f"[Mismatch] {city} raster '{key}' EPSG: {raster_epsg} != {target_epsg}")
                        all_match = False
            except Exception as e:
                print(f"[Error] Could not open {city} raster '{key}': {e}")
                all_match = False

    if all_match:
        print(f"[OK] All rasters for {city} have the correct EPSG {target_epsg}")

## ------------------------------------------- CONVERT CANOPY HEIGHT MODELS TO BINARY CANOPY COVER MAPS -------------------------------------------

print("Starting CHM to binary raster conversion...")

# Create a list of all CHM file paths to process for the progress bar
files_to_process = [
    datasets[city][chm_key]
    for city in datasets
    for chm_key in chms
    if chm_key in datasets[city]
]

# Process each CHM file
for chm_path in tqdm(files_to_process, desc="Converting Rasters"):
    try:
        # Construct the full output path while preserving the original filename
        file_name = os.path.basename(chm_path)
        output_path = os.path.join(OUT_DIR, file_name)

        # Open the source CHM raster
        with rasterio.open(chm_path) as src:
            # Get the metadata (profile) from the source raster
            profile = src.profile.copy()
            
            # Read the raster's data, filling nodata values with 0
            # This ensures nodata areas are treated as False (0) in the next step
            chm_data = src.read(1, masked=True).filled(0)

            # Apply the binary threshold condition:
            # Pixels >= 2 become True (1), and pixels < 2 (including former nodata) become False (0)
            binary_data = (chm_data >= 2).astype(rasterio.uint8)

            # Update the profile for the output binary raster
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw'
            )
            
            # Remove the nodata key from profile since we filled all nodata values
            profile.pop('nodata', None)

            # Write the new binary data to the output file
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(binary_data, 1)

    except Exception as e:
        print(f"An error occurred while processing {chm_path}: {e}")

print(f"\n✅ Conversion complete. Binary rasters are saved in: {OUT_DIR}")

## ------------------------------------------- CONVERT LAND COVER RASTERS TO BINARY CANOPY COVER MAPS -------------------------------------------

print("Starting Land Cover to binary raster conversion...")

# Create a list of (file_path, lc_key) tuples to process
files_to_process = [
    (datasets[city][lc_key], lc_key)
    for city in datasets
    for lc_key in lcs
    if lc_key in datasets[city]
]

# Process each land cover file
for lc_path, lc_key in tqdm(files_to_process, desc="Converting LC Rasters"):
    try:
        # Get the specific value that represents canopy for this dataset
        target_value = canopy_values[lc_key]

        # Construct the full output path while preserving the original filename
        file_name = os.path.basename(lc_path)
        output_path = os.path.join(OUT_DIR, file_name)

        # Open the source land cover raster
        with rasterio.open(lc_path) as src:
            # Get the metadata from the source raster
            profile = src.profile.copy()
            
            # Read the raster's data, filling nodata values with a non-target value (e.g., 0)
            # This ensures nodata areas are treated as False (0) in the next step
            lc_data = src.read(1, masked=True).filled(0)

            # Apply the binary condition:
            # Pixels that equal the target_value become 1, all others become 0
            binary_data = (lc_data == target_value).astype(rasterio.uint8)

            # Update the profile for the output binary raster
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw'
            )
            
            # Remove the nodata key from profile since we filled all nodata values
            profile.pop('nodata', None)

            # Write the new binary data to the output file
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(binary_data, 1)

    except Exception as e:
        print(f"An error occurred while processing {lc_path}: {e}")

print(f"\n✅ Land Cover conversion complete. Binary rasters are saved in: {OUT_DIR}")
