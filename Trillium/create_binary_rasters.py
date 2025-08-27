import os
import geopandas as gpd
import rasterio
from multiprocessing import Pool
from tqdm import tqdm

# Constants
OUT_DIR = "/scratch/arbmarta/Binary Rasters"
os.makedirs(OUT_DIR, exist_ok=True)
N_CPUS = 192

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
raster_keys = chms + lcs

# Dictionary mapping each land cover type to its specific canopy value
canopy_values = {
    "DW_10m": 1,
    "ESRI": 2,
    "Terrascope 2020": 10,
    "Terrascope 2021": 10
}

## ------------------------------------------- FUNCTIONS -------------------------------------------

def get_epsg_int(crs):
    if crs is None:
        return None
    return int(crs.to_string().split(":")[1])

def process_chm_file(chm_path):
    """Process a single CHM file to binary format"""
    try:
        # Construct the full output path while preserving the original filename
        file_name = os.path.basename(chm_path)
        output_path = os.path.join(OUT_DIR, file_name)

        # Open the source CHM raster
        with rasterio.open(chm_path) as src:
            # Get the metadata (profile) from the source raster
            profile = src.profile.copy()
            
            # Read the raster's data, filling nodata values with 0
            chm_data = src.read(1, masked=True).filled(0)

            # Apply the binary threshold condition
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

        return f"Success: {file_name}"
    
    except Exception as e:
        return f"Error processing {chm_path}: {e}"

def process_lc_file(lc_info):
    """Process a single land cover file to binary format"""
    lc_path, lc_key = lc_info
    
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
            
            # Read the raster's data, filling nodata values with 0
            lc_data = src.read(1, masked=True).filled(0)

            # Apply the binary condition
            binary_data = (lc_data == target_value).astype(rasterio.uint8)

            # Update the profile for the output binary raster
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw'
            )
            
            # Remove the nodata key from profile
            profile.pop('nodata', None)

            # Write the new binary data to the output file
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(binary_data, 1)

        return f"Success: {file_name}"
    
    except Exception as e:
        return f"Error processing {lc_path}: {e}"

def check_raster_projection(city_info):
    """Check projection for a single city's rasters"""
    city, info = city_info
    target_epsg = info["epsg"]
    results = []
    
    for key in raster_keys:
        if key in info:
            raster_path = info[key]
            try:
                with rasterio.open(raster_path) as src:
                    raster_epsg = get_epsg_int(src.crs)
                    if raster_epsg is None:
                        results.append(f"[Mismatch] {city} raster '{key}' has no CRS defined")
                    elif raster_epsg != target_epsg:
                        results.append(f"[Mismatch] {city} raster '{key}' EPSG: {raster_epsg} != {target_epsg}")
            except Exception as e:
                results.append(f"[Error] Could not open {city} raster '{key}': {e}")
    
    if not results:
        results.append(f"[OK] All rasters for {city} have the correct EPSG {target_epsg}")
    
    return results

## ------------------------------------------- CHECK PROJECTIONS IN PARALLEL -------------------------------------------

print("Checking raster projections...")

with Pool(N_CPUS) as pool:
    projection_results = pool.map(check_raster_projection, datasets.items())

# Print all projection results
for city_results in projection_results:
    for result in city_results:
        print(result)

## ------------------------------------------- CONVERT CHM RASTERS IN PARALLEL -------------------------------------------

print("\nStarting CHM to binary raster conversion...")

# Create a list of all CHM file paths to process
chm_files = [
    datasets[city][chm_key]
    for city in datasets
    for chm_key in chms
    if chm_key in datasets[city]
]

print(f"Processing {len(chm_files)} CHM files using {N_CPUS} CPUs...")

# Process CHM files in parallel
with Pool(N_CPUS) as pool:
    # Use imap for progress tracking
    results = list(tqdm(
        pool.imap(process_chm_file, chm_files),
        total=len(chm_files),
        desc="Converting CHM Rasters"
    ))

# Print results
for result in results:
    if result.startswith("Error"):
        print(result)

print(f"\n✅ CHM conversion complete. Binary rasters are saved in: {OUT_DIR}")

## ------------------------------------------- CONVERT LAND COVER RASTERS IN PARALLEL -------------------------------------------

print("\nStarting Land Cover to binary raster conversion...")

# Create a list of (file_path, lc_key) tuples to process
lc_files = [
    (datasets[city][lc_key], lc_key)
    for city in datasets
    for lc_key in lcs
    if lc_key in datasets[city]
]

print(f"Processing {len(lc_files)} Land Cover files using {N_CPUS} CPUs...")

# Process land cover files in parallel
with Pool(N_CPUS) as pool:
    # Use imap for progress tracking
    results = list(tqdm(
        pool.imap(process_lc_file, lc_files),
        total=len(lc_files),
        desc="Converting LC Rasters"
    ))

# Print results
for result in results:
    if result.startswith("Error"):
        print(result)

print(f"\n✅ Land Cover conversion complete. Binary rasters are saved in: {OUT_DIR}")
