import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from multiprocessing import Pool
from tqdm import tqdm

# Constants
OUT_DIR = "/scratch/arbmarta/Outputs/CSVs"
os.makedirs(OUT_DIR, exist_ok=True)
N_CPUS = 192

## ------------------------------------------- INPUT DATASETS -------------------------------------------

datasets = {
    "Vancouver": {
        "shp": "/scratch/arbmarta/Trinity/Vancouver/TVAN.shp",
        "epsg": 32610,
        "ETH": "/scratch/arbmarta/Binary Rasters/Vancouver ETH.tif",
        "Meta": "/scratch/arbmarta/Binary Rasters/Vancouver Meta.tif",
        "Potapov": "/scratch/arbmarta/Binary Rasters/Vancouver Potapov.tif",
        "DW_10m": "/scratch/arbmarta/Binary Rasters/Vancouver DW.tif",
        "ESRI": "/scratch/arbmarta/Binary Rasters/Vancouver ESRI.tif",
        "Terrascope 2020": "/scratch/arbmarta/Binary Rasters/Vancouver 2020 Terrascope.tif",
        "Terrascope 2021": "/scratch/arbmarta/Binary Rasters/Vancouver 2021 Terrascope.tif",
        "GLCF": "/scratch/arbmarta/CCPs/GLCF/Vancouver GLCF.tif",
        "GLOBMAPFTC": "/scratch/arbmarta/CCPs/GLOBMAPFTC/Vancouver GLOBMAPFTC.tif"
    },
    "Winnipeg": {
        "shp": "/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp",
        "epsg": 32614,
        "ETH": "/scratch/arbmarta/Binary Rasters/Winnipeg ETH.tif",
        "Meta": "/scratch/arbmarta/Binary Rasters/Winnipeg Meta.tif",
        "Potapov": "/scratch/arbmarta/Binary Rasters/Winnipeg Potapov.tif",
        "DW_10m": "/scratch/arbmarta/Binary Rasters/Winnipeg DW.tif",
        "ESRI": "/scratch/arbmarta/Binary Rasters/Winnipeg ESRI.tif",
        "Terrascope 2020": "/scratch/arbmarta/Binary Rasters/Winnipeg 2020 Terrascope.tif",
        "Terrascope 2021": "/scratch/arbmarta/Binary Rasters/Winnipeg 2021 Terrascope.tif",
        "GLCF": "/scratch/arbmarta/CCPs/GLCF/Winnipeg GLCF.tif",
        "GLOBMAPFTC": "/scratch/arbmarta/CCPs/GLOBMAPFTC/Winnipeg GLOBMAPFTC.tif"
    },
    "Ottawa": {
        "shp": "/scratch/arbmarta/Trinity/Ottawa/TOTT.shp",
        "epsg": 32618,
        "ETH": "/scratch/arbmarta/Binary Rasters/Ottawa ETH.tif",
        "Meta": "/scratch/arbmarta/Binary Rasters/Ottawa Meta.tif",
        "Potapov": "/scratch/arbmarta/Binary Rasters/Ottawa Potapov.tif",
        "DW_10m": "/scratch/arbmarta/Binary Rasters/Ottawa DW.tif",
        "ESRI": "/scratch/arbmarta/Binary Rasters/Ottawa ESRI.tif",
        "Terrascope 2020": "/scratch/arbmarta/Binary Rasters/Ottawa 2020 Terrascope.tif",
        "Terrascope 2021": "/scratch/arbmarta/Binary Rasters/Ottawa 2021 Terrascope.tif",
        "GLCF": "/scratch/arbmarta/CCPs/GLCF/Ottawa GLCF.tif",
        "GLOBMAPFTC": "/scratch/arbmarta/CCPs/GLOBMAPFTC/Ottawa GLOBMAPFTC.tif"
    }
}

binary_raster_keys = ["ETH", "Meta", "Potapov", "DW_10m", "ESRI", "Terrascope 2020", "Terrascope 2021"]
fractional_raster_keys = ["GLCF", "GLOBMAPFTC"]
all_raster_keys = binary_raster_keys + fractional_raster_keys

# Single grid size only
GRID_SIZE = 120
CELL_AREA = GRID_SIZE * GRID_SIZE

## ------------------------------------------- FUNCTIONS -------------------------------------------

def get_epsg_int(crs):
    if crs is None:
        return None
    return int(crs.to_string().split(":")[1])

def raster_to_polygons_binary(masked_arr, out_transform, nodata=None, crs=None):
    """Convert binary raster to polygons"""
    band = masked_arr[0]
    valid = ~np.isnan(band) if np.issubdtype(band.dtype, np.floating) else np.ones_like(band, dtype=bool)
    if nodata is not None:
        valid &= (band != nodata)
    mask_vals = valid & (band == 1)  # canopy is binary: 1 = tree, 0 = not tree
    results = [
        (shape(geom), int(val))
        for geom, val in shapes(band, mask=mask_vals, transform=out_transform)
        if val == 1
    ]
    if not results:
        return gpd.GeoDataFrame(columns=["value", "geometry"], geometry=[], crs=crs)
    geoms, vals = zip(*results)
    return gpd.GeoDataFrame({"value": vals}, geometry=list(geoms), crs=crs)

def raster_to_polygons_fractional(masked_arr, out_transform, nodata=None, crs=None):
    """Convert fractional raster to polygons with canopy cover percentages"""
    band = masked_arr[0]
    valid = ~np.isnan(band) if np.issubdtype(band.dtype, np.floating) else np.ones_like(band, dtype=bool)
    if nodata is not None:
        valid &= (band != nodata)

    # For fractional data, we want all valid pixels with canopy cover > 0
    mask_vals = valid & (band > 0)
    results = [
        (shape(geom), float(val))
        for geom, val in shapes(band, mask=mask_vals, transform=out_transform)
        if val > 0
    ]
    if not results:
        return gpd.GeoDataFrame(columns=["canopy_percent", "geometry"], geometry=[], crs=crs)
    geoms, vals = zip(*results)
    return gpd.GeoDataFrame({"canopy_percent": vals}, geometry=list(geoms), crs=crs)

def process_grid_binary(args):
    """Process binary raster for 120m grid"""
    city, raster_path, grid_geom, grid_id, epsg, source = args

    result = {
        "grid_id": grid_id,
        "city": city,
        "source": source,
        "grid_size_m": GRID_SIZE
    }

    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, [grid_geom], crop=True)
            polygons = raster_to_polygons_binary(out_image, out_transform, src.nodata, crs=src.crs)

            if polygons.empty:
                result.update({
                    "total_m2": 0,
                    "percent_cover": 0,
                    "polygon_count": 0
                })
            else:
                polygons.set_crs(src.crs, inplace=True)
                polygons = polygons.to_crs(epsg)
                clipped = gpd.overlay(
                    polygons,
                    gpd.GeoDataFrame(geometry=[grid_geom], crs=epsg),
                    how="intersection"
                )
                clipped["m2"] = clipped.geometry.area

                total_m2 = clipped["m2"].sum()
                result.update({
                    "total_m2": total_m2,
                    "polygon_count": len(clipped),
                    "percent_cover": (total_m2 / CELL_AREA) * 100
                })

    except Exception as e:
        print(f"Error processing binary {city} {source}: {e}")
        result.update({
            "total_m2": 0,
            "percent_cover": 0,
            "polygon_count": 0
        })

    return result

def process_grid_fractional(args):
    """Process fractional raster for 120m grid"""
    city, raster_path, grid_geom, grid_id, epsg, source = args

    result = {
        "grid_id": grid_id,
        "city": city,
        "source": source,
        "grid_size_m": GRID_SIZE
    }

    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, [grid_geom], crop=True, all_touched=True)
            polygons = raster_to_polygons_fractional(out_image, out_transform, src.nodata, crs=src.crs)

            if polygons.empty:
                result.update({
                    "total_m2": 0,
                    "percent_cover": 0,
                    "polygon_count": 0
                })
            else:
                polygons.set_crs(src.crs, inplace=True)
                polygons = polygons.to_crs(epsg)
                clipped = gpd.overlay(
                    polygons,
                    gpd.GeoDataFrame(geometry=[grid_geom], crs=epsg),
                    how="intersection"
                )

                # Calculate area for each polygon
                clipped["polygon_area"] = clipped.geometry.area

                # Calculate canopy area by multiplying polygon area by canopy percentage
                clipped["canopy_m2"] = clipped["polygon_area"] * (clipped["canopy_percent"] / 100.0)

                total_canopy_m2 = clipped["canopy_m2"].sum()
                result.update({
                    "total_m2": total_canopy_m2,
                    "polygon_count": len(clipped),
                    "percent_cover": (total_canopy_m2 / CELL_AREA) * 100
                })

    except Exception as e:
        print(f"Error processing fractional {city} {source}: {e}")
        result.update({
            "total_m2": 0,
            "percent_cover": 0,
            "polygon_count": 0
        })

    return result

def worker(task):
    """Worker function to route tasks to appropriate processor"""
    task_type, args = task
    if task_type == "binary":
        return process_grid_binary(args)
    elif task_type == "fractional":
        return process_grid_fractional(args)

## ------------------------------------------- CHECK PROJECTIONS -------------------------------------------

print("Checking projections...")

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
    for key in all_raster_keys:
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

## ------------------------------------------- CHECK RASTER COVERAGE -------------------------------------------

print("Checking raster coverage...")

for city, info in datasets.items():
    bayan_gdf = gpd.read_file(info["shp"]).to_crs(info["epsg"])
    bayan_bounds = bayan_gdf.total_bounds  # [minx, miny, maxx, maxy]

    for key in all_raster_keys:
        if key in info:
            raster_path = info[key]
            try:
                with rasterio.open(raster_path) as src:
                    raster_bounds = src.bounds  # left, bottom, right, top

                    # Check if raster fully covers shapefile
                    if (raster_bounds.left > bayan_bounds[0] or
                        raster_bounds.bottom > bayan_bounds[1] or
                        raster_bounds.right < bayan_bounds[2] or
                        raster_bounds.top < bayan_bounds[3]):
                        print(f"[Coverage Error] {city} raster '{key}' does not fully cover the shapefile")
            except Exception as e:
                print(f"[Error] Could not open {city} raster '{key}' for coverage check: {e}")

## ------------------------------------------- MAIN PROCESSING -------------------------------------------

print(f"Starting 120m grid analysis...")

# Build all tasks
tasks = []

for city, dataset in datasets.items():
    epsg = dataset["epsg"]
    bayan_gdf = gpd.read_file(dataset["shp"]).to_crs(epsg)
    bayan_gdf["grid_id"] = (
        (bayan_gdf.geometry.centroid.x // 120).astype(int).astype(str) + "_" +
        (bayan_gdf.geometry.centroid.y // 120).astype(int).astype(str)
    )

    for source in all_raster_keys:
        if source in dataset:
            raster_path = dataset[source]

            for i, row in bayan_gdf.iterrows():
                grid_geom = row.geometry
                grid_id = row["grid_id"]
                args = (city, raster_path, grid_geom, grid_id, epsg, source)

                # Determine task type based on source
                if source in binary_raster_keys:
                    tasks.append(("binary", args))
                elif source in fractional_raster_keys:
                    tasks.append(("fractional", args))

print(f"Processing {len(tasks)} tasks using {N_CPUS} CPUs...")

# Process all tasks in parallel
with Pool(processes=N_CPUS) as pool:
    results = list(tqdm(
        pool.imap_unordered(worker, tasks),
        total=len(tasks),
        desc="Processing 120m grids"
    ))

# Save results
if results:
    df = pd.DataFrame(results)
    output_path = os.path.join(OUT_DIR, "Canopy_120m_Grids.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved results to: {output_path}")
    print(f"Total records: {len(df)}")
else:
    print("No results generated")

print("Processing complete!")
