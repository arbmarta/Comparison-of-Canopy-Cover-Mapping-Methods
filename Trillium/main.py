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
        "Terrascope 2021": "/scratch/arbmarta/Binary Rasters/Vancouver 2021 Terrascope.tif"
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
        "Terrascope 2021": "/scratch/arbmarta/Binary Rasters/Winnipeg 2021 Terrascope.tif"
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
        "Terrascope 2021": "/scratch/arbmarta/Binary Rasters/Ottawa 2021 Terrascope.tif"
    }
}

raster_keys = ["ETH", "Meta", "Potapov", "DW_10m", "ESRI", "Terrascope 2020", "Terrascope 2021"]
grid_sizes = [120, 60, 40, 30, 20, 10]

## ------------------------------------------- FUNCTIONS -------------------------------------------

def get_epsg_int(crs):
    if crs is None:
        return None
    return int(crs.to_string().split(":")[1])

def raster_to_polygons(masked_arr, out_transform, nodata=None, crs=None):
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

def process_subgrid(args):
    """Process a single subgrid for canopy cover analysis"""
    city, raster_path, subgeom, grid_id, epsg, cell_area, size, source = args

    c = subgeom.centroid
    sub_id = f"{int(c.x // size)}_{int(c.y // size)}_{size}"

    result = {
        "grid_id": grid_id,          # original grid ID
        "subgrid_id": sub_id,        # new subgrid ID
        "city": city,
        "source": source,
        "Grid Cell Size": size
    }
    
    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, [subgeom], crop=True)
            polygons = raster_to_polygons(out_image, out_transform, src.nodata, crs=src.crs)
            
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
                    gpd.GeoDataFrame(geometry=[subgeom], crs=epsg),
                    how="intersection"
                )
                clipped["m2"] = clipped.geometry.area

                total_m2 = clipped["m2"].sum()
                result.update({
                    "total_m2": total_m2,
                    "polygon_count": len(clipped),
                    "percent_cover": (total_m2 / cell_area) * 100
                })
                
    except Exception as e:
        print(f"Error processing {city} {source}: {e}")
        result.update({
            "total_m2": 0,
            "percent_cover": 0,
            "polygon_count": 0
        })

    return result

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

## ------------------------------------------- CHECK RASTER COVERAGE -------------------------------------------

print("Checking raster coverage...")

for city, info in datasets.items():
    bayan_gdf = gpd.read_file(info["shp"]).to_crs(info["epsg"])
    bayan_bounds = bayan_gdf.total_bounds  # [minx, miny, maxx, maxy]

    for key in raster_keys:
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

print("Starting subgrid analysis...")

# Build all tasks
tasks = []

for city, dataset in datasets.items():
    epsg = dataset["epsg"]
    bayan_gdf = gpd.read_file(dataset["shp"]).to_crs(epsg)
    bayan_gdf["grid_id"] = (
        (bayan_gdf.geometry.centroid.x // 120).astype(int).astype(str) + "_" +
        (bayan_gdf.geometry.centroid.y // 120).astype(int).astype(str)
    )

    for source in raster_keys:
        if source in dataset:
            raster_path = dataset[source]

            for size in grid_sizes:
                cell_area = size * size

                for i, row in bayan_gdf.iterrows():
                    subgeom = row.geometry
                    args = (city, raster_path, subgeom, row["grid_id"], epsg, cell_area, size, source)
                    tasks.append(args)

print(f"Processing {len(tasks)} tasks using {N_CPUS} CPUs...")

# Process all tasks in parallel
with Pool(processes=N_CPUS) as pool:
    results = list(tqdm(
        pool.imap_unordered(process_subgrid, tasks),
        total=len(tasks),
        desc="Processing subgrids"
    ))

# Save results
if results:
    df = pd.DataFrame(results)
    output_path = os.path.join(OUT_DIR, "All_Methods_Percent_Cover.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved results to: {output_path}")
    print(f"Total records: {len(df)}")
else:
    print("❌ No results generated")

print("Processing complete!")
