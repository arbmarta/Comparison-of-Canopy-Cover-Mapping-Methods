import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from multiprocessing import Pool

# Input boundaries (already in UTM)
van_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs("EPSG:32610")
wpg_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs("EPSG:32614")
ott_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs("EPSG:32618")

# Rasters
rasters = {
    "Vancouver": {
        "ETH": '/scratch/arbmarta/ETH/Vancouver ETH.tif',
        "Meta": '/scratch/arbmarta/Meta/Vancouver Meta.tif',
        "LiDAR": '/scratch/arbmarta/LiDAR/Vancouver LiDAR.tif',
        "bayan": van_bayan,
        "epsg": "EPSG:32610",
    },
    "Winnipeg": {
        "ETH": '/scratch/arbmarta/ETH/Winnipeg ETH.tif',
        "Meta": '/scratch/arbmarta/Meta/Winnipeg Meta.tif',
        "LiDAR": '/scratch/arbmarta/LiDAR/Winnipeg LiDAR.tif',
        "bayan": wpg_bayan,
        "epsg": "EPSG:32614",
    },
    "Ottawa": {
        "ETH": '/scratch/arbmarta/ETH/Ottawa ETH.tif',
        "Meta": '/scratch/arbmarta/Meta/Ottawa Meta.tif',
        "LiDAR": '/scratch/arbmarta/LiDAR/Ottawa LiDAR.tif',
        "bayan": ott_bayan,
        "epsg": "EPSG:32618",
    }
}

OUT_DIR = "/scratch/arbmarta/Outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def raster_to_polygons(masked_arr, out_transform, nodata=None):
    band = masked_arr[0]

    # Build a validity mask: non-nodata, non-NaN, and > 0
    valid = ~np.isnan(band) if np.issubdtype(band.dtype, np.floating) else np.ones_like(band, dtype=bool)
    if nodata is not None:
        valid &= (band != nodata)

    # Apply threshold: only values ≥ 2
    mask_vals = valid & (band >= 2)

    # Polygonize using rasterio.features.shapes
    results = [
        (shape(geom), float(val))
        for geom, val in shapes(band, mask=mask_vals, transform=out_transform)
        if val >= 2  # redundant but safe
    ]

    if not results:
        return gpd.GeoDataFrame(columns=["value", "geometry"], geometry=[], crs=None)

    geoms, vals = zip(*results)
    return gpd.GeoDataFrame({"value": vals}, geometry=list(geoms), crs=None)

def process_city_source(args):
    city, source, raster_path, bayan_gdf, utm_epsg = args

    with rasterio.open(raster_path) as src:
        masked, transform = mask(src, bayan_gdf.to_crs(src.crs).geometry, crop=True)
        polygons = raster_to_polygons(masked, transform, src.nodata)
        if not polygons.empty:
            polygons.set_crs(src.crs, inplace=True)
            if polygons.crs != utm_epsg:
                polygons = polygons.to_crs(utm_epsg)
            clipped = gpd.overlay(polygons, bayan_gdf, how="intersection")
            clipped["m2"] = clipped.geometry.area
            summary = clipped.groupby("grid_id")["m2"].sum().reset_index(name="total_m2")
        else:
            summary = gpd.pd.DataFrame(columns=["grid_id", "total_m2"])

    # Create grid_id in bayan_gdf and merge
    bayan_gdf = bayan_gdf.copy()
    bayan_gdf["grid_id"] = (
        (bayan_gdf.geometry.centroid.x // 120).astype(int).astype(str) + "_" +
        (bayan_gdf.geometry.centroid.y // 120).astype(int).astype(str)
    )
    bayan_meta = bayan_gdf.drop(columns="geometry").drop_duplicates(subset="grid_id")

    merged = bayan_meta.merge(summary, on="grid_id", how="left")
    merged["total_m2"] = merged["total_m2"].fillna(0)
    merged["percent_cover"] = (merged["total_m2"] / 14400) * 100

    # Add traceability columns
    merged["source"] = source
    merged["city"] = city

    return merged

def main():
    tasks = []
    for city, config in rasters.items():
        for source, path in config.items():
            if source in ["bayan", "epsg"]:
                continue
            tasks.append((
                city, 
                source, 
                path, 
                config["bayan"].to_crs(config["epsg"]), 
                config["epsg"]))

    with Pool(processes=9) as pool:
        pool.map(process_city_source, tasks)

if __name__ == "__main__":
    main()
