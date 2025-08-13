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
    valid = ~np.isnan(band) if np.issubdtype(band.dtype, np.floating) else np.ones_like(band, dtype=bool)
    if nodata is not None:
        valid &= (band != nodata)
    mask_vals = valid & (band > 0)
    geoms, vals = zip(*[
        (shape(geom), float(val)) 
        for geom, val in shapes(band, mask=mask_vals, transform=out_transform)
        if val > 0
    ])
    return gpd.GeoDataFrame({"value": vals}, geometry=list(geoms), crs=None)

def process_city_source(args):
    city, source, raster_path, bayan_gdf, utm_epsg = args
    with rasterio.open(raster_path) as src:
        masked, transform = mask(src, bayan_gdf.to_crs(src.crs).geometry, crop=True)
        polygons = raster_to_polygons(masked, transform, src.nodata)
        polygons.set_crs(src.crs, inplace=True)

    # Reproject to UTM and overlay with bayan grid
    polygons = polygons.to_crs(utm_epsg)
    bayan_gdf = bayan_gdf.to_crs(utm_epsg)
    clipped = gpd.overlay(polygons, bayan_gdf, how="intersection")

    # Calculate area and percent
    clipped["m2"] = clipped.geometry.area
    if 'grid_id' not in clipped.columns:
        clipped["grid_id"] = (clipped.geometry.centroid.x // 120).astype(int).astype(str) + "_" + (clipped.geometry.centroid.y // 120).astype(int).astype(str)

    summary = clipped.groupby("grid_id")["m2"].sum().reset_index(name="total_m2")
    summary["percent_cover"] = (summary["total_m2"] / 14400) * 100

    # Write CSV
    out_csv = os.path.join(OUT_DIR, f"{city}_{source}_percent_cover.csv")
    summary.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

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
