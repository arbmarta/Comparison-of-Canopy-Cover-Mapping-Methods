import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from multiprocessing import Pool

# Constants
OUT_DIR = "/scratch/arbmarta/Outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Input boundaries
bayan_configs = {
    "Vancouver": {
        "shp": "/scratch/arbmarta/Trinity/Vancouver/TVAN.shp",
        "epsg": "EPSG:32610",
        "LiDAR": "/scratch/arbmarta/LiDAR/Vancouver LiDAR.tif"
    },
    "Winnipeg": {
        "shp": "/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp",
        "epsg": "EPSG:32614",
        "LiDAR": "/scratch/arbmarta/LiDAR/Winnipeg LiDAR.tif"
    },
    "Ottawa": {
        "shp": "/scratch/arbmarta/Trinity/Ottawa/TOTT.shp",
        "epsg": "EPSG:32618",
        "LiDAR": "/scratch/arbmarta/LiDAR/Ottawa LiDAR.tif"
    }
}

def raster_to_polygons(masked_arr, out_transform, nodata=None):
    band = masked_arr[0]
    valid = ~np.isnan(band) if np.issubdtype(band.dtype, np.floating) else np.ones_like(band, dtype=bool)
    if nodata is not None:
        valid &= (band != nodata)
    mask_vals = valid & (band >= 2)
    results = [
        (shape(geom), float(val))
        for geom, val in shapes(band, mask=mask_vals, transform=out_transform)
        if val >= 2
    ]
    if not results:
        return gpd.GeoDataFrame(columns=["value", "geometry"], geometry=[], crs=None)
    geoms, vals = zip(*results)
    return gpd.GeoDataFrame({"value": vals}, geometry=list(geoms), crs=None)

def process_grid(args):
    city, source, raster_path, grid, grid_id, grid_meta, epsg = args

    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, [grid], crop=True)
            polygons = raster_to_polygons(out_image, out_transform, src.nodata)
            if not polygons.empty:
                polygons.set_crs(src.crs, inplace=True)
                polygons = polygons.to_crs(epsg)
                clipped = gpd.overlay(polygons, gpd.GeoDataFrame(geometry=[grid], crs=epsg), how="intersection")
                total_m2 = clipped.geometry.area.sum()
            else:
                total_m2 = 0
    except Exception:
        total_m2 = 0

    percent_cover = (total_m2 / 14400) * 100
    result = grid_meta.copy()
    result["total_m2"] = total_m2
    result["percent_cover"] = percent_cover
    result["city"] = city
    result["source"] = source
    result["grid_id"] = grid_id
    return result

def main():
    tasks = []

    for city, config in bayan_configs.items():
        epsg = config["epsg"]
        bayan_gdf = gpd.read_file(config["shp"]).to_crs(epsg)
        bayan_gdf["grid_id"] = (
            (bayan_gdf.geometry.centroid.x // 120).astype(int).astype(str) + "_" +
            (bayan_gdf.geometry.centroid.y // 120).astype(int).astype(str)
        )
        bayan_meta = bayan_gdf.drop(columns="geometry")

        for source in ["LiDAR"]:
            raster_path = config[source]
            for i, row in bayan_gdf.iterrows():
                tasks.append((
                    city,
                    source,
                    raster_path,
                    row.geometry,
                    row["grid_id"],
                    bayan_meta.iloc[[i]].reset_index(drop=True),
                    epsg
                ))

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_grid, tasks)

    if results:
        df = gpd.pd.concat(results, ignore_index=True)
        df.to_csv(os.path.join(OUT_DIR, "All_Cities_Percent_Cover.csv"), index=False)
        print("Saved: All_Cities_Percent_Cover.csv")

if __name__ == "__main__":
    main()
