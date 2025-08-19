import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm

# Input boundaries
van_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs("EPSG:32610")
wpg_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs("EPSG:32614")
ott_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs("EPSG:32618")

rasters = {
    "Vancouver": {"LiDAR": '/scratch/arbmarta/LiDAR/Vancouver LiDAR.tif', "bayan": van_bayan, "epsg": "EPSG:32610"},
    "Winnipeg": {"LiDAR": '/scratch/arbmarta/LiDAR/Winnipeg LiDAR.tif', "bayan": wpg_bayan, "epsg": "EPSG:32614"},
    "Ottawa": {"LiDAR": '/scratch/arbmarta/LiDAR/Ottawa LiDAR.tif', "bayan": ott_bayan, "epsg": "EPSG:32618"},
}

OUT_DIR = "/scratch/arbmarta/Outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def raster_to_polygons(masked_arr, out_transform, nodata=None):
    band = masked_arr[0]
    valid = ~np.isnan(band) if np.issubdtype(band.dtype, np.floating) else np.ones_like(band, dtype=bool)
    if nodata is not None:
        valid &= (band != nodata)
    mask_vals = valid & (band >= 2)
    results = [(shape(geom), float(val)) for geom, val in shapes(band, mask=mask_vals, transform=out_transform) if val >= 2]
    if not results:
        return gpd.GeoDataFrame(columns=["value", "geometry"], geometry=[], crs=None)
    geoms, vals = zip(*results)
    return gpd.GeoDataFrame({"value": vals}, geometry=list(geoms), crs=None)

def compute_fragmentation_metrics(polygon_df, grid_area=14400):
    if polygon_df.empty:
        return pd.Series({
            "PAFRAC": 0, "nLSI": 0, "CAI_AM": 0, "LSI": 0, "ED": 0
        })

    areas = polygon_df.geometry.area
    perimeters = polygon_df.geometry.length

    if (areas > 0).all() and (perimeters > 0).all():
        logs = np.log(perimeters) / np.log(areas)
        pafrac = 2 * logs.mean()
    else:
        pafrac = 0

    E = perimeters.sum()
    A = areas.sum()
    lsi = E / (4 * np.sqrt(A)) if A > 0 else 0
    max_lsi = (2 * np.sqrt(grid_area)) / (4 * np.sqrt(A)) if A > 0 else 1
    nlsi = (lsi - 1) / (max_lsi - 1) if max_lsi != 1 else 0

    buffer_width = 1
    cores = polygon_df.geometry.buffer(-buffer_width)
    cores = cores[cores.area > 0]
    if not cores.empty:
        core_areas = cores.area
        cai_am = core_areas.sum() / A
    else:
        cai_am = 0

    ed = E / grid_area * 10000

    return pd.Series({
        "PAFRAC": pafrac,
        "nLSI": nlsi,
        "CAI_AM": cai_am,
        "LSI": lsi,
        "ED": ed
    })

def process_grid(args):
    city, raster_path, grid_geom, grid_id, epsg = args
    result = {"grid_id": grid_id, "city": city}
    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, [grid_geom], crop=True)
            polygons = raster_to_polygons(out_image, out_transform, src.nodata)
            if polygons.empty:
                result.update({
                    "total_m2": 0, "polygon_count": 0, "total_perimeter": 0,
                    "percent_cover": 0, "mean_patch_size": 0, "patch_density": 0,
                    "area_cv": 0, "perimeter_cv": 0,
                    "PAFRAC": 0, "nLSI": 0, "CAI_AM": 0, "LSI": 0, "ED": 0
                })
            else:
                polygons.set_crs(src.crs, inplace=True)
                polygons = polygons.to_crs(epsg)
                clipped = gpd.overlay(polygons, gpd.GeoDataFrame(geometry=[grid_geom], crs=epsg), how="intersection")
                clipped["m2"] = clipped.geometry.area
                clipped["perimeter"] = clipped.geometry.length

                result["total_m2"] = total_m2 = clipped["m2"].sum()
                result["polygon_count"] = poly_ct = len(clipped)
                result["total_perimeter"] = clipped["perimeter"].sum()
                result["percent_cover"] = (total_m2 / 14400) * 100
                result["mean_patch_size"] = total_m2 / poly_ct if poly_ct else 0
                result["patch_density"] = poly_ct / 14400
                result["area_cv"] = clipped["m2"].std() / clipped["m2"].mean() if clipped["m2"].mean() > 0 else 0
                result["perimeter_cv"] = clipped["perimeter"].std() / clipped["perimeter"].mean() if clipped["perimeter"].mean() > 0 else 0

                frag_metrics = compute_fragmentation_metrics(clipped)
                result.update(frag_metrics)
    except:
        result.update({
            "total_m2": 0, "polygon_count": 0, "total_perimeter": 0,
            "percent_cover": 0, "mean_patch_size": 0, "patch_density": 0,
            "area_cv": 0, "perimeter_cv": 0,
            "PAFRAC": 0, "nLSI": 0, "CAI_AM": 0, "LSI": 0, "ED": 0
        })
    return result

def main():
    tasks = []
    for city, config in rasters.items():
        epsg = config["epsg"]
        bayan = config["bayan"].to_crs(epsg)
        raster = config["LiDAR"]
        bayan["grid_id"] = ((bayan.geometry.centroid.x // 120).astype(int).astype(str) + "_" +
                             (bayan.geometry.centroid.y // 120).astype(int).astype(str))
        for i, row in bayan.iterrows():
            tasks.append((city, raster, row.geometry, row.grid_id, epsg))

    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_grid, tasks), total=len(tasks)))

    df = pd.DataFrame(results)
    cols = ["city", "grid_id", "total_m2", "percent_cover", "polygon_count",
            "mean_patch_size", "patch_density", "total_perimeter",
            "area_cv", "perimeter_cv", "PAFRAC", "nLSI", 
            "CAI_AM", "LSI", "ED"]
    df = df[cols]
    df.to_csv(os.path.join(OUT_DIR, "LiDAR.csv"), index=False)
    print("Saved: LiDAR.csv")

if __name__ == "__main__":
    main()
