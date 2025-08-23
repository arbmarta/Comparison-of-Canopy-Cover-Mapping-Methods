import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape, box
from multiprocessing import Pool
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.io
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

# ---------- Step 1: Add grid_id to each Bayan cell ----------
def add_grid_id(grid_gdf):
    centroids = grid_gdf.geometry.centroid
    grid_gdf["grid_id"] = (
        (centroids.x // 120).astype(int).astype(str) + "_" +
        (centroids.y // 120).astype(int).astype(str)
    )
    return grid_gdf

# ---------- Step 2: Process one grid cell ----------
def process_grid(args):
    grid_row, buildings_gdf, city = args
    grid_id = grid_row['grid_id']
    grid_geom = grid_row['geometry']

    try:
        # Clip buildings to grid cell
        clipped = buildings_gdf[buildings_gdf.intersects(grid_geom)].copy()
        if clipped.empty:
            return {
                'city': city,
                'grid_id': grid_id,
                'built_area_total_m2': 0,
                'number_of_buildings': 0,
                'mean_building_size': 0
            }

        # Intersect and clean
        clipped['geometry'] = clipped.geometry.intersection(grid_geom)
        clipped = clipped[~clipped.is_empty & clipped.is_valid]
        clipped['area_m2'] = clipped.geometry.area

        return {
            'city': city,
            'grid_id': grid_id,
            'built_area_total_m2': clipped['area_m2'].sum(),
            'number_of_buildings': len(clipped),
            'mean_building_size': clipped['area_m2'].mean()
        }
    except Exception as e:
        return {
            'city': city,
            'grid_id': grid_id,
            'built_area_total_m2': None,
            'number_of_buildings': None,
            'mean_building_size': None
        }

# ---------- Step 3: Main processing ----------
def main():
    # Read and reproject all grids
    van_bayan = add_grid_id(gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs("EPSG:32610"))
    wpg_bayan = add_grid_id(gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs("EPSG:32614"))
    ott_bayan = add_grid_id(gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs("EPSG:32618"))

    # Read and reproject all buildings
    van_buildings = gpd.read_file('/scratch/arbmarta/Buildings/Vancouver Buildings.fgb').to_crs("EPSG:32610")
    wpg_buildings = gpd.read_file('/scratch/arbmarta/Buildings/Winnipeg Buildings.shp').to_crs("EPSG:32614")
    ott_buildings = gpd.read_file('/scratch/arbmarta/Buildings/Ottawa Buildings.shp').to_crs("EPSG:32618")

    # Build processing tasks (one per grid cell)
    tasks = []
    for _, row in van_bayan.iterrows():
        tasks.append((row, van_buildings, "Vancouver"))
    for _, row in wpg_bayan.iterrows():
        tasks.append((row, wpg_buildings, "Winnipeg"))
    for _, row in ott_bayan.iterrows():
        tasks.append((row, ott_buildings, "Ottawa"))

    # Run parallel processing
    with Pool(processes=192) as pool:
        results = pool.map(process_grid, tasks)

    # Export results to CSV
    df = pd.DataFrame(results)
    df.to_csv('/scratch/arbmarta/Outputs/Building_Area_By_Grid.csv', index=False)

if __name__ == '__main__':
    main()
    
# Function to reproject raster in memory
def reproject_raster_in_memory(src, target_epsg):
    transform, width, height = calculate_default_transform(
        src.crs, f"EPSG:{target_epsg}", src.width, src.height, *src.bounds
    )
    profile = src.profile.copy()
    profile.update({
        'crs': f"EPSG:{target_epsg}",
        'transform': transform,
        'width': width,
        'height': height
    })

    memfile = rasterio.io.MemoryFile()
    with memfile.open(**profile) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=f"EPSG:{target_epsg}",
                resampling=Resampling.nearest
            )
        return memfile.open()

# Constants
OUT_DIR = "/scratch/arbmarta/Outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Input boundaries
bayan_configs = {
    "Vancouver": {
        "shp": "/scratch/arbmarta/Trinity/Vancouver/TVAN.shp",
        "epsg": "EPSG:32610",
        "ETH": "/scratch/arbmarta/ETH/Vancouver_ETH_32610.tif",
        "Meta": "/scratch/arbmarta/Meta/Vancouver Meta.tif",
        "Potapov": "/scratch/arbmarta/Potapov/Vancouver Potapov.tif"
    },
    "Winnipeg": {
        "shp": "/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp",
        "epsg": "EPSG:32614",
        "ETH": "/scratch/arbmarta/ETH/Winnipeg_ETH_32614.tif",
        "Meta": "/scratch/arbmarta/Meta/Winnipeg Meta.tif",
        "Potapov": "/scratch/arbmarta/Potapov/Winnipeg Potapov.tif"
    },
    "Ottawa": {
        "shp": "/scratch/arbmarta/Trinity/Ottawa/TOTT.shp",
        "epsg": "EPSG:32618",
        "ETH": "/scratch/arbmarta/ETH/Ottawa_ETH_32618.tif",
        "Meta": "/scratch/arbmarta/Meta/Ottawa Meta.tif",
        "Potapov": "/scratch/arbmarta/Potapov/Ottawa Potapov.tif"
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
            if src.crs != epsg:
                src = reproject_raster_in_memory(src, epsg)

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

        for source in ["ETH", "Meta", "Potapov"]:
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
        df = pd.concat(results, ignore_index=True)
        df.to_csv(os.path.join(OUT_DIR, "All_Cities_Percent_Cover.csv"), index=False)
        print("Saved: All_Cities_Percent_Cover.csv")

if __name__ == "__main__":
    main()
