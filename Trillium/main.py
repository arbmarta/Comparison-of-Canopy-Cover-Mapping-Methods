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

# Constants
OUT_DIR = "/scratch/arbmarta/Outputs"
os.makedirs(OUT_DIR, exist_ok=True)
GRID_SIZES = [120, 60, 40, 30, 20, 10]

# Input boundaries and rasters
rasters = {
    "Vancouver": {
        "LiDAR": '/scratch/arbmarta/LiDAR/Vancouver LiDAR.tif',
        "bayan": gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs("EPSG:32610"),
        "epsg": "EPSG:32610",
        "buildings": gpd.read_file('/scratch/arbmarta/Buildings/Vancouver Buildings.fgb').to_crs("EPSG:32610")
    },
    "Winnipeg": {
        "LiDAR": '/scratch/arbmarta/LiDAR/Winnipeg LiDAR.tif',
        "bayan": gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs("EPSG:32614"),
        "epsg": "EPSG:32614",
        "buildings": gpd.read_file('/scratch/arbmarta/Buildings/Winnipeg Buildings.shp').to_crs("EPSG:32614")
    },
    "Ottawa": {
        "LiDAR": '/scratch/arbmarta/LiDAR/Ottawa LiDAR.tif',
        "bayan": gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs("EPSG:32618"),
        "epsg": "EPSG:32618",
        "buildings": gpd.read_file('/scratch/arbmarta/Buildings/Ottawa Buildings.shp').to_crs("EPSG:32618")
    }
}

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

# Add these keys to each city's raster config:
rasters[city]["esri"] = f"/scratch/arbmarta/ESRI/{city} ESRI.tif"
rasters[city]["terrascope"] = f"/scratch/arbmarta/Terrascope/{city} Terrascope.tif"

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

def compute_fragmentation_metrics(polygon_df, grid_area):
    if polygon_df.empty:
        return pd.Series({"PAFRAC": 0, "nLSI": 0, "CAI_AM": 0, "LSI": 0, "ED": 0})
    areas = polygon_df.geometry.area
    perimeters = polygon_df.geometry.length
    logs = np.log(perimeters) / np.log(areas)
    pafrac = 2 * logs.mean() if (areas > 0).all() and (perimeters > 0).all() else 0
    E = perimeters.sum()
    A = areas.sum()
    lsi = E / (4 * np.sqrt(A)) if A > 0 else 0
    max_lsi = (2 * np.sqrt(grid_area)) / (4 * np.sqrt(A)) if A > 0 else 1
    nlsi = (lsi - 1) / (max_lsi - 1) if max_lsi != 1 else 0
    cores = polygon_df.geometry.buffer(-1)
    cores = cores[cores.area > 0]
    cai_am = cores.area.sum() / A if not cores.empty else 0
    ed = E / grid_area * 10000
    return pd.Series({"PAFRAC": pafrac, "nLSI": nlsi, "CAI_AM": cai_am, "LSI": lsi, "ED": ed})

def subdivide_grids(grid_gdf, cell_size):
    new_grids = []
    for _, row in grid_gdf.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        x_start = int(minx // cell_size) * cell_size
        y_start = int(miny // cell_size) * cell_size
        for x in np.arange(x_start, maxx, cell_size):
            for y in np.arange(y_start, maxy, cell_size):
                new_geom = box(x, y, x + cell_size, y + cell_size)
                if new_geom.intersects(row.geometry):
                    clipped = new_geom.intersection(row.geometry)
                    grid_id = f"{int(x // cell_size)}_{int(y // cell_size)}"
                    new_grids.append({
                        "geometry": clipped,
                        "grid_id": grid_id,
                        "grid_cell_size": cell_size
                    })
    return gpd.GeoDataFrame(new_grids, geometry="geometry", crs=grid_gdf.crs)

def process_lidar_grid(args):
    city, raster_path, geom, grid_id, epsg, size = args
    result = {"city": city, "grid_id": grid_id, "grid_cell_size": size}
    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, [geom], crop=True)
            polygons = raster_to_polygons(out_image, out_transform, src.nodata)
            if polygons.empty:
                result.update(dict.fromkeys(["total_m2", "polygon_count", "total_perimeter",
                                             "percent_cover", "mean_patch_size", "patch_density",
                                             "area_cv", "perimeter_cv", "PAFRAC", "nLSI",
                                             "CAI_AM", "LSI", "ED"], 0))
            else:
                polygons.set_crs(src.crs, inplace=True)
                polygons = polygons.to_crs(epsg)
                clipped = gpd.overlay(polygons, gpd.GeoDataFrame(geometry=[geom], crs=epsg), how="intersection")
                clipped["m2"] = clipped.geometry.area
                clipped["perimeter"] = clipped.geometry.length
                result.update({
                    "total_m2": clipped["m2"].sum(),
                    "polygon_count": len(clipped),
                    "total_perimeter": clipped["perimeter"].sum(),
                    "percent_cover": clipped["m2"].sum() / (size ** 2) * 100,
                    "mean_patch_size": clipped["m2"].mean(),
                    "patch_density": len(clipped) / (size ** 2),
                    "area_cv": clipped["m2"].std() / clipped["m2"].mean() if clipped["m2"].mean() > 0 else 0,
                    "perimeter_cv": clipped["perimeter"].std() / clipped["perimeter"].mean() if clipped["perimeter"].mean() > 0 else 0
                })
                result.update(compute_fragmentation_metrics(clipped, size ** 2))
    except:
        result.update(dict.fromkeys(["total_m2", "polygon_count", "total_perimeter",
                                     "percent_cover", "mean_patch_size", "patch_density",
                                     "area_cv", "perimeter_cv", "PAFRAC", "nLSI",
                                     "CAI_AM", "LSI", "ED"], 0))
    return result

def main_land_cover():
    all_results = []
    for city, config in rasters.items():
        for size in GRID_SIZES:
            subdivided = subdivide_grids(config["bayan"], size)
            meta = subdivided.drop(columns="geometry")
            for source in ["esri", "terrascope"]:
                raster_path = config[source]
                tasks = [(city, source, raster_path, row.geometry, row.grid_id, meta.iloc[[i]].reset_index(drop=True), config["epsg"], size) for i, row in subdivided.iterrows()]
                with Pool(os.cpu_count()) as pool:
                    results = pool.map(process_land_cover_grid, tasks)
                all_results.extend(results)
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUT_DIR, "Land_Cover_Tree_Grass.csv"), index=False)
    
def process_building_grid(args):
    grid_row, buildings_gdf, city, size = args
    grid_id = grid_row['grid_id']
    grid_geom = grid_row['geometry']
    try:
        clipped = buildings_gdf[buildings_gdf.intersects(grid_geom)].copy()
        if clipped.empty:
            return {"city": city, "grid_id": grid_id, "grid_cell_size": size, "built_area_total_m2": 0, "number_of_buildings": 0, "mean_building_size": 0}
        clipped['geometry'] = clipped.geometry.intersection(grid_geom)
        clipped = clipped[~clipped.is_empty & clipped.is_valid]
        clipped['area_m2'] = clipped.geometry.area
        return {
            'city': city,
            'grid_id': grid_id,
            'grid_cell_size': size,
            'built_area_total_m2': clipped['area_m2'].sum(),
            'number_of_buildings': len(clipped),
            'mean_building_size': clipped['area_m2'].mean()
        }
    except:
        return {"city": city, "grid_id": grid_id, "grid_cell_size": size, "built_area_total_m2": None, "number_of_buildings": None, "mean_building_size": None}

def process_percent_cover_grid(args):
    city, source, raster_path, grid, grid_id, grid_meta, epsg, size = args
    try:
        with rasterio.open(raster_path) as src:
            if src.crs != epsg:
                src = reproject_raster_in_memory(src, epsg)
            out_image, out_transform = mask(src, [grid], crop=True)
            polygons = raster_to_polygons(out_image, out_transform, src.nodata)
            total_m2 = polygons.to_crs(epsg).geometry.area.sum() if not polygons.empty else 0
    except:
        total_m2 = 0
    result = grid_meta.copy()
    result.update({
        "total_m2": total_m2,
        "percent_cover": total_m2 / (size ** 2) * 100,
        "city": city,
        "source": source,
        "grid_id": grid_id,
        "grid_cell_size": size
    })
    return result
    
def process_land_cover_grid_esri(args):
    city, source, raster_path, grid, grid_id, grid_meta, epsg, size = args
    try:
        with rasterio.open(raster_path) as src:
            if src.crs != epsg:
                src = reproject_raster_in_memory(src, epsg)
            out_image, out_transform = mask(src, [grid], crop=True)
            band = out_image[0]
            valid_mask = (band == 2)
band = np.where(valid_mask, band, np.nan)
total_pixels = np.count_nonzero(~np.isnan(band))
            tree_pixels = np.count_nonzero(band == 2)  # ESRI: tree class is 2  # assuming class 1 = trees
            grass_pixels = 0  # Not applicable for ESRI  # adjust if needed
    except:
        tree_pixels = 0
        grass_pixels = 0
        total_pixels = 1  # avoid div by zero

    result = grid_meta.copy()
    result.update({
        "tree_percent": (tree_pixels / total_pixels) * 100,
        "grass_percent": (grass_pixels / total_pixels) * 100,
        "city": city,
        "source": source,
        "grid_id": grid_id,
        "grid_cell_size": size
    })
    return result

def main_lidar():
    all_results = []
    for city, config in rasters.items():
        raster = config["LiDAR"]
        epsg = config["epsg"]
        for size in GRID_SIZES:
            subdivided = subdivide_grids(config["bayan"], size)
            tasks = [(city, raster, row.geometry, row.grid_id, epsg, size) for _, row in subdivided.iterrows()]
            with Pool(os.cpu_count()) as pool:
                results = list(tqdm(pool.imap_unordered(process_lidar_grid, tasks), total=len(tasks)))
            all_results.extend(results)
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUT_DIR, "LiDAR.csv"), index=False)

def main_buildings():
    all_results = []
    for city, config in rasters.items():
        buildings = config["buildings"]
        for size in GRID_SIZES:
            subdivided = subdivide_grids(config["bayan"], size)
            tasks = [(row, buildings, city, size) for _, row in subdivided.iterrows()]
            with Pool(192) as pool:
                results = pool.map(process_building_grid, tasks)
            all_results.extend(results)
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUT_DIR, "Building_Area_By_Grid.csv"), index=False)

def main_percent_cover():
    all_results = []
    for city, config in rasters.items():
        for size in GRID_SIZES:
            subdivided = subdivide_grids(config["bayan"], size)
            meta = subdivided.drop(columns="geometry")
            for source in ["ETH", "Meta", "Potapov"]:
                raster_path = bayan_configs[city][source]
                tasks = [(city, source, raster_path, row.geometry, row.grid_id, meta.iloc[[i]].reset_index(drop=True), config["epsg"], size) for i, row in subdivided.iterrows()]
                with Pool(os.cpu_count()) as pool:
                    results = pool.map(process_percent_cover_grid, tasks)
                all_results.extend(results)
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUT_DIR, "All_Cities_Percent_Cover.csv"), index=False)

if __name__ == "__main__":
    main_lidar()
    main_buildings()
    main_percent_cover()
    main_land_cover()
