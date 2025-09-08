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

# Constants
N_CPUS = 192
OUT_DIR = "/scratch/arbmarta/Outputs/CSVs"
os.makedirs(OUT_DIR, exist_ok=True)

# Input boundaries
van_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs("EPSG:32610")
wpg_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs("EPSG:32614")
ott_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs("EPSG:32618")

rasters = {
    "Vancouver": {"LiDAR": '/scratch/arbmarta/Binary Rasters/Vancouver LiDAR.tif', "bayan": van_bayan, "epsg": "EPSG:32610"},
    "Winnipeg": {"LiDAR": '/scratch/arbmarta/Binary Rasters/Winnipeg LiDAR.tif', "bayan": wpg_bayan, "epsg": "EPSG:32614"},
    "Ottawa": {"LiDAR": '/scratch/arbmarta/Binary Rasters/Ottawa LiDAR.tif', "bayan": ott_bayan, "epsg": "EPSG:32618"},
}

# Single grid size only
GRID_SIZE = 120
CELL_AREA = GRID_SIZE * GRID_SIZE

def raster_to_polygons(masked_arr, out_transform, nodata=None):
    """Convert binary raster to polygons - for LiDAR binary data (1=canopy, 0=no canopy)"""
    band = masked_arr[0]
    valid = ~np.isnan(band) if np.issubdtype(band.dtype, np.floating) else np.ones_like(band, dtype=bool)
    if nodata is not None:
        valid &= (band != nodata)

    # For binary LiDAR: 1 = canopy, 0 = no canopy
    mask_vals = valid & (band == 1)
    results = [(shape(geom), int(val)) for geom, val in shapes(band, mask=mask_vals, transform=out_transform) if val == 1]

    if not results:
        return gpd.GeoDataFrame(columns=["value", "geometry"], geometry=[], crs=None)
    geoms, vals = zip(*results)
    return gpd.GeoDataFrame({"value": vals}, geometry=list(geoms), crs=None)

def compute_fragmentation_metrics(polygon_df, grid_area=CELL_AREA, edge_depth=10):
    """
    Compute stable fragmentation metrics for a grid:
      - LSI: Landscape Shape Index
      - CAI_AM: Area-Weighted Mean Core Area Index (percent)
    
    Parameters:
        polygon_df: GeoDataFrame with patch polygons
        grid_area: float, area of the grid (default 120*120)
        edge_depth: float, buffer distance in meters to define "core"
    
    Returns:
        pd.Series with keys: 'LSI', 'CAI_AM'
    """
    if polygon_df.empty:
        return pd.Series({"LSI": 0, "CAI_AM": 0})

    # Compute total perimeter and area
    areas = polygon_df.geometry.area
    perimeters = polygon_df.geometry.length
    total_area = areas.sum()
    total_perimeter = perimeters.sum()

    # Landscape Shape Index (LSI)
    lsi = total_perimeter / (4 * np.sqrt(total_area)) if total_area > 0 else 0

    # Compute core areas for each patch
    cai_values = []
    weights = []
    for patch in polygon_df.geometry:
        patch_area = patch.area
        if patch_area <= 0:
            continue

        # inward buffer to define core
        core = patch.buffer(-edge_depth)
        core_area = core.area if not core.is_empty else 0.0

        cai_patch = (core_area / patch_area) * 100  # percent of patch that is core
        cai_values.append(cai_patch)
        weights.append(patch_area)

    # Area-weighted mean core area index
    cai_am = np.average(cai_values, weights=weights) if cai_values else 0

    return pd.Series({"LSI": lsi, "CAI_AM": cai_am})

def process_grid(args):
    """Process a single 120m grid for LiDAR canopy analysis"""
    city, raster_path, grid_geom, grid_id, epsg = args

    result = {
        "grid_id": grid_id,
        "city": city,
        "grid_size_m": GRID_SIZE
    }

    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, [grid_geom], crop=True)
            polygons = raster_to_polygons(out_image, out_transform, src.nodata)

            if polygons.empty:
                result.update({
                    "total_m2": 0, "patch_count": 0, "total_perimeter": 0,
                    "percent_cover": 0, "mean_patch_size": 0,
                    "area_cv": 0, "perimeter_cv": 0,
                    "CAI_AM": 0, "LSI": 0
                })
            else:
                polygons.set_crs(src.crs, inplace=True)
                polygons = polygons.to_crs(epsg)
                clipped = gpd.overlay(polygons, gpd.GeoDataFrame(geometry=[grid_geom], crs=epsg), how="intersection")
                clipped["m2"] = clipped.geometry.area
                clipped["perimeter"] = clipped.geometry.length

                total_m2 = clipped["m2"].sum()
                poly_ct = len(clipped)

                result.update({
                    "total_m2": total_m2,
                    "patch_count": patch_ct,
                    "total_perimeter": clipped["perimeter"].sum(),
                    "percent_cover": (total_m2 / CELL_AREA) * 100,
                    "mean_patch_size": total_m2 / patch_ct if patch_ct else 0,
                    "area_cv": clipped["m2"].std() / clipped["m2"].mean() if clipped["m2"].mean() > 0 else 0,
                    "perimeter_cv": clipped["perimeter"].std() / clipped["perimeter"].mean() if clipped["perimeter"].mean() > 0 else 0
                })
                result.update(compute_fragmentation_metrics(clipped, grid_area=CELL_AREA))

    except Exception as e:
        print(f"Error processing {city} grid: {e}")
        result.update({
            "total_m2": 0, "patch_count": 0, "total_perimeter": 0,
            "percent_cover": 0, "mean_patch_size": 0,
            "area_cv": 0, "perimeter_cv": 0,
            "CAI_AM": 0, "LSI": 0
        })
    return result

def main():
    print(f"Building processing tasks for 120m grids...")
    tasks = []

    for city, config in rasters.items():
        epsg = config["epsg"]
        raster = config["LiDAR"]
        bayan = config["bayan"].to_crs(epsg)
        bayan["grid_id"] = ((bayan.geometry.centroid.x // 120).astype(int).astype(str) + "_" +
                            (bayan.geometry.centroid.y // 120).astype(int).astype(str))

        for _, row in bayan.iterrows():
            grid_geom = row.geometry
            grid_id = row.grid_id
            tasks.append((city, raster, grid_geom, grid_id, epsg))

    print(f"Processing {len(tasks)} tasks using {N_CPUS} CPUs...")

    with Pool(processes=N_CPUS) as pool:
        results = list(tqdm(pool.imap_unordered(process_grid, tasks), total=len(tasks), desc="Processing LiDAR 120m grids"))

    print("Saving results...")
    df = pd.DataFrame(results)
    cols = ["city", "grid_id", "grid_size_m", "total_m2", "percent_cover", "patch_count",
            "mean_patch_size", "total_perimeter",
            "area_cv", "perimeter_cv", "CAI_AM", "LSI"]
    df = df[cols]

    output_path = os.path.join(OUT_DIR, "LiDAR_120m_Grid_Canopy_Metrics.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Total records: {len(df)}")

if __name__ == "__main__":
    main()
