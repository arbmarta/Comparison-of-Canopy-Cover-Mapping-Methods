import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import box, shape
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

grid_sizes = [120, 60, 40, 30, 20, 10]  # Subgrid sizes

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

def compute_fragmentation_metrics(polygon_df, grid_area=14400):
    """Compute landscape fragmentation metrics"""
    if polygon_df.empty:
        return pd.Series({ "PAFRAC": 0, "nLSI": 0, "CAI_AM": 0, "LSI": 0, "ED": 0 })

    areas = polygon_df.geometry.area
    perimeters = polygon_df.geometry.length

    pafrac = 2 * (np.log(perimeters) / np.log(areas)).mean() if (areas > 0).all() and (perimeters > 0).all() else 0
    E = perimeters.sum()
    A = areas.sum()
    lsi = E / (4 * np.sqrt(A)) if A > 0 else 0
    max_lsi = (2 * np.sqrt(grid_area)) / (4 * np.sqrt(A)) if A > 0 else 1
    nlsi = (lsi - 1) / (max_lsi - 1) if max_lsi != 1 else 0

    cores = polygon_df.geometry.buffer(-1)
    cores = cores[cores.area > 0]
    cai_am = cores.area.sum() / A if not cores.empty else 0

    ed = E / grid_area * 10000
    return pd.Series({ "PAFRAC": pafrac, "nLSI": nlsi, "CAI_AM": cai_am, "LSI": lsi, "ED": ed })

def process_subgrid(args):
    """Process a single subgrid for LiDAR canopy analysis"""
    city, raster_path, subgeom, grid_id, epsg, cell_area, size = args

    c = subgeom.centroid
    sub_id = f"{int(c.x // size)}_{int(c.y // size)}_{size}"
    
    result = {
        "grid_id": grid_id,       # parent grid ID
        "subgrid_id": sub_id,     # subgrid ID
        "city": city,
        "Grid Cell Size": size
    }
    
    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, [subgeom], crop=True)
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
                clipped = gpd.overlay(polygons, gpd.GeoDataFrame(geometry=[subgeom], crs=epsg), how="intersection")
                clipped["m2"] = clipped.geometry.area
                clipped["perimeter"] = clipped.geometry.length

                total_m2 = clipped["m2"].sum()
                poly_ct = len(clipped)
                
                result.update({
                    "total_m2": total_m2,
                    "polygon_count": poly_ct,
                    "total_perimeter": clipped["perimeter"].sum(),
                    "percent_cover": (total_m2 / cell_area) * 100,
                    "mean_patch_size": total_m2 / poly_ct if poly_ct else 0,
                    "patch_density": poly_ct / cell_area,
                    "area_cv": clipped["m2"].std() / clipped["m2"].mean() if clipped["m2"].mean() > 0 else 0,
                    "perimeter_cv": clipped["perimeter"].std() / clipped["perimeter"].mean() if clipped["perimeter"].mean() > 0 else 0
                })
                result.update(compute_fragmentation_metrics(clipped, grid_area=cell_area))
                
    except Exception as e:
        print(f"Error processing {city} subgrid: {e}")
        result.update({
            "total_m2": 0, "polygon_count": 0, "total_perimeter": 0,
            "percent_cover": 0, "mean_patch_size": 0, "patch_density": 0,
            "area_cv": 0, "perimeter_cv": 0,
            "PAFRAC": 0, "nLSI": 0, "CAI_AM": 0, "LSI": 0, "ED": 0
        })
    return result

def main():
    print("Building processing tasks...")
    tasks = []

    for city, config in rasters.items():
        epsg = config["epsg"]
        raster = config["LiDAR"]
        bayan = config["bayan"].to_crs(epsg)
        bayan["grid_id"] = ((bayan.geometry.centroid.x // 120).astype(int).astype(str) + "_" +
                            (bayan.geometry.centroid.y // 120).astype(int).astype(str))

        for _, row in bayan.iterrows():
            parent_geom = row.geometry
            parent_id = row.grid_id
            cell_area = 120 * 120
            tasks.append((city, raster, parent_geom, parent_id, epsg, cell_area, 120))  # full grid
    
            minx, miny, maxx, maxy = parent_geom.bounds
            for size in grid_sizes:
                nx = int((maxx - minx) // size)
                ny = int((maxy - miny) // size)
                for i in range(nx):
                    for j in range(ny):
                        subgeom = box(minx + i*size, miny + j*size,
                                      minx + (i+1)*size, miny + (j+1)*size)
                        if not parent_geom.intersects(subgeom):
                            continue
                        subgeom = parent_geom.intersection(subgeom)
                        if subgeom.is_empty:
                            continue
                        tasks.append((city, raster, subgeom, parent_id, epsg, size*size, size))

    print(f"Processing {len(tasks)} tasks using {N_CPUS} CPUs...")

    with Pool(processes=N_CPUS) as pool:
        results = list(tqdm(pool.imap_unordered(process_subgrid, tasks), total=len(tasks), desc="Processing LiDAR subgrids"))

    print("Saving results...")
    df = pd.DataFrame(results)
    cols = ["city", "grid_id", "subgrid_id", "Grid Cell Size", "total_m2", "percent_cover", "polygon_count",
        "mean_patch_size", "patch_density", "total_perimeter",
        "area_cv", "perimeter_cv", "PAFRAC", "nLSI", "CAI_AM", "LSI", "ED"]
    df = df[cols]
    
    output_path = os.path.join(OUT_DIR, "lidar_and_canopy_metrics.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Total records: {len(df)}")

if __name__ == "__main__":
    main()
