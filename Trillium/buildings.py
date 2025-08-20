import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from multiprocessing import Pool
import os

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
