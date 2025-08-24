import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from multiprocessing import Pool
from tqdm import tqdm
import os

# Output path
OUT_PATH = "/scratch/arbmarta/Outputs/CSVs/Building_Area_By_Grid.csv"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Grid sizes to process (starting with 120 then subdividing)
grid_sizes = [120, 60, 40, 30, 20, 10]

CITY_BUILDINGS = {}

def init_worker(van, wpg, ott):
    global CITY_BUILDINGS
    CITY_BUILDINGS = {
        "Vancouver": van,
        "Winnipeg": wpg,
        "Ottawa": ott,
    }

# Step 1: Add grid_id for 120 m Bayan grids
def add_grid_id(grid_gdf):
    centroids = grid_gdf.geometry.centroid
    grid_gdf["grid_id"] = (
        (centroids.x // 120).astype(int).astype(str) + "_" +
        (centroids.y // 120).astype(int).astype(str) + "_120"
    )
    return grid_gdf

# Step 2: Subdivide each 120 m cell into smaller grids
def subdivide_grid(grid_row, size, city, epsg):
    tasks = []
    geom = grid_row.geometry
    minx, miny, maxx, maxy = geom.bounds
    nx = int((maxx - minx) // size)
    ny = int((maxy - miny) // size)
    for i in range(nx):
        for j in range(ny):
            subgeom = box(minx + i*size, miny + j*size, minx + (i+1)*size, miny + (j+1)*size)
            if not geom.intersects(subgeom):
                continue
            clipped_geom = geom.intersection(subgeom)
            if clipped_geom.is_empty:
                continue
            tasks.append((clipped_geom, city, size))
    return tasks

# Step 3: Process a grid cell (120 m or subgrid)
def process_grid(args):
    geom, city, size = args
    buildings_gdf = CITY_BUILDINGS[city]
    c = geom.centroid
    grid_id = f"{int(c.x // size)}_{int(c.y // size)}_{size}"

    try:
        clipped = buildings_gdf[buildings_gdf.intersects(geom)].copy()
        if clipped.empty:
            return {
                'city': city,
                'grid_id': grid_id,
                'Grid Cell Size': size,
                'built_area_total_m2': 0,
                'number_of_buildings': 0,
                'mean_building_size': 0
            }

        clipped['geometry'] = clipped.geometry.intersection(geom)
        clipped = clipped[~clipped.is_empty & clipped.is_valid]
        clipped['area_m2'] = clipped.geometry.area

        return {
            'city': city,
            'grid_id': grid_id,
            'Grid Cell Size': size,
            'built_area_total_m2': clipped['area_m2'].sum(),
            'number_of_buildings': len(clipped),
            'mean_building_size': clipped['area_m2'].mean()
        }
    except Exception:
        return {
            'city': city,
            'grid_id': grid_id,
            'Grid Cell Size': size,
            'built_area_total_m2': None,
            'number_of_buildings': None,
            'mean_building_size': None
        }

# Step 4: Main execution
def main():
    # Load Bayan grids
    van_bayan = add_grid_id(gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs("EPSG:32610"))
    wpg_bayan = add_grid_id(gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs("EPSG:32614"))
    ott_bayan = add_grid_id(gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs("EPSG:32618"))

    # Load buildings
    van_buildings = gpd.read_file('/scratch/arbmarta/Buildings/Vancouver Buildings.fgb').to_crs("EPSG:32610")
    wpg_buildings = gpd.read_file('/scratch/arbmarta/Buildings/Winnipeg Buildings.shp').to_crs("EPSG:32614")
    ott_buildings = gpd.read_file('/scratch/arbmarta/Buildings/Ottawa Buildings.shp').to_crs("EPSG:32618")

    # Collect tasks for all cities and grid sizes
    tasks = []
    for row in van_bayan.itertuples(index=False):
        tasks.append((row.geometry, "Vancouver", 120))
        for size in grid_sizes[1:]:
            tasks.extend(subdivide_grid(row, size, "Vancouver", "EPSG:32610"))

    for row in wpg_bayan.itertuples(index=False):
        tasks.append((row.geometry, "Winnipeg", 120))
        for size in grid_sizes[1:]:
            tasks.extend(subdivide_grid(row, size, "Winnipeg", "EPSG:32614"))

    for row in ott_bayan.itertuples(index=False):
        tasks.append((row.geometry, "Ottawa", 120))
        for size in grid_sizes[1:]:
            tasks.extend(subdivide_grid(row, size, "Ottawa", "EPSG:32618"))

    # Parallel processing
    with Pool(processes=192, initializer=init_worker,
              initargs=(van_buildings, wpg_buildings, ott_buildings)) as pool:
        results = list(tqdm(pool.imap_unordered(process_grid, tasks), total=len(tasks)))

    # Export
    df = pd.DataFrame(results)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")

if __name__ == '__main__':
    main()
