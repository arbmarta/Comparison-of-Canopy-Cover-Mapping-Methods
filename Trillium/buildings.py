import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from multiprocessing import Pool
from tqdm import tqdm
import os

# ---------------------- GLOBAL CONSTANTS ----------------------
CITY_BUILDINGS = {}
GRID_SIZES = [120, 60, 40, 30, 20, 10]

# ---------------------- WORKER INIT ----------------------
def init_worker(van, wpg, ott):
    global CITY_BUILDINGS
    CITY_BUILDINGS = {
        "Vancouver": van,
        "Winnipeg": wpg,
        "Ottawa": ott,
    }

# ---------------------- GRID ID FUNCTION ----------------------
def generate_grid_id(geometry, size):
    c = geometry.centroid
    return f"{int(c.x // size)}_{int(c.y // size)}_{size}"

# ---------------------- SUBDIVIDE GRID ----------------------
def subdivide_grid_cell(cell_geom, size):
    bounds = cell_geom.bounds
    minx, miny, maxx, maxy = bounds
    subcells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            subcell = box(x, y, min(x + size, maxx), min(y + size, maxy))
            if subcell.intersects(cell_geom):
                subcells.append(subcell.intersection(cell_geom))
            y += size
        x += size
    return subcells

# ---------------------- PROCESS FUNCTION ----------------------
def process_grid(args):
    city, grid_geom, size = args
    buildings_gdf = CITY_BUILDINGS[city]
    results = []

    # Use spatial index to preselect candidates
    sindex = buildings_gdf.sindex

    if size == 120:
        subgeoms = [grid_geom]  # Use the full cell
    else:
        subgeoms = subdivide_grid_cell(grid_geom, size)

    for subgeom in subgeoms:
        sub_id = generate_grid_id(subgeom, size)
        possible_matches_index = list(sindex.query(subgeom))
        candidates = buildings_gdf.iloc[possible_matches_index]
        clipped = candidates[candidates.intersects(subgeom)].copy()

        if clipped.empty:
            results.append({
                'city': city,
                'grid_id': sub_id,
                'Grid Cell Size': size,
                'built_area_total_m2': 0,
                'number_of_buildings': 0,
                'mean_building_size': 0
            })
            continue

        # Intersect and calculate areas
        clipped['geometry'] = clipped.geometry.intersection(subgeom)
        clipped = clipped[~clipped.is_empty & clipped.is_valid]
        clipped['area_m2'] = clipped.geometry.area

        results.append({
            'city': city,
            'grid_id': sub_id,
            'Grid Cell Size': size,
            'built_area_total_m2': clipped['area_m2'].sum(),
            'number_of_buildings': len(clipped),
            'mean_building_size': clipped['area_m2'].mean()
        })

    return results

# ---------------------- LOAD DATA ----------------------
print("Loading buildings and grids...")
van_buildings = gpd.read_file('/scratch/arbmarta/Buildings/Vancouver Buildings.fgb').to_crs("EPSG:32610")
wpg_buildings = gpd.read_file('/scratch/arbmarta/Buildings/Winnipeg Buildings.shp').to_crs("EPSG:32614")
ott_buildings = gpd.read_file('/scratch/arbmarta/Buildings/Ottawa Buildings.shp').to_crs("EPSG:32618")

van_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs("EPSG:32610")
wpg_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs("EPSG:32614")
ott_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs("EPSG:32618")

# ---------------------- MAIN FUNCTION ----------------------
def main():
    tasks = []
    for size in GRID_SIZES:
        for _, row in van_bayan.iterrows():
            tasks.append(("Vancouver", row.geometry, size))
        for _, row in wpg_bayan.iterrows():
            tasks.append(("Winnipeg", row.geometry, size))
        for _, row in ott_bayan.iterrows():
            tasks.append(("Ottawa", row.geometry, size))

    with Pool(processes=192, initializer=init_worker,
              initargs=(van_buildings, wpg_buildings, ott_buildings)) as pool:
        nested_results = list(tqdm(pool.imap_unordered(process_grid, tasks), total=len(tasks)))

    # Flatten nested lists
    results = [item for sublist in nested_results for item in sublist]

    # Export
    df = pd.DataFrame(results)
    out_path = '/scratch/arbmarta/Outputs/CSVs/Buildings.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
