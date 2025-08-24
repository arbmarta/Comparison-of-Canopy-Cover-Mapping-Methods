import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm

# Input boundaries
van_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs("EPSG:32610")
wpg_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs("EPSG:32614")
ott_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs("EPSG:32618")

buildings = {
    "Vancouver": {"buildings": gpd.read_file('/scratch/arbmarta/Buildings/Vancouver Buildings.fgb').to_crs("EPSG:32610"), "bayan": van_bayan, "epsg": "EPSG:32610"},
    "Winnipeg": {"buildings": gpd.read_file('/scratch/arbmarta/Buildings/Winnipeg Buildings.shp').to_crs("EPSG:32614"), "bayan": wpg_bayan, "epsg": "EPSG:32614"},
    "Ottawa": {"buildings": gpd.read_file('/scratch/arbmarta/Buildings/Ottawa Buildings.shp').to_crs("EPSG:32618"), "bayan": ott_bayan, "epsg": "EPSG:32618"},
}
buildings["Winnipeg"]["buildings"]["geometry"] = buildings["Winnipeg"]["buildings"].geometry.buffer(0)

OUT_DIR = "/scratch/arbmarta/Outputs/CSVs"
os.makedirs(OUT_DIR, exist_ok=True)

grid_sizes = [120, 60, 40, 30, 20, 10]  # Subgrid sizes

def compute_building_metrics(buildings_gdf, cell_geom, cell_area):
    if buildings_gdf.empty:
        return pd.Series({
            'built_area_total_m2': 0,
            'number_of_buildings': 0,
            'mean_building_size': 0
        })

    buildings_gdf['geometry'] = buildings_gdf.geometry.intersection(cell_geom)
    buildings_gdf = buildings_gdf[~buildings_gdf.is_empty & buildings_gdf.is_valid]
    buildings_gdf['area_m2'] = buildings_gdf.geometry.area

    return pd.Series({
        'built_area_total_m2': buildings_gdf['area_m2'].sum(),
        'number_of_buildings': len(buildings_gdf),
        'mean_building_size': buildings_gdf['area_m2'].mean()
    })

def process_subgrid(args):
    city, buildings_gdf, subgeom, epsg, size = args
    c = subgeom.centroid
    sub_id = f"{int(c.x // size)}_{int(c.y // size)}_{size}"
    cell_area = size * size

    try:
        idx = buildings_gdf.sindex.query(subgeom, predicate="intersects")
        clipped = buildings_gdf.iloc[idx].copy()
        if clipped.empty:
            metrics = pd.Series({
                'built_area_total_m2': 0,
                'number_of_buildings': 0,
                'mean_building_size': 0
            })
        else:
            metrics = compute_building_metrics(clipped, subgeom, cell_area)

        return {
            'city': city,
            'grid_id': sub_id,
            'Grid Cell Size': size,
            **metrics
        }
    except Exception as e:
        return {
            'city': city,
            'grid_id': sub_id,
            'Grid Cell Size': size,
            'built_area_total_m2': None,
            'number_of_buildings': None,
            'mean_building_size': None
        }

def main():
    tasks = []

    for city, config in buildings.items():
        epsg = config["epsg"]
        buildings_gdf = config["buildings"]
        bayan = config["bayan"].to_crs(epsg)

        for _, row in bayan.iterrows():
            parent_geom = row.geometry
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
                        tasks.append((city, buildings_gdf, subgeom, epsg, size))

    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_subgrid, tasks), total=len(tasks)))

    df = pd.DataFrame(results)
    cols = ['city', 'grid_id', 'Grid Cell Size', 'built_area_total_m2', 'number_of_buildings', 'mean_building_size']
    df = df[cols]
    df.to_csv(os.path.join(OUT_DIR, "Buildings.csv"), index=False)
    print("Saved: Buildings.csv")

if __name__ == "__main__":
    main()
