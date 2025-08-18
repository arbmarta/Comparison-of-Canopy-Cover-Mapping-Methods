import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.geometry import Point
from multiprocessing import Pool
import networkx as nx

# Input boundaries (already in UTM)
van_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs("EPSG:32610")
wpg_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs("EPSG:32614")
ott_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs("EPSG:32618")

# Rasters
rasters = {
    "Vancouver": {
        "LiDAR": '/scratch/arbmarta/LiDAR/Vancouver LiDAR.tif',
        "bayan": van_bayan,
        "epsg": "EPSG:32610",
    },
    "Winnipeg": {
        "LiDAR": '/scratch/arbmarta/LiDAR/Winnipeg LiDAR.tif',
        "bayan": wpg_bayan,
        "epsg": "EPSG:32614",
    },
    "Ottawa": {
        "LiDAR": '/scratch/arbmarta/LiDAR/Ottawa LiDAR.tif',
        "bayan": ott_bayan,
        "epsg": "EPSG:32618",
    }
}

OUT_DIR = "/scratch/arbmarta/Outputs"
os.makedirs(OUT_DIR, exist_ok=True)

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

def compute_iic_nh_from_patches(polygon_df, grid_area=14400, threshold=0.6):
    if len(polygon_df) == 0:
        return pd.Series({"IIC": 0.0, "NH": 0.0})
    areas = polygon_df["m2"].values
    n = len(areas)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            d = polygon_df.geometry.iloc[i].distance(polygon_df.geometry.iloc[j])
            if d <= threshold:
                G.add_edge(i, j)

    iic_sum = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                d_ij = 0
            else:
                try:
                    d_ij = nx.shortest_path_length(G, source=i, target=j)
                except nx.NetworkXNoPath:
                    d_ij = float('inf')
            if d_ij != float('inf'):
                iic_sum += (areas[i] * areas[j]) / (1 + d_ij)
    IIC = iic_sum / (grid_area ** 2)

    h_sum = 0
    for i in range(n):
        try:
            spl = nx.single_source_shortest_path_length(G, source=i)
        except:
            continue
        for j, l in spl.items():
            if i != j and l > 0:
                h_sum += 1 / l
    H = 0.5 * h_sum

    if n >= 2:
        H_chain = sum((n - k) / k for k in range(1, n))
        H_planar = (n * (n + 5)) / 4.0 - 3.0
        NH = (H - H_chain) / (H_planar - H_chain) if H_planar != H_chain else 0
        NH = max(0.0, min(1.0, NH))
    else:
        NH = 0.0

    return pd.Series({"IIC": IIC, "NH": NH})

def process_city_source(args):
    city, source, raster_path, bayan_gdf, utm_epsg = args
    with rasterio.open(raster_path) as src:
        masked, transform = mask(src, bayan_gdf.to_crs(src.crs).geometry, crop=True)
        polygons = raster_to_polygons(masked, transform, src.nodata)
        polygons.set_crs(src.crs, inplace=True)

    if polygons.crs != utm_epsg:
        polygons = polygons.to_crs(utm_epsg)
    clipped = gpd.overlay(polygons, bayan_gdf, how="intersection")

    all_grids = bayan_gdf.copy()
    if 'grid_id' not in all_grids.columns:
      all_grids["grid_id"] = (all_grids.geometry.centroid.x // 120).astype(int).astype(str) + "_" + (all_grids.geometry.centroid.y // 120).astype(int).astype(str)
    grid_ids_df = all_grids[["grid_id"]].drop_duplicates()

  
    clipped["m2"] = clipped.geometry.area
    if 'grid_id' not in clipped.columns:
        clipped["grid_id"] = (clipped.geometry.centroid.x // 120).astype(int).astype(str) + "_" + (clipped.geometry.centroid.y // 120).astype(int).astype(str)

    clipped["perimeter"] = clipped.geometry.length

    summary = clipped.groupby("grid_id").agg(
        total_m2=("m2", "sum"),
        polygon_count=("geometry", "count"),
        total_perimeter=("perimeter", "sum")
    ).reset_index()

    # Calculate CVs for area and perimeter
    area_cv = clipped.groupby("grid_id")["m2"].agg(lambda x: x.std() / x.mean() if x.mean() > 0 else 0).rename("area_cv")
    perimeter_cv = clipped.groupby("grid_id")["perimeter"].agg(lambda x: x.std() / x.mean() if x.mean() > 0 else 0).rename("perimeter_cv")

    summary = summary.merge(area_cv, on="grid_id", how="left").merge(perimeter_cv, on="grid_id", how="left")

    summary["percent_cover"] = (summary["total_m2"] / 14400) * 100
    summary["mean_patch_size"] = summary["total_m2"] / summary["polygon_count"]
    summary["patch_density"] = summary["polygon_count"] / 14400

    if source == "LiDAR":
        frag_df = (
            clipped.groupby("grid_id")
            .apply(lambda df: compute_iic_nh_from_patches(df, grid_area=14400, threshold=0.6))
            .reset_index()
        )
        summary = summary.merge(frag_df, on="grid_id", how="left")

    out_csv = os.path.join(OUT_DIR, f"{city}_{source}_percent_cover.csv")

    # Merge to include all grid IDs (even empty ones)
    summary = grid_ids_df.merge(summary, on="grid_id", how="left")
    
    # Fill missing values for empty grid cells
    summary.fillna({
        "total_m2": 0,
        "polygon_count": 0,
        "total_perimeter": 0,
        "percent_cover": 0,
        "mean_patch_size": 0,
        "patch_density": 0,
        "area_cv": 0,
        "perimeter_cv": 0,
        "IIC": 0,
        "NH": 0,
    }, inplace=True)
    
    summary.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

def main():
    tasks = []
    for city, config in rasters.items():
        for source, path in config.items():
            if source in ["bayan", "epsg"]:
                continue
            tasks.append((
                city,
                source,
                path,
                config["bayan"].to_crs(config["epsg"]),
                config["epsg"]
            ))
    with Pool(processes=9) as pool:
        pool.map(process_city_source, tasks)

if __name__ == "__main__":
    main()
