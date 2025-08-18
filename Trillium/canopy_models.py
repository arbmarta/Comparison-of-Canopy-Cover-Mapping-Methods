import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
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

    # Ensure grid_id exists in the bayan grid
    if 'grid_id' not in bayan_gdf.columns:
        bayan_gdf = bayan_gdf.copy()
        bayan_gdf["grid_id"] = (
            (bayan_gdf.geometry.centroid.x // 120).astype(int).astype(str) + "_" +
            (bayan_gdf.geometry.centroid.y // 120).astype(int).astype(str)
        )

    grid_ids_df = bayan_gdf[["grid_id"]].drop_duplicates()

    with rasterio.open(raster_path) as src:
        masked, transform = mask(src, bayan_gdf.to_crs(src.crs).geometry, crop=True)
        polygons = raster_to_polygons(masked, transform, src.nodata)
        if polygons.empty:
            summary = grid_ids_df.copy()
            summary["total_m2"] = summary["polygon_count"] = summary["total_perimeter"] = 0
            summary["percent_cover"] = summary["mean_patch_size"] = summary["patch_density"] = 0
            summary["area_cv"] = summary["perimeter_cv"] = summary["IIC"] = summary["NH"] = 0
            out_csv = os.path.join(OUT_DIR, f"{city}_{source}_percent_cover.csv")
            summary.to_csv(out_csv, index=False)
            print(f"Saved (empty): {out_csv}")
            return

        polygons.set_crs(src.crs, inplace=True)
        if polygons.crs != utm_epsg:
            polygons = polygons.to_crs(utm_epsg)

    # Spatial join to assign each polygon to its grid cell by inheriting grid_id
    clipped = gpd.sjoin(polygons, bayan_gdf[["grid_id", "geometry"]], how="inner", predicate="intersects")
    clipped["m2"] = clipped.geometry.area
    clipped["perimeter"] = clipped.geometry.length

    # Group and summarize
    summary = clipped.groupby("grid_id").agg(
        total_m2=("m2", "sum"),
        polygon_count=("geometry", "count"),
        total_perimeter=("perimeter", "sum")
    ).reset_index()

    # Calculate CVs
    area_cv = clipped.groupby("grid_id")["m2"].agg(lambda x: x.std() / x.mean() if x.mean() > 0 else 0).rename("area_cv")
    perimeter_cv = clipped.groupby("grid_id")["perimeter"].agg(lambda x: x.std() / x.mean() if x.mean() > 0 else 0).rename("perimeter_cv")

    summary = summary.merge(area_cv, on="grid_id", how="left").merge(perimeter_cv, on="grid_id", how="left")

    # Additional metrics
    summary["percent_cover"] = (summary["total_m2"] / 14400) * 100
    summary["mean_patch_size"] = summary["total_m2"] / summary["polygon_count"]
    summary["patch_density"] = summary["polygon_count"] / 14400

    # Fragmentation metrics
    if source == "LiDAR":
        frag_df = (
            clipped.groupby("grid_id")
            .apply(lambda df: compute_iic_nh_from_patches(df, grid_area=14400, threshold=0.6))
            .reset_index()
        )
        summary = summary.merge(frag_df, on="grid_id", how="left")

    # Fill in missing grid cells
    summary = grid_ids_df.merge(summary, on="grid_id", how="left")
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

    summary["city"] = city
    return summary

def main():
    tasks = []
    for city, config in rasters.items():
        source = "LiDAR"  # Only processing LiDAR
        path = config[source]
        tasks.append((
            city,
            source,
            path,
            config["bayan"].to_crs(config["epsg"]),
            config["epsg"]
        ))

    with Pool(processes=os.cpu_count()) as pool:
        all_summaries = pool.map(process_city_source, tasks)

    # Filter out empty results
    all_summaries = [s for s in all_summaries if s is not None]

    # Concatenate all city summaries into one DataFrame
    merged_df = gpd.pd.concat(all_summaries, ignore_index=True)

    # Reorder columns (drop 'source', since it's all LiDAR)
    cols = ["city", "grid_id", "total_m2", "percent_cover", "polygon_count",
            "mean_patch_size", "patch_density", "total_perimeter",
            "area_cv", "perimeter_cv", "IIC", "NH"]
    merged_df = merged_df[cols]

    # Save to single CSV
    out_path = os.path.join(OUT_DIR, "LiDAR.csv")
    merged_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
