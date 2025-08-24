import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from multiprocessing import Pool
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.io

# Constants
OUT_DIR = "/scratch/arbmarta/Outputs/CSVs"
os.makedirs(OUT_DIR, exist_ok=True)

## ------------------------------------------- FUNCTIONS TO REPROJECT RASTERS IN MEMORY -------------------------------------------

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

## ------------------------------------------- INPUT BOUNDARIES -------------------------------------------

bayan_configs = {
    "Vancouver": {
        "shp": "/scratch/arbmarta/Trinity/Vancouver/TVAN.shp",
        "epsg": "EPSG:32610",
        "ETH": "/scratch/arbmarta/CHMs/ETH/Vancouver ETH.tif",
        "Meta": "/scratch/arbmarta/CHMs/Meta/Vancouver Meta.tif",
        "Potapov": "/scratch/arbmarta/CHMs/Potapov/Vancouver Potapov.tif",
        "GLCF": "/scratch/arbmarta/CCPs/GLCF/Vancouver GLCF.tif",
        "GLOBMAPFTC": "/scratch/arbmarta/CCPs/GLOBMAPFTC/Vancouver GLOBMAPFTC.tif",
        "DW_10m": "/scratch/arbmarta/Land Cover/DW_10m/Vancouver DW_10m.tif", # Trees indicated by Value 1
        "ESRI": "/scratch/arbmarta/Land Cover/ESRI/Vancouver ESRI.tif", # Trees indicated by Value 2
        "Terrascope 2020": "/scratch/arbmarta/Land Cover/Terrascope/Vancouver 2020 Terrascope.tif", # Trees indicated by Value 10
        "Terrascope 2021": "/scratch/arbmarta/Land Cover/Terrascope/Vancouver 2021 Terrascope.tif"
    },
    "Winnipeg": {
        "shp": "/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp",
        "epsg": "EPSG:32614",
        "ETH": "/scratch/arbmarta/CHMs/ETH/Winnipeg ETH.tif",
        "Meta": "/scratch/arbmarta/CHMs/Meta/Winnipeg Meta.tif",
        "Potapov": "/scratch/arbmarta/CHMs/Potapov/Winnipeg Potapov.tif",
        "GLCF": "/scratch/arbmarta/CCPs/GLCF/Winnipeg GLCF.tif",
        "GLOBMAPFTC": "/scratch/arbmarta/CCPs/GLOBMAPFTC/Winnipeg GLOBMAPFTC.tif",
        "DW_10m": "/scratch/arbmarta/Land Cover/DW_10m/Winnipeg DW_10m.tif",
        "ESRI": "/scratch/arbmarta/Land Cover/ESRI/Winnipeg ESRI.tif",
        "Terrascope 2020": "/scratch/arbmarta/Land Cover/Terrascope/Winnipeg 2020 Terrascope.tif",
        "Terrascope 2021": "/scratch/arbmarta/Land Cover/Terrascope/Winnipeg 2021 Terrascope.tif"
    },
    "Ottawa": {
        "shp": "/scratch/arbmarta/Trinity/Ottawa/TOTT.shp",
        "epsg": "EPSG:32618",
        "ETH": "/scratch/arbmarta/CHMs/ETH/Ottawa ETH.tif",
        "Meta": "/scratch/arbmarta/CHMs/Meta/Ottawa Meta.tif",
        "Potapov": "/scratch/arbmarta/CHMs/Potapov/Ottawa Potapov.tif",
        "GLCF": "/scratch/arbmarta/CCPs/GLCF/Ottawa GLCF.tif",
        "GLOBMAPFTC": "/scratch/arbmarta/CCPs/GLOBMAPFTC/Ottawa GLOBMAPFTC.tif",
        "DW_10m": "/scratch/arbmarta/Land Cover/DW_10m/Ottawa DW_10m.tif",
        "ESRI": "/scratch/arbmarta/Land Cover/ESRI/Ottawa ESRI.tif",
        "Terrascope 2020": "/scratch/arbmarta/Land Cover/Terrascope/Ottawa 2020 Terrascope.tif",
        "Terrascope 2021": "/scratch/arbmarta/Land Cover/Terrascope/Ottawa 2021 Terrascope.tif"
    }
}

## ------------------------------------------- CONVERT CANOPY HEIGHT MODELS TO BINARY CANOPY COVER MAPS -------------------------------------------

def create_canopy_mask_from_chm(raster_path, boundary_gdf=None, threshold=2):
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(float)
        profile = src.profile

        canopy_mask = (data >= threshold).astype(np.uint8)

        if boundary_gdf is not None:
            boundary_gdf = boundary_gdf.to_crs(src.crs)
            out_image, _ = mask(src, boundary_gdf.geometry, crop=True)
            canopy_mask = (out_image[0].astype(float) >= threshold).astype(np.uint8)
            profile.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": src.window_transform(
                    rasterio.windows.from_bounds(*boundary_gdf.total_bounds, transform=src.transform)
                )
            })

        return canopy_mask[np.newaxis, :, :], profile

## ------------------------------------------- CONVERT LAND COVER RASTERS TO BINARY CANOPY COVER MAPS -------------------------------------------

def create_canopy_mask_from_lc_raster(raster_path, canopy_value, boundary_gdf=None):
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        profile = src.profile

        canopy_mask = (data == canopy_value).astype(np.uint8)

        if boundary_gdf is not None:
            boundary_gdf = boundary_gdf.to_crs(src.crs)
            out_image, _ = mask(src, boundary_gdf.geometry, crop=True)
            canopy_mask = (out_image[0] == canopy_value).astype(np.uint8)
            profile.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": src.window_transform(
                    rasterio.windows.from_bounds(*boundary_gdf.total_bounds, transform=src.transform)
                )
            })

        return canopy_mask[np.newaxis, :, :], profile

canopy_values = {
    "DW_10m": 1,
    "ESRI": 2,
    "Terrascope 2020": 10
}

for city, config in bayan_configs.items():
    boundary = gpd.read_file(config["shp"])

    for lc_type in ["DW_10m", "ESRI", "Terrascope 2020"]:
        canopy_val = canopy_values[lc_type]
        raster_path = config[lc_type]
        mask_array, meta = create_canopy_mask_from_lc_raster(raster_path, canopy_val, boundary)

## ------------------------------------------- CREATE SUBGRIDS -------------------------------------------

grid_sizes = [120, 60, 40, 30, 20, 10]

def process_subgrid(args):
    city, raster_path, subgeom, grid_id, epsg, cell_area, size = args

    c = subgeom.centroid
    sub_id = f"{int(c.x // size)}_{int(c.y // size)}_{size}"

    result = {"grid_id": sub_id, "city": city, "Grid Cell Size": size}
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
                clipped = gpd.overlay(
                    polygons,
                    gpd.GeoDataFrame(geometry=[subgeom], crs=epsg),
                    how="intersection"
                )
                clipped["m2"] = clipped.geometry.area
                clipped["perimeter"] = clipped.geometry.length

                result["total_m2"] = total_m2 = clipped["m2"].sum()
                result["polygon_count"] = poly_ct = len(clipped)
                result["total_perimeter"] = clipped["perimeter"].sum()
                result["percent_cover"] = (total_m2 / cell_area) * 100
                result["mean_patch_size"] = total_m2 / poly_ct if poly_ct else 0
                result["patch_density"] = poly_ct / cell_area
                result["area_cv"] = (
                    clipped["m2"].std() / clipped["m2"].mean()
                    if clipped["m2"].mean() > 0 else 0
                )
                result["perimeter_cv"] = (
                    clipped["perimeter"].std() / clipped["perimeter"].mean()
                    if clipped["perimeter"].mean() > 0 else 0
                )
                result.update(compute_fragmentation_metrics(clipped, grid_area=cell_area))
    except:
        result.update({
            "total_m2": 0, "polygon_count": 0, "total_perimeter": 0,
            "percent_cover": 0, "mean_patch_size": 0, "patch_density": 0,
            "area_cv": 0, "perimeter_cv": 0,
            "PAFRAC": 0, "nLSI": 0, "CAI_AM": 0, "LSI": 0, "ED": 0
        })
    return result
    
## ------------------------------------------- CONVERT RASTERS TO POLYGONS -------------------------------------------

def raster_to_polygons(masked_arr, out_transform, nodata=None):
    band = masked_arr[0]
    valid = ~np.isnan(band) if np.issubdtype(band.dtype, np.floating) else np.ones_like(band, dtype=bool)
    if nodata is not None:
        valid &= (band != nodata)
    mask_vals = valid & (band == 1)  # canopy is binary: 1 = tree, 0 = not tree
    results = [
        (shape(geom), int(val))
        for geom, val in shapes(band, mask=mask_vals, transform=out_transform)
        if val == 1
    ]
    if not results:
        return gpd.GeoDataFrame(columns=["value", "geometry"], geometry=[], crs=None)
    geoms, vals = zip(*results)
    return gpd.GeoDataFrame({"value": vals}, geometry=list(geoms), crs=None)

## ------------------------------------------- MAIN CANOPY COVER ANALYZER -------------------------------------------

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

        for source in ["ETH", "Meta", "Potapov", "DW_10m", "ESRI", "Terrascope 2020"]:
            raster_path = config[source]
            for size in grid_sizes:
                for i, row in bayan_gdf.iterrows():
                    subgeom = row.geometry
                    cell_area = size * size
                    tasks.append((
                        city,
                        raster_path,
                        subgeom,
                        row["grid_id"],
                        epsg,
                        cell_area,
                        size
                    ))
                
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_subgrid, tasks)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(OUT_DIR, "All_Methods_Percent_Cover.csv"), index=False)
        print("Saved: All_Methods_Percent_Cover.csv")

if __name__ == "__main__":
    main()
