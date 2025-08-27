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
from rasterio.windows import from_bounds
from tqdm import tqdm

# Constants
OUT_DIR = "/scratch/arbmarta/Outputs/CSVs"
os.makedirs(OUT_DIR, exist_ok=True)

## ------------------------------------------- INPUT DATASETS -------------------------------------------

datasets = {
    "Vancouver": {
        "shp": "/scratch/arbmarta/Trinity/Vancouver/TVAN.shp",
        "epsg": 32610,
        "ETH": "/scratch/arbmarta/CHMs/ETH/Vancouver ETH.tif",
        "Meta": "/scratch/arbmarta/CHMs/Meta/Vancouver Meta.tif",
        "Potapov": "/scratch/arbmarta/CHMs/Potapov/Vancouver Potapov.tif",
        "GLCF": "/scratch/arbmarta/CCPs/GLCF/Vancouver GLCF.tif",
        "GLOBMAPFTC": "/scratch/arbmarta/CCPs/GLOBMAPFTC/Vancouver GLOBMAPFTC.tif",
        "DW_10m": "/scratch/arbmarta/Land Cover/DW_2020/Vancouver DW.tif", # Trees indicated by Value 1
        "ESRI": "/scratch/arbmarta/Land Cover/ESRI/Vancouver ESRI.tif", # Trees indicated by Value 2
        "Terrascope 2020": "/scratch/arbmarta/Land Cover/Terrascope/Vancouver 2020 Terrascope.tif", # Trees indicated by Value 10
        "Terrascope 2021": "/scratch/arbmarta/Land Cover/Terrascope/Vancouver 2021 Terrascope.tif"
    },
    "Winnipeg": {
        "shp": "/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp",
        "epsg": 32614,
        "ETH": "/scratch/arbmarta/CHMs/ETH/Winnipeg ETH.tif",
        "Meta": "/scratch/arbmarta/CHMs/Meta/Winnipeg Meta.tif",
        "Potapov": "/scratch/arbmarta/CHMs/Potapov/Winnipeg Potapov.tif",
        "GLCF": "/scratch/arbmarta/CCPs/GLCF/Winnipeg GLCF.tif",
        "GLOBMAPFTC": "/scratch/arbmarta/CCPs/GLOBMAPFTC/Winnipeg GLOBMAPFTC.tif",
        "DW_10m": "/scratch/arbmarta/Land Cover/DW_2020/Winnipeg DW.tif",
        "ESRI": "/scratch/arbmarta/Land Cover/ESRI/Winnipeg ESRI.tif",
        "Terrascope 2020": "/scratch/arbmarta/Land Cover/Terrascope/Winnipeg 2020 Terrascope.tif",
        "Terrascope 2021": "/scratch/arbmarta/Land Cover/Terrascope/Winnipeg 2021 Terrascope.tif"
    },
    "Ottawa": {
        "shp": "/scratch/arbmarta/Trinity/Ottawa/TOTT.shp",
        "epsg": 32618,
        "ETH": "/scratch/arbmarta/CHMs/ETH/Ottawa ETH.tif",
        "Meta": "/scratch/arbmarta/CHMs/Meta/Ottawa Meta.tif",
        "Potapov": "/scratch/arbmarta/CHMs/Potapov/Ottawa Potapov.tif",
        "GLCF": "/scratch/arbmarta/CCPs/GLCF/Ottawa GLCF.tif",
        "GLOBMAPFTC": "/scratch/arbmarta/CCPs/GLOBMAPFTC/Ottawa GLOBMAPFTC.tif",
        "DW_10m": "/scratch/arbmarta/Land Cover/DW_2020/Ottawa DW.tif",
        "ESRI": "/scratch/arbmarta/Land Cover/ESRI/Ottawa ESRI.tif",
        "Terrascope 2020": "/scratch/arbmarta/Land Cover/Terrascope/Ottawa 2020 Terrascope.tif",
        "Terrascope 2021": "/scratch/arbmarta/Land Cover/Terrascope/Ottawa 2021 Terrascope.tif"
    }
}

raster_keys = ["ETH", "Meta", "Potapov", "GLCF", "GLOBMAPFTC",
               "DW_10m", "ESRI", "Terrascope 2020", "Terrascope 2021"]

## ------------------------------------------- FUNCTIONS TO CHECK PROJECTION OF RASTERS AND SHAPEFILES -------------------------------------------

def get_epsg_int(crs):
    if crs is None:
        return None
    return int(crs.to_string().split(":")[1])

for city, info in datasets.items():
    target_epsg = info["epsg"]
    
    # Shapefile EPSG
    gdf = gpd.read_file(info["shp"])
    shp_epsg = get_epsg_int(gdf.crs)
    if shp_epsg is None:
        print(f"[Error] {city} shapefile has no CRS defined")
    elif shp_epsg != target_epsg:
        print(f"[Mismatch] {city} shapefile EPSG: {shp_epsg} != {target_epsg}")
    else:
        print(f"[OK] {city} shapefile EPSG matches {target_epsg}")

    # Raster EPSG
    all_match = True
    for key in raster_keys:
        if key in info:
            raster_path = info[key]
            try:
                with rasterio.open(raster_path) as src:
                    raster_epsg = get_epsg_int(src.crs)
                    if raster_epsg is None:
                        print(f"[Mismatch] {city} raster '{key}' has no CRS defined")
                        all_match = False
                    elif raster_epsg != target_epsg:
                        print(f"[Mismatch] {city} raster '{key}' EPSG: {raster_epsg} != {target_epsg}")
                        all_match = False
            except Exception as e:
                print(f"[Error] Could not open {city} raster '{key}': {e}")
                all_match = False

    if all_match:
        print(f"[OK] All rasters for {city} have the correct EPSG {target_epsg}")

## ------------------------------------------- CHECK RASTER COVERAGE OF BAYAN SHAPEFILES -------------------------------------------

for city, info in datasets.items():
    bayan_gdf = gpd.read_file(info["shp"]).to_crs(info["epsg"])
    bayan_bounds = bayan_gdf.total_bounds  # [minx, miny, maxx, maxy]

    for key in raster_keys:
        if key in info:
            raster_path = info[key]
            try:
                with rasterio.open(raster_path) as src:
                    raster_bounds = src.bounds  # left, bottom, right, top

                    # Check if raster fully covers shapefile
                    if (raster_bounds.left > bayan_bounds[0] or
                        raster_bounds.bottom > bayan_bounds[1] or
                        raster_bounds.right < bayan_bounds[2] or
                        raster_bounds.top < bayan_bounds[3]):
                        print(f"[Coverage Error] {city} raster '{key}' does not fully cover the bayan shapefile")
            except Exception as e:
                print(f"[Error] Could not open {city} raster '{key}' for coverage check: {e}")

## ------------------------------------------- CONVERT CANOPY HEIGHT MODELS TO BINARY CANOPY COVER MAPS -------------------------------------------

def create_canopy_mask_from_chm(raster_path, boundary_gdf=None):
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(float)
        profile = src.profile

        canopy_mask = (data >= 2).astype(np.uint8)

        if boundary_gdf is not None:
            if boundary_gdf.crs != src.crs:
                boundary_gdf = boundary_gdf.to_crs(src.crs)
            out_image, _ = mask(src, boundary_gdf.geometry, crop=True)
            canopy_mask = (out_image[0].astype(float) >= 2).astype(np.uint8)
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
            if boundary_gdf.crs != src.crs:
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
    "Terrascope 2020": 10,
    "Terrascope 2021": 10
}

for city, dataset in datasets.items():
    boundary = gpd.read_file(dataset["shp"])

    for lc_type in ["DW_10m", "ESRI", "Terrascope 2020", "Terrascope 2021"]:
        canopy_val = canopy_values[lc_type]
        raster_path = dataset[lc_type]
        mask_array, meta = create_canopy_mask_from_lc_raster(raster_path, canopy_val, boundary)

## ------------------------------------------- PROCESS FRACTIONAL CANOPY COVER RASTERS -------------------------------------------

def process_fractional_raster(args):
    city, raster_path, subgeom, grid_id, epsg, cell_area, size, source = args

    c = subgeom.centroid
    sub_id = f"{int(c.x // size)}_{int(c.y // size)}_{size}"

    result = {
        "grid_id": grid_id,
        "subgrid_id": sub_id,
        "city": city,
        "source": source,
        "Grid Cell Size": size
    }

    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, [subgeom], crop=True)
            band = out_image[0]

            # Convert raster to polygons for calculation
            polygons = [
                (shape(geom), float(val))
                for geom, val in shapes(band, mask=(band > 0), transform=out_transform)
            ]

            if not polygons:
                result.update({
                    "total_m2": 0,
                    "percent_cover": 0,
                    "polygon_count": 0
                })
            else:
                gdf = gpd.GeoDataFrame(
                    {"value": [v for _, v in polygons]},
                    geometry=[g for g, _ in polygons],
                    crs=src.crs
                )
                gdf = gdf.to_crs(epsg)
                clipped = gpd.overlay(
                    gdf,
                    gpd.GeoDataFrame(geometry=[subgeom], crs=epsg),
                    how="intersection"
                )
                clipped["pixel_area"] = clipped.geometry.area
                clipped["canopy_m2"] = clipped["pixel_area"] * (clipped["value"] / 100)

                result["total_m2"] = clipped["canopy_m2"].sum()
                result["polygon_count"] = len(clipped)
                result["percent_cover"] = (result["total_m2"] / cell_area) * 100

    except Exception as e:
        print(f"Error in {city}, {grid_id}, {size}: {e}")
        result.update({
            "total_m2": 0,
            "percent_cover": 0,
            "polygon_count": 0
        })

    return result
    main()
