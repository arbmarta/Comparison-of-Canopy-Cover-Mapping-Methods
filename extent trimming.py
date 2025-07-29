# This file uses city boundary data of the three cities to trim the raster data to the AOI.
# The data is reprojected into the local UTM zones.

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import mapping
from rasterio.transform import array_bounds
import numpy as np

## ------------------------------------------- IMPORT CITY BOUNDARY FILES USING AOI --------------------------------------------------
#region

# Dictionary of city names, shapefile paths, and their UTM EPSG codes
city_info = {
    "Vancouver": {"path": "City Boundaries/Vancouver.shp", "epsg": 32610},
    "Winnipeg": {"path": "City Boundaries/Winnipeg.shp", "epsg": 32614},
    "Ottawa": {"path": "City Boundaries/Ottawa.shp", "epsg": 32618}
}

# Read and reproject city boundaries
city_boundaries = {}
for city, info in city_info.items():
    gdf = gpd.read_file(info["path"])
    gdf_utm = gdf.to_crs(epsg=info["epsg"])
    city_boundaries[city] = gdf_utm

    print(f"{city} reprojected to CRS: {gdf_utm.crs}")

    # Plot
    gdf_utm.plot(edgecolor='black', facecolor='lightblue', figsize=(8, 8))
    plt.title(f"{city} City Boundary (UTM Zone)")
    plt.axis('off')
    plt.show()

#endregion

## ------------------------------------------ FUNCTION: PROCESS & PLOT RASTER --------------------------------------------------
#region

def process_and_plot_rasters(file_dict, title_prefix):
    result = {}
    for city, paths in file_dict.items():
        print(f"\nProcessing {city} {title_prefix} rasters...")

        # Merge rasters if more than one
        with rasterio.open(paths[0]) as first_src:
            src_crs = first_src.crs

        src_files = [rasterio.open(path) for path in paths]
        mosaic, mosaic_transform = merge(src_files)
        for src in src_files:
            src.close()

        # Get target CRS
        target_crs = city_boundaries[city].crs

        # Reproject the mosaic to match the city's UTM CRS
        bounds = array_bounds(mosaic.shape[1], mosaic.shape[2], mosaic_transform)
        dst_transform, width, height = calculate_default_transform(
            src_crs, target_crs, mosaic.shape[2], mosaic.shape[1], *bounds
        )
        dst_array = np.empty((1, height, width), dtype=mosaic.dtype)

        reproject(
            source=mosaic,
            destination=dst_array,
            src_transform=mosaic_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

        # Clip to polygon (not just bbox)
        shapes = [mapping(geom) for geom in city_boundaries[city].geometry]
        clipped_array, clipped_transform = mask(
            dataset={
                "data": dst_array[0],
                "transform": dst_transform,
                "crs": target_crs
            },
            shapes=shapes,
            crop=True,
            filled=True,
            nodata=0
        )

        result[city] = {
            "array": clipped_array[0],
            "transform": clipped_transform,
            "crs": target_crs
        }

        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(clipped_array[0], cmap='viridis')
        plt.title(f"{city} {title_prefix} Canopy Height Model (Clipped to Polygon)")
        plt.colorbar(label="Height (m)")
        plt.axis('off')
        plt.show()

    return result

#endregion

## -------------------------------------------------- IMPORT ETH CHMs --------------------------------------------------
#region

eth_files = {
    "Vancouver": ["Imagery/Lang CHM/Vancouver - East ETH.tif", "Imagery/Lang CHM/Vancouver - West ETH.tif"],
    "Winnipeg": ["Imagery/Lang CHM/Winnipeg ETH.tif"],
    "Ottawa": ["Imagery/Lang CHM/Ottawa ETH.tif"]
}

eth_rasters = process_and_plot_rasters(eth_files, "ETH")

#endregion

## -------------------------------------------------- IMPORT META CHMs --------------------------------------------------
#region

meta_files = {
    "Vancouver": ["Imagery/Meta CHM/Vancouver Meta.tif"],
    "Winnipeg": ["Imagery/Meta CHM/Winnipeg Meta.tif"],
    "Ottawa": ["Imagery/Meta CHM/Ottawa Meta.tif"]
}

meta_rasters = process_and_plot_rasters(meta_files, "Meta")

#endregion
