# This file uses city boundary data of the three cities to trim the raster data to the AOI.
# The data is reprojected into the local UTM zones.

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
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

        # Merge if multiple rasters
        with rasterio.open(paths[0]) as first_src:
            src_crs = first_src.crs

        src_files = [rasterio.open(path) for path in paths]
        mosaic, mosaic_transform = merge(src_files)
        for src in src_files:
            src.close()

        # Reproject to city UTM CRS
        target_crs = city_boundaries[city].crs
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

        # Crop to city bounding box
        bbox = city_boundaries[city].total_bounds
        crop_window = from_bounds(*bbox, transform=dst_transform)
        cropped_array = dst_array[0][
            int(crop_window.row_off):int(crop_window.row_off + crop_window.height),
            int(crop_window.col_off):int(crop_window.col_off + crop_window.width)
        ]

        result[city] = {
            "array": cropped_array,
            "transform": dst_transform,
            "crs": target_crs
        }

        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(cropped_array, cmap='viridis')
        plt.title(f"{city} {title_prefix} Canopy Height Model (Reprojected & Cropped)")
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
