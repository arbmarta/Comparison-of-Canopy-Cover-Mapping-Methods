import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping

## --------------------------------------------------- CONFIGURATION ---------------------------------------------------
#region

cities = ['Vancouver', 'Winnipeg', 'Ottawa']
utm_crs_map = {
    'Vancouver': 'EPSG:32610',
    'Winnipeg':  'EPSG:32614',
    'Ottawa':    'EPSG:32618'
}

# Paths
input_dir = 'Imagery/LiDAR Canopy Height Model'
output_dir = os.path.join(input_dir, 'Reprojected')
boundary_dir = 'City Boundaries'
os.makedirs(output_dir, exist_ok=True)

#endregion

## ------------------------------------------------ LOAD CITY BOUNDARIES -----------------------------------------------
#region

boundaries = {
    city: gpd.read_file(os.path.join(boundary_dir, f'{city}.shp')).to_crs(utm_crs_map[city])
    for city in cities
}

#endregion

## -------------------------------------------- REPROJECT AND CLIP THE CHMs --------------------------------------------
#region

for city in cities:
    src_path = os.path.join(input_dir, f'{city}_chm.tif')
    dst_path = os.path.join(output_dir, f'{city}_chm_utm_clipped.tif')
    boundary = boundaries[city]
    geometry = [mapping(geom) for geom in boundary.geometry]

    try:
        with rasterio.open(src_path) as src:
            # Reproject transform and metadata
            transform, width, height = calculate_default_transform(
                src.crs, utm_crs_map[city], src.width, src.height, *src.bounds
            )
            out_meta = src.meta.copy()
            out_meta.update({
                'crs': utm_crs_map[city],
                'transform': transform,
                'width': width,
                'height': height
            })

            # Reproject raster data
            dst_array = np.empty((src.count, height, width), dtype=src.dtypes[0])
            for i in range(src.count):
                reproject(
                    source=rasterio.band(src, i + 1),
                    destination=dst_array[i],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=utm_crs_map[city],
                    resampling=Resampling.bilinear
                )

        # Clip using MemoryFile
        with rasterio.MemoryFile() as memfile:
            with memfile.open(**out_meta) as temp_dst:
                temp_dst.write(dst_array)
                clipped, clipped_transform = mask(temp_dst, geometry, crop=True, filled=True)
                clipped = np.where(clipped < 1, np.nan, clipped)

        # Final output metadata
        out_meta.update({
            'height': clipped.shape[1],
            'width': clipped.shape[2],
            'transform': clipped_transform
        })

        # Save clipped raster
        with rasterio.open(dst_path, 'w', **out_meta) as dest:
            dest.write(clipped)

        print(f"✅ {city} CHM reprojected, clipped, and saved to: {dst_path}")

    except Exception as e:
        print(f"❌ Failed to process {city} CHM: {e}")

#endregion
