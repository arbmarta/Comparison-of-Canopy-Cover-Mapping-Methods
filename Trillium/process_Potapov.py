from shapely.geometry import box, mapping
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio
import os

city_info = {
    "Vancouver": {
        "bounds": (-123.445035, 48.918190, -122.446536, 49.550882),
        "utm_epsg": 32610  # UTM zone 10N
    },
    "Ottawa": {
        "bounds": (-76.381667, 45.052941, -75.140385, 45.682254),
        "utm_epsg": 32618  # UTM zone 18N
    },
    "Winnipeg": {
        "bounds": (-97.580902, 49.533571, -96.635600, 50.175443),
        "utm_epsg": 32614  # UTM zone 14N
    }
}

input_raster = "original_raster.tif"
output_dir = "clipped_rasters"
os.makedirs(output_dir, exist_ok=True)

for city, info in city_info.items():
    bounds = info["bounds"]
    epsg = info["utm_epsg"]
    geom = [mapping(box(*bounds))]

    with rasterio.open(input_raster) as src:
        # Clip to bounding box
        clipped_img, clipped_transform = mask(src, geom, crop=True)
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "height": clipped_img.shape[1],
            "width": clipped_img.shape[2],
            "transform": clipped_transform
        })

        # Save intermediate clipped raster (still in WGS84)
        clipped_path = os.path.join(output_dir, f"{city}_clipped.tif")
        with rasterio.open(clipped_path, "w", **clipped_meta) as dst:
            dst.write(clipped_img)

        # Reopen and reproject
        with rasterio.open(clipped_path) as clipped_src:
            dst_crs = f"EPSG:{epsg}"
            transform, width, height = calculate_default_transform(
                clipped_src.crs, dst_crs,
                clipped_src.width, clipped_src.height,
                *clipped_src.bounds
            )
            reprojected_meta = clipped_src.meta.copy()
            reprojected_meta.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height
            })

            reprojected_path = os.path.join(output_dir, f"{city}_clipped_utm.tif")
            with rasterio.open(reprojected_path, "w", **reprojected_meta) as dst:
                for i in range(1, clipped_src.count + 1):
                    reproject(
                        source=rasterio.band(clipped_src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=clipped_src.transform,
                        src_crs=clipped_src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )
