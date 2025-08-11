import rasterio
import geopandas as gpd
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from multiprocessing import Pool

# UTM zones
van_utm = "EPSG:32610"  # Vancouver
win_utm = "EPSG:32614"  # Winnipeg
ott_utm = "EPSG:32618"  # Ottawa

# Access Bayan datasets
van_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs(van_utm)
win_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs(win_utm)
ott_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs(ott_utm)

# Raster to canopy polygons function
def raster_to_canopy_polygons(raster_path: str, boundary_gdf: gpd.GeoDataFrame,
                              utm_epsg: str, out_path: str, canopy_threshold: int = 2) -> None:
    """
    Convert a raster to binary canopy (>= canopy_threshold) clipped to boundary_gdf,
    vectorize, dissolve touching polygons, reproject to UTM, and write shapefile.
    """
    with rasterio.open(raster_path) as src:
        # Reproject city boundary to raster CRS (cheaper than warping raster)
        boundary_src = boundary_gdf.to_crs(src.crs)

        # Clip/mask raster by boundary (outside → nodata=0)
        masked, transform = mask(
            src, boundary_src.geometry, crop=True, filled=True, nodata=0
        )

        # Binary canopy: 1 if value >= threshold else 0
        binary = (masked[0] >= canopy_threshold).astype("uint8")

        # Polygonize canopy (value==1)
        poly_iter = shapes(binary, mask=(binary == 1), transform=transform)
        polys = [shape(geom) for geom, val in poly_iter if val == 1]

        # Build GDF in raster CRS, then project once to UTM
        gdf = gpd.GeoDataFrame(geometry=polys, crs=src.crs)

    # Dissolve touching polygons (fast path)
    dissolved = gdf.unary_union
    out_geoms = [dissolved] if dissolved.geom_type == "Polygon" else list(dissolved.geoms)
    out_gdf = gpd.GeoDataFrame(geometry=out_geoms, crs=gdf.crs).to_crs(utm_epsg)

    # Save as ESRI Shapefile
    out_gdf.to_file(out_path, driver="ESRI Shapefile")
    print(f"✅ Wrote {out_path} | CRS={utm_epsg} | features={len(out_gdf)}")

# --- Load city boundaries (read as-is; reprojection happens inside the function) ---
van_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp')
win_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp')
ott_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp')

# --- Jobs (ETH + Meta for each city) ---
jobs = [
    # Vancouver
    ('/scratch/arbmarta/ETH/Vancouver ETH new.tif', van_bayan, van_utm,
     '/scratch/arbmarta/ETH/Vancouver ETH canopy.shp'),
    ('/scratch/arbmarta/Meta/Vancouver Meta.tif', van_bayan, van_utm,
     '/scratch/arbmarta/Meta/Vancouver Meta canopy.shp'),

    # Winnipeg
    ('/scratch/arbmarta/ETH/Winnipeg ETH.tif', win_bayan, win_utm,
     '/scratch/arbmarta/ETH/Winnipeg ETH canopy.shp'),
    ('/scratch/arbmarta/Meta/Winnipeg Meta.tif', win_bayan, win_utm,
     '/scratch/arbmarta/Meta/Winnipeg Meta canopy.shp'),

    # Ottawa
    ('/scratch/arbmarta/ETH/Ottawa ETH.tif', ott_bayan, ott_utm,
     '/scratch/arbmarta/ETH/Ottawa ETH canopy.shp'),
    ('/scratch/arbmarta/Meta/Ottawa Meta.tif', ott_bayan, ott_utm,
     '/scratch/arbmarta/Meta/Ottawa Meta canopy.shp'),
]

if __name__ == "__main__":
    with Pool(processes=6) as pool:  # adjust to available cores per node
        pool.starmap(raster_to_canopy_polygons, jobs)
