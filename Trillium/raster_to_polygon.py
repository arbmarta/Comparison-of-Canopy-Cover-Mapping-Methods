import rasterio
import geopandas as gpd
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape

# UTM zones
van_utm = "EPSG:32610"  # Vancouver
win_utm = "EPSG:32614"  # Winnipeg
ott_utm = "EPSG:32618"  # Ottawa

# Access Bayan datasets
van_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs(van_utm)
win_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs(win_utm)
ott_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs(ott_utm)

# Raster to canopy polygons function
def raster_to_canopy_polygons(raster_path, mask_gdf, utm_epsg, out_path):
    # Reproject raster on-the-fly to match the target UTM
    with rasterio.open(raster_path) as src, \
         WarpedVRT(src, crs=utm_epsg, resampling=Resampling.nearest) as vrt:
        # Mask raster to polygon boundary
        masked, transform = mask(vrt, mask_gdf.geometry, crop=True, filled=True, nodata=0)

    # Convert to binary canopy: 1 if value >= 2, else 0
    binary = (masked[0] >= 2).astype("uint8")

    # Extract polygons for canopy pixels only
    poly_iter = (
        (shape(geom), val)
        for geom, val in shapes(binary, mask=(binary == 1), transform=transform)
    )
    polys = [geom for geom, val in poly_iter if val == 1]

    # Create GeoDataFrame in UTM CRS
    gdf = gpd.GeoDataFrame(geometry=polys, crs=utm_epsg)

    # Merge touching polygons
    dissolved = gdf.buffer(0).unary_union
    out_geoms = [dissolved] if dissolved.geom_type == "Polygon" else list(dissolved.geoms)
    out_gdf = gpd.GeoDataFrame(geometry=out_geoms, crs=utm_epsg)

    # Save as ESRI Shapefile
    out_gdf.to_file(out_path, driver="ESRI Shapefile")
    print(f"✅ Wrote {out_path}")

# Loop through rasters
city_jobs = [
    ("Vancouver", "/scratch/arbmarta/ETH/Vancouver ETH new.tif", van_bayan, van_utm, "/scratch/arbmarta/ETH/Vancouver ETH canopy.shp"),
    ("Winnipeg",  "/scratch/arbmarta/ETH/Winnipeg ETH.tif",     win_bayan, win_utm, "/scratch/arbmarta/ETH/Winnipeg ETH canopy.shp"),
    ("Ottawa",    "/scratch/arbmarta/ETH/Ottawa ETH.tif",       ott_bayan, ott_utm, "/scratch/arbmarta/ETH/Ottawa ETH canopy.shp"),
    ("Vancouver", "/scratch/arbmarta/Meta/Vancouver Meta.tif",  van_bayan, van_utm, "/scratch/arbmarta/Meta/Vancouver Meta canopy.shp"),
    ("Winnipeg",  "/scratch/arbmarta/Meta/Winnipeg Meta.tif",   win_bayan, win_utm, "/scratch/arbmarta/Meta/Winnipeg Meta canopy.shp"),
    ("Ottawa",    "/scratch/arbmarta/Meta/Ottawa Meta.tif",     ott_bayan, ott_utm, "/scratch/arbmarta/Meta/Ottawa Meta canopy.shp"),
]
for _, r, g, crs, out in city_jobs:
    raster_to_canopy_polygons(r, g, crs, out)
