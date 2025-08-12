import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from multiprocessing import Pool

# Import the bayan datasets
van_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs("EPSG:32610")
wpg_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs("EPSG:32614")
ott_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs("EPSG:32618")

# Location of rasters
# Vancouver
van_eth = '/scratch/arbmarta/ETH/Vancouver ETH.tif'
van_meta = '/scratch/arbmarta/Meta/Vancouver Meta.tif'

# Winnipeg
win_eth = '/scratch/arbmarta/ETH/Winnipeg ETH.tif'
win_meta = '/scratch/arbmarta/Meta/Winnipeg Meta.tif'

# Ottawa
ott_eth = '/scratch/arbmarta/ETH/Ottawa ETH.tif'
ott_meta = '/scratch/arbmarta/Meta/Ottawa Meta.tif'

# Local UTM zones
UTM = {"Vancouver": "EPSG:32610", "Winnipeg": "EPSG:32614", "Ottawa": "EPSG:32618"}

# Single output folder for everything
OUT_DIR = "/scratch/arbmarta/ETH/Outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------- HELPERS ------------------------------------
def ensure_valid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gdf.copy()
    g["geometry"] = g.geometry.buffer(0)
    g = g[~g.geometry.is_empty]
    return g

def raster_to_polygons(masked_arr, out_transform, nodata=None) -> gpd.GeoDataFrame:
    band = masked_arr[0]

    if nodata is None:
        include_mask = ~np.isnan(band) if np.issubdtype(band.dtype, np.floating) else np.ones_like(band, dtype=bool)
    else:
        include_mask = band != nodata

    geoms, vals = [], []
    for geom, value in shapes(band, mask=include_mask, transform=out_transform):
        if value is None:
            continue
        geoms.append(shape(geom))
        vals.append(float(value))
    return gpd.GeoDataFrame({"value": vals}, geometry=geoms)

def mask_polygonize_split_to_utm(raster_path: str,
                                 boundary_utm: gpd.GeoDataFrame,
                                 utm_epsg: str,
                                 city: str,
                                 source: str) -> None:
    """
    1) mask raster by boundary (in raster CRS)
    2) polygonize
    3) reproject polygons to UTM
    4) split polygons by boundary (in UTM)
    5) write both unsplit and split shapefiles to OUT_DIR
    """
    boundary_utm = ensure_valid(boundary_utm)

    with rasterio.open(raster_path) as src:
        boundary_in_src = boundary_utm.to_crs(src.crs)
        masked_arr, out_transform = mask(src, boundary_in_src.geometry, crop=True)

        # Optional thresholding example (commented):
        # band = masked_arr[0].astype(float)
        # band[band < 2] = np.nan
        # masked_arr[0] = band

        polys = raster_to_polygons(masked_arr, out_transform, nodata=src.nodata)
        polys.set_crs(src.crs, inplace=True)

    # Reproject to UTM
    polys_utm = ensure_valid(polys.to_crs(utm_epsg))
    split_utm = ensure_valid(
        gpd.overlay(polys_utm, boundary_utm, how="identity").explode(index_parts=False, ignore_index=True)
    )

    # Save
    base = f"{city}_{source}_Polygons_UTM"
    out_unsplit = os.path.join(OUT_DIR, f"{base}.shp")
    out_split   = os.path.join(OUT_DIR, f"{base}_SplitByBoundary.shp")

    polys_utm.to_file(out_unsplit)
    split_utm.to_file(out_split)
    print(f"Wrote:\n  {out_unsplit}\n  {out_split}")

# ------------------------------------ RUN ---------------------------------------
def run_task(args):
    raster_path, bayan_gdf, utm_epsg, city, source = args
    mask_polygonize_split_to_utm(raster_path, bayan_gdf, utm_epsg, city, source)

def main():
    tasks = [
        # Vancouver
        (van_eth,  van_bayan, UTM["Vancouver"], "Vancouver", "ETH"),
        (van_meta, van_bayan, UTM["Vancouver"], "Vancouver", "Meta"),
        # Winnipeg
        (win_eth,  wpg_bayan, UTM["Winnipeg"],  "Winnipeg",  "ETH"),
        (win_meta, wpg_bayan, UTM["Winnipeg"],  "Winnipeg",  "Meta"),
        # Ottawa
        (ott_eth,  ott_bayan, UTM["Ottawa"],    "Ottawa",    "ETH"),
        (ott_meta, ott_bayan, UTM["Ottawa"],    "Ottawa",    "Meta"),
    ]

    with Pool(processes=6) as pool:
        pool.map(run_task, tasks)

if __name__ == "__main__":
    main()
