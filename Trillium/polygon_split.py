import geopandas as gpd
from multiprocessing import Pool

# UTM zones
van_utm = "EPSG:32610"  # Vancouver
win_utm = "EPSG:32614"  # Winnipeg
ott_utm = "EPSG:32618"  # Ottawa

# Helper to process one city (load → reproject → fix → split → save)
def process_layer(layer_path, bayan_path, utm_epsg, out_path):
    gdf   = gpd.read_file(layer_path).to_crs(utm_epsg)
    bayan = gpd.read_file(bayan_path).to_crs(utm_epsg)

    # fix invalids
    gdf["geometry"]   = gdf.geometry.buffer(0)
    bayan["geometry"] = bayan.geometry.buffer(0)

    # split + save
    split = gpd.overlay(gdf, bayan, how="identity").explode(index_parts=False, ignore_index=True)
    split.to_file(out_path)

# REPLACE jobs with one job per (city, layer)
jobs = [
    # Vancouver (meta & eth)
    ("/scratch/arbmarta/Meta/Vancouver Meta canopy.shp",
     "/scratch/arbmarta/Trinity/Vancouver/TVAN.shp",
     van_utm,
     "/scratch/arbmarta/Meta/Vancouver Meta canopy_SPLIT.shp"),
    ("/scratch/arbmarta/ETH/Vancouver ETH canopy.shp",
     "/scratch/arbmarta/Trinity/Vancouver/TVAN.shp",
     van_utm,
     "/scratch/arbmarta/ETH/Vancouver ETH canopy_SPLIT.shp"),

    # Winnipeg (meta & eth)
    ("/scratch/arbmarta/Meta/Winnipeg Meta canopy.shp",
     "/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp",
     win_utm,
     "/scratch/arbmarta/Meta/Winnipeg Meta canopy_SPLIT.shp"),
    ("/scratch/arbmarta/ETH/Winnipeg ETH canopy.shp",
     "/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp",
     win_utm,
     "/scratch/arbmarta/ETH/Winnipeg ETH canopy_SPLIT.shp"),

    # Ottawa (meta & eth)
    ("/scratch/arbmarta/Meta/Ottawa Meta canopy.shp",
     "/scratch/arbmarta/Trinity/Ottawa/TOTT.shp",
     ott_utm,
     "/scratch/arbmarta/Meta/Ottawa Meta canopy_SPLIT.shp"),
    ("/scratch/arbmarta/ETH/Ottawa ETH canopy.shp",
     "/scratch/arbmarta/Trinity/Ottawa/TOTT.shp",
     ott_utm,
     "/scratch/arbmarta/ETH/Ottawa ETH canopy_SPLIT.shp"),
]

if __name__ == "__main__":
    with Pool(processes=6) as pool:
        pool.starmap(process_layer, jobs)
