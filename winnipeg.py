import geopandas as gpd
import pandas as pd

# Notes
# Canopy cover area in shapefiles is calculated in square meters

# Municipal boundary area in km²
wpg_area = 475382755.7377889 / 1000000

## ------------------------------------------ IMPORT THE CANOPY COVER DATASETS -----------------------------------------
#region

print("Importing the canopy cover datasets...")
wpg_meta = gpd.read_file('Winnipeg/Winnipeg Meta Canopy Cover Polygon.shp').to_crs("EPSG:32614")
wpg_eth = gpd.read_file('Winnipeg/Winnipeg ETH Canopy Cover Polygon.shp').to_crs("EPSG:32614")
wpg_bayan = gpd.read_file('Winnipeg/Bayan/TWPG.shp').to_crs("EPSG:32614")
wpg_lidar = gpd.read_file('Winnipeg/Winnipeg LiDAR Canopy Cover Polygon.shp').to_crs("EPSG:32614")

# Read the photo-interpretation csv
wpg_itree = pd.read_csv('Winnipeg/Winnipeg Photointerpretation.csv')
wpg_itree_lidar = pd.read_csv('Winnipeg/Winnipeg Photointerpretation LiDAR.csv')

#endregion

## ---------------------------------------------- TOTAL CANOPY COVER AREA ----------------------------------------------
#region

print(f"\n--- WINNIPEG TOTAL CANOPY COVER AREA ---\n")

# Calculate total canopy cover from wpg_lidar
lidar_canopy_area = wpg_lidar['CanopyArea'].sum() / 1000000
lidar_canopy_cover = lidar_canopy_area / wpg_area * 100
print(f"\nLiDAR-derived canopy cover: {lidar_canopy_cover:.2f}% ({lidar_canopy_area:.2f} km²)")

# Estimate total canopy cover from meta
meta_canopy_area = wpg_meta['CanopyArea'].sum() / 1000000
meta_canopy_cover = meta_canopy_area / wpg_area * 100
print(f"\nMeta CHM canopy cover: {meta_canopy_cover:.2f}% ({meta_canopy_area:.2f} km²)")

# Estimate total canopy cover from eth
eth_canopy_area = wpg_eth['CanopyArea'].sum() / 1000000
eth_canopy_cover = eth_canopy_area / wpg_area * 100
print(f"\nETH CHM canopy cover: {eth_canopy_cover:.2f}% ({eth_canopy_area:.2f} km²)")

# Estimate total canopy cover from Bayan
wpg_bayan['Bayan_Area'] = wpg_bayan['pred'] / 100 * 14400
bayan_canopy_area = wpg_bayan['Bayan_Area'].sum() / 1000000
bayan_canopy_cover = bayan_canopy_area / wpg_area * 100
print(f"\nBayan canopy cover: {bayan_canopy_cover:.2f}% ({bayan_canopy_area:.2f} km²)")

# Estimate total canopy cover from i-tree
itree_canopy_cover = wpg_itree['Canopy'].sum() / 1000
itree_lidar_canopy_cover = wpg_itree_lidar['Canopy'].sum() / 1000
print(f"\nPhotointerpretation canopy cover: {itree_canopy_cover:.2f}%")
print(f"Photointerpretation canopy cover (LiDAR): {itree_lidar_canopy_cover:.2f}%")

#endregion

## --------------------------------------- COMPARE RESULTS WITHIN THE BAYAN GRID ---------------------------------------
#region

print("\nSplitting canopy polygons by Bayan grid boundaries...")
meta_split = gpd.overlay(wpg_meta, wpg_bayan, how='intersection')
eth_split = gpd.overlay(wpg_eth, wpg_bayan, how='intersection')
lidar_split = gpd.overlay(wpg_lidar, wpg_bayan, how='intersection')

print("Calculating split polygon areas...")
meta_split['Meta_Area'] = meta_split.geometry.area
eth_split['ETH_Area'] = eth_split.geometry.area
lidar_split['LiDAR_Area'] = lidar_split.geometry.area

print("Aggregating canopy area per grid cell...")
meta_by_grid = meta_split.groupby('id')['Meta_Area'].sum().reset_index()
eth_by_grid = eth_split.groupby('id')['ETH_Area'].sum().reset_index()
lidar_by_grid = lidar_split.groupby('id')['LiDAR_Area'].sum().reset_index()

print("Merging canopy area values back into Bayan grid...")
wpg_bayan = wpg_bayan.merge(meta_by_grid, on='id', how='left')
wpg_bayan = wpg_bayan.merge(eth_by_grid, on='id', how='left')
wpg_bayan = wpg_bayan.merge(lidar_by_grid, on='id', how='left')

wpg_bayan[['Meta_Area', 'ETH_Area', 'LiDAR_Area']] = wpg_bayan[['Meta_Area', 'ETH_Area', 'LiDAR_Area']].fillna(0)

print("Calculating percent canopy cover per grid cell...")
wpg_bayan = wpg_bayan.assign(
    Meta_Canopy_Percent = (wpg_bayan['Meta_Area'] / 14400) * 100,
    ETH_Canopy_Percent = (wpg_bayan['ETH_Area'] / 14400) * 100,
    LiDAR_Canopy_Percent = (wpg_bayan['LiDAR_Area'] / 14400) * 100
)

print("Calculating total canopy cover across full grid extent...")
total_eth_area_m2 = wpg_bayan['ETH_Area'].sum()
total_meta_area_m2 = wpg_bayan['Meta_Area'].sum()
total_lidar_area_m2 = wpg_bayan['LiDAR_Area'].sum()

total_eth_area_km2 = total_eth_area_m2 / 1e6
total_meta_area_km2 = total_meta_area_m2 / 1e6
total_lidar_area_km2 = total_lidar_area_m2 / 1e6

grid_area_m2 = 466718400
eth_cover_percent = (total_eth_area_m2 / grid_area_m2) * 100
meta_cover_percent = (total_meta_area_m2 / grid_area_m2) * 100
lidar_cover_percent = (total_lidar_area_m2 / grid_area_m2) * 100

print("\n\n--- Canopy Cover Summary Across Entire Grid ---")
print(f"LiDAR Canopy Cover: {lidar_cover_percent:.2f}% ({total_lidar_area_km2:.2f} km²)")
print(f"Meta-estimated canopy cover: {meta_cover_percent:.2f}% ({total_meta_area_km2:.2f} km²)")
print(f"ETH-estimated canopy cover: {eth_cover_percent:.2f}% ({total_eth_area_km2:.2f} km²)")
print(f"Bayan-estimated canopy cover: {bayan_canopy_cover:.2f}% ({bayan_canopy_area:.2f} km²)")

print("\nExporting final GeoDataFrame to CSV with centroid coordinates...")
gdf = wpg_bayan.copy()
gdf['X'] = gdf.geometry.centroid.x
gdf['Y'] = gdf.geometry.centroid.y
gdf.drop(columns='geometry').to_csv("Winnipeg/Bayan/Canopy_Cover_Results_with_coords.csv", index=False)

print("Done! ✅")

#endregion
