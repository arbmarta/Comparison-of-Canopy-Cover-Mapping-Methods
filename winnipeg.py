import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# Notes
# Canopy cover area in shapefiles is calculated in square meters

# Municipal boundary area
wpg_area = 475382755.7377889 / 1000000

## ------------------------------------------ IMPORT THE CANOPY COVER DATASETS -----------------------------------------
#region

# Read the canopy dataset shapefiles
wpg_meta = gpd.read_file('Winnipeg/Winnipeg Meta Canopy Cover Polygon.shp')
wpg_eth = gpd.read_file('Winnipeg/Winnipeg ETH Canopy Cover Polygon.shp')
wpg_bayan = gpd.read_file('Winnipeg/Bayan/TWPG.shp')
wpg_lidar = gpd.read_file('Winnipeg/Winnipeg LiDAR Canopy Cover Polygon.shp')

# Define the target CRS (UTM Zone 14N - meters)
wpg_utm_crs = "EPSG:32614"

# Reproject all layers to UTM Zone 14N
wpg_meta = wpg_meta.to_crs(wpg_utm_crs)
wpg_eth = wpg_eth.to_crs(wpg_utm_crs)
wpg_bayan = wpg_bayan.to_crs(wpg_utm_crs)
wpg_lidar = wpg_lidar.to_crs(wpg_utm_crs)

# Read the photo-interpretation csv
wpg_itree = pd.read_csv('Winnipeg\Winnipeg Photointerpretation.csv')
wpg_itree_lidar = pd.read_csv('Winnipeg\Winnipeg Photointerpretation LiDAR.csv')

# endregion

## ---------------------------------------------- TOTAL CANOPY COVER AREA ----------------------------------------------
#region

print(f"--- WINNIPEG TOTAL CANOPY COVER AREA---\n")

# Calculate total canopy cover from wpg_lidar
lidar_canopy_area = wpg_lidar['CanopyArea'].sum() / 1000000
lidar_canopy_cover = lidar_canopy_area / wpg_area * 100

print(f"LiDAR-derived canopy cover: {lidar_canopy_cover:.2f}% ({lidar_canopy_area:.2f} km²)")

# Estimate total canopy cover from wpg_meta
meta_canopy_area = wpg_meta['CanopyArea'].sum() / 1000000
meta_canopy_cover = meta_canopy_area / wpg_area * 100

print(f"\nMeta CHM canopy cover: {meta_canopy_cover:.2f}% ({meta_canopy_area:.2f} km²)")

# Estimate total canopy cover from wpg_eth
eth_canopy_area = wpg_eth['CanopyArea'].sum() / 1000000
eth_canopy_cover = eth_canopy_area / wpg_area * 100

print(f"\nETH CHM canopy cover: {eth_canopy_cover:.2f}% ({eth_canopy_area:.2f} km²)")

# Estimate total canopy cover from wpg_bayan
wpg_bayan['Bayan_Area'] = wpg_bayan['pred'] / 100 * 14400
bayan_canopy_area = wpg_bayan['Bayan_Area'].sum() / 1000000
bayan_canopy_cover = bayan_canopy_area / wpg_area * 100

print(f"\nBayan canopy cover: {bayan_canopy_cover:.2f}% ({bayan_canopy_area:.2f} km²)")

# Estimate total canopy cover from photointerpretation
itree_canopy_cover = (wpg_itree['Canopy'].sum()) / 1000 # divide by 100,000 and multiply by 100
print(f"\nPhotointerpretation canopy cover: {itree_canopy_cover:.2f}%")

itree_lidar_canopy_cover = (wpg_itree_lidar['Canopy'].sum()) / 1000 # divide by 100,000 and multiply by 100
print(f"Photointerpretation canopy cover (LiDAR): {itree_lidar_canopy_cover:.2f}%")

#endregion

## --------------------------------------- COMPARE RESULTS WITHIN THE BAYAN GRID ---------------------------------------
#region

# Split wpg_eth and wpg_meta at grid boundaries
meta_split = gpd.overlay(wpg_meta, wpg_bayan, how='intersection')
eth_split = gpd.overlay(wpg_eth, wpg_bayan, how='intersection')
lidar_split = gpd.overlay(wpg_lidar, wpg_bayan, how='intersection')

# Calculate area of each split polygon; add a new column for area in square meters
meta_split['Meta_Area'] = meta_split.geometry.area
eth_split['ETH_Area'] = eth_split.geometry.area
lidar_split['LiDAR_Area'] = lidar_split.geometry.area

# Sum canopy cover area within each grid cell
meta_by_grid = meta_split.groupby('id')['Meta_Area'].sum().reset_index()
eth_by_grid = eth_split.groupby('id')['ETH_Area'].sum().reset_index()
lidar_by_grid = lidar_split.groupby('id')['LiDAR_Area'].sum().reset_index()

# Join canopy cover area back to Bayan grid
wpg_bayan = wpg_bayan.merge(meta_by_grid, on='id', how='left')
wpg_bayan = wpg_bayan.merge(eth_by_grid, on='id', how='left')
wpg_bayan = wpg_bayan.merge(lidar_by_grid, on='id', how='left')

# Replace NaN values with 0 for cells without canopy
wpg_bayan['Meta_Area'] = wpg_bayan['Meta_Area'].fillna(0)
wpg_bayan['ETH_Area'] = wpg_bayan['ETH_Area'].fillna(0)
wpg_bayan['LiDAR_Area'] = wpg_bayan['LiDAR_Area'].fillna(0)

# Calculate canopy cover percentage (area / 14,400 m² grid cell)
wpg_bayan['Meta_Canopy_Percent'] = (wpg_bayan['Meta_Area'] / 14400) * 100
wpg_bayan['ETH_Canopy_Percent'] = (wpg_bayan['ETH_Area'] / 14400) * 100
wpg_bayan['LiDAR_Canopy_Percent'] = (wpg_bayan['LiDAR_Area'] / 14400) * 100

# Calculate total canopy area (in m² and km²)
total_eth_area_m2 = wpg_bayan['ETH_Area'].sum()
total_meta_area_m2 = wpg_bayan['Meta_Area'].sum()
total_lidar_area_m2 = wpg_bayan['LiDAR_Area'].sum()

total_eth_area_km2 = total_eth_area_m2 / 1e6
total_meta_area_km2 = total_meta_area_m2 / 1e6
total_lidar_area_km2 = total_lidar_area_m2 / 1e6

# Calculate canopy cover % across entire grid extent; 466718400 is the total grid area in square meters
eth_cover_percent = (total_eth_area_m2 / 466718400) * 100
meta_cover_percent = (total_meta_area_m2 / 466718400) * 100
lidar_cover_percent = (total_lidar_area_m2 / 466718400) * 100

# Print results
print("\n\n--- Canopy Cover Summary Across Entire Grid ---")
print(f"LiDAR Canopy Cover: {lidar_cover_percent:.2f}% ({total_lidar_area_km2:.2f} km²)")
print(f"Meta-estimated canopy cover: {meta_cover_percent:.2f}% ({total_meta_area_km2:.2f} km²)")
print(f"ETH-estimated canopy cover: {eth_cover_percent:.2f}% ({total_eth_area_km2:.2f} km²)")
print(f"Bayan-estimated canopy cover: {bayan_canopy_cover:.2f}% ({bayan_canopy_area:.2f} km²)")

#endregion

## --------------------------------------------------- CALCULATE RMSE --------------------------------------------------
#region

# Copy the wpg_bayan dataframe
gdf = wpg_bayan.drop.copy()

# Ensure no missing values in any of the three columns
valid = gdf[['LiDAR_Area', 'ETH_Area', 'Meta_Area', 'Bayan_Area']].dropna()

# Calculate RMSE value
eth_rmse = mean_squared_error(valid['LiDAR_Area'], valid['ETH_Area'], squared=False)
meta_rmse = mean_squared_error(valid['LiDAR_Area'], valid['Meta_Area'], squared=False)

# Print results
print(f"RMSE (ETH vs LiDAR): {eth_rmse:.2f} m²")
print(f"RMSE (Meta vs LiDAR): {meta_rmse:.2f} m²")

#
print(gdf.columns)

#endregion

## -------------------------------------------- COMPARE PHOTOINTERPRETATION --------------------------------------------
#region



#endregion
