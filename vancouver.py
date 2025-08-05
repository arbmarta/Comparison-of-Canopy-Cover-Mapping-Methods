import geopandas as gpd
import pandas as pd

# Notes
# Canopy cover area in shapefiles is calculated in square meters

# Municipal boundary area
van_area = XXX / 1000000

## ------------------------------------------ IMPORT THE CANOPY COVER DATASETS -----------------------------------------
#region

# Read the canopy dataset shapefiles
van_meta = gpd.read_file('Vancouver/Vancouver Meta Canopy Cover Polygon.shp')
van_eth = gpd.read_file('Vancouver/Vancouver ETH Canopy Cover Polygon.shp')
van_bayan = gpd.read_file('Vancouver/Bayan/TWPG.shp')
van_lidar = gpd.read_file('Vancouver/Vancouver LiDAR Canopy Cover Polygon.shp')

# Define the target CRS (UTM Zone 14N - meters)
van_utm_crs = "EPSG:32614"

# Reproject all layers to UTM Zone 14N
van_meta = van_meta.to_crs(van_utm_crs)
van_eth = van_eth.to_crs(van_utm_crs)
van_bayan = van_bayan.to_crs(van_utm_crs)
van_lidar = van_lidar.to_crs(van_utm_crs)

# Read the photo-interpretation csv
van_itree = pd.read_csv('Vancouver\Vancouver Photointerpretation.csv')
van_itree_lidar = pd.read_csv('Vancouver\Vancouver Photointerpretation LiDAR.csv')

# endregion

## ---------------------------------------------- TOTAL CANOPY COVER AREA ----------------------------------------------
#region

print(f"--- Vancouver TOTAL CANOPY COVER AREA---\n")

# Calculate total canopy cover from van_lidar
lidar_canopy_area = van_lidar['CanopyArea'].sum() / 1000000
lidar_canopy_cover = lidar_canopy_area / van_area * 100

print(f"LiDAR-derived canopy cover: {lidar_canopy_cover:.2f}% ({lidar_canopy_area:.2f} km²)")

# Estimate total canopy cover from van_meta
meta_canopy_area = van_meta['CanopyArea'].sum() / 1000000
meta_canopy_cover = meta_canopy_area / van_area * 100

print(f"\nMeta CHM canopy cover: {meta_canopy_cover:.2f}% ({meta_canopy_area:.2f} km²)")

# Estimate total canopy cover from van_eth
eth_canopy_area = van_eth['CanopyArea'].sum() / 1000000
eth_canopy_cover = eth_canopy_area / van_area * 100

print(f"\nETH CHM canopy cover: {eth_canopy_cover:.2f}% ({eth_canopy_area:.2f} km²)")

# Estimate total canopy cover from van_bayan
van_bayan['Bayan_Area'] = van_bayan['pred'] / 100 * 14400
bayan_canopy_area = van_bayan['Bayan_Area'].sum() / 1000000
bayan_canopy_cover = bayan_canopy_area / van_area * 100

print(f"\nBayan canopy cover: {bayan_canopy_cover:.2f}% ({bayan_canopy_area:.2f} km²)")

# Estimate total canopy cover from photointerpretation
itree_canopy_cover = (van_itree['Canopy'].sum()) / 1000 # divide by 100,000 and multiply by 100
print(f"\nPhotointerpretation canopy cover: {itree_canopy_cover:.2f}%")

itree_lidar_canopy_cover = (van_itree_lidar['Canopy'].sum()) / 1000 # divide by 100,000 and multiply by 100
print(f"Photointerpretation canopy cover (LiDAR): {itree_lidar_canopy_cover:.2f}%")

#endregion

## --------------------------------------- COMPARE RESULTS WITHIN THE BAYAN GRID ---------------------------------------
#region

# Split van_eth and van_meta at grid boundaries
meta_split = gpd.overlay(van_meta, van_bayan, how='intersection')
eth_split = gpd.overlay(van_eth, van_bayan, how='intersection')
lidar_split = gpd.overlay(van_lidar, van_bayan, how='intersection')

# Calculate area of each split polygon; add a new column for area in square meters
meta_split['Meta_Area'] = meta_split.geometry.area
eth_split['ETH_Area'] = eth_split.geometry.area
lidar_split['LiDAR_Area'] = lidar_split.geometry.area

# Sum canopy cover area within each grid cell
meta_by_grid = meta_split.groupby('id')['Meta_Area'].sum().reset_index()
eth_by_grid = eth_split.groupby('id')['ETH_Area'].sum().reset_index()
lidar_by_grid = lidar_split.groupby('id')['LiDAR_Area'].sum().reset_index()

# Join canopy cover area back to Bayan grid
van_bayan = van_bayan.merge(meta_by_grid, on='id', how='left')
van_bayan = van_bayan.merge(eth_by_grid, on='id', how='left')
van_bayan = van_bayan.merge(lidar_by_grid, on='id', how='left')

# Replace NaN values with 0 for cells without canopy
van_bayan['Meta_Area'] = van_bayan['Meta_Area'].fillna(0)
van_bayan['ETH_Area'] = van_bayan['ETH_Area'].fillna(0)
van_bayan['LiDAR_Area'] = van_bayan['LiDAR_Area'].fillna(0)

# Calculate canopy cover percentage (area / 14,400 m² grid cell)
van_bayan['Meta_Canopy_Percent'] = (van_bayan['Meta_Area'] / 14400) * 100
van_bayan['ETH_Canopy_Percent'] = (van_bayan['ETH_Area'] / 14400) * 100
van_bayan['LiDAR_Canopy_Percent'] = (van_bayan['LiDAR_Area'] / 14400) * 100

# Calculate total canopy area (in m² and km²)
total_eth_area_m2 = van_bayan['ETH_Area'].sum()
total_meta_area_m2 = van_bayan['Meta_Area'].sum()
total_lidar_area_m2 = van_bayan['LiDAR_Area'].sum()

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

# Copy the van_bayan dataframe
gdf = van_bayan.drop.copy()

# Extract the centroid coordinates
gdf['X'] = gdf.geometry.centroid.x
gdf['Y'] = gdf.geometry.centroid.y

# Save with centroid coordinates
gdf.drop(columns='geometry').to_csv("Vancouver/Bayan/Canopy_Cover_Results_with_coords.csv", index=False)

#endregion
