import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## -------------------------------------------- DATASET IMPORT AND HANDLING --------------------------------------------
#region

df_lidar = pd.read_csv('CSVs/LiDAR_and_Canopy_Models.csv')
df_cc = pd.read_csv('CSVs/All_Methods_Percent_Cover.csv')
df_buildings = pd.read_csv('CSVs/Buildings.csv')

print(f"df_lidar columns:")
print(df_lidar.columns)

print(f"df_cc columns:")
print(df_cc.columns)

## Handle the canopy cover dataset
#region

# Select which columns to keep
df_cc = df_cc[['city', 'source', 'Grid Cell Size', 'subgrid_id', 'total_m2', 'percent_cover', 'polygon_count']]

# Convert column source to lowercase in-place
df_cc['source'] = df_cc['source'].str.strip().str.lower()

# Keep one Bayan_percent_cover and one city per grid_id
city = df_cc[['subgrid_id', 'city']].drop_duplicates()

# Pivot Meta/ETH
df_wide = df_cc.pivot(
    index="subgrid_id",
    columns="source",
    values=["total_m2", "percent_cover", "polygon_count"]
)

# Flatten multi-level columns
df_wide.columns = [f"{src}_{val}" for val, src in df_wide.columns]
df_wide = df_wide.reset_index()

# Merge city name back
df_wide = df_wide.merge(city, on="subgrid_id", how="left")

#endregion

# ------------ Handle the LiDAR dataset ------------
#region

# Rename the total m2 and percent canopy cover columns
df_lidar = df_lidar.rename(columns={"total_m2": "lidar_total_m2"})
df_lidar = df_lidar.rename(columns={"percent_cover": "lidar_percent_cover"})

# Drop the city column
df_lidar = df_lidar.drop(columns='city')

# Merge df_lidar with df_wide
df = df_wide.merge(df_lidar, left_on="subgrid_id", right_on="grid_id", how="left")

#endregion

#endregion

## ------------------------------------------- CALCULATE TOTAL CANOPY COVER --------------------------------------------
#region

# Number of grids per city
van_grid_count = 8224
win_grid_count = 32411
ott_grid_count = 41539

# Select where grid cell size is 120 m
df_120 = df[df["Grid Cell Size"] == 120]
print(df_120.columns)

# Define total area values (replace with your actual values)
area_map = {
    'Vancouver': van_grid_count * (120 * 120),
    'Winnipeg': win_grid_count * (120 * 120),
    'Ottawa': ott_grid_count * (120 * 120)
}

# Print the total m2 of canopy from each method per city
m2_cols = [col for col in df.columns if col.endswith('_total_m2')]
sum_by_city = df.groupby('city')[m2_cols].sum().reset_index()
print("Total canopy area (m²) by city and method:")

# Add a new column with the appropriate area
sum_by_city['city_area'] = sum_by_city['city'].map(area_map)

# Compute percent canopy cover (as new columns)
for col in m2_cols:
    percent_col = col.replace('_total_m2', '_percent_cover')
    sum_by_city[percent_col] = sum_by_city[col] / sum_by_city['city_area'] * 100

# Optional: reorder for clarity
ordered_cols = ['city', 'city_area'] + m2_cols + [col.replace('_total_m2', '_percent_cover') for col in m2_cols]
sum_by_city = sum_by_city[ordered_cols]

# Print overall canopy cover
for city in ['Vancouver', 'Winnipeg', 'Ottawa']:
    row = sum_by_city[sum_by_city['city'] == city].iloc[0]

    print(f"--- {city} Canopy Cover ---")
    print(f"{city} LiDAR canopy cover: {row['lidar_percent_cover']:.2f}% ({row['lidar_total_m2'] / 1_000_000:.2f} km²)")
    print(f"CHM Canopy Cover:")
    print(f"{city} Meta canopy cover: {row['meta_percent_cover']:.2f}% ({row['meta_total_m2'] / 1_000_000:.2f} km²)")
    print(f"{city} ETH canopy cover: {row['eth_percent_cover']:.2f}% ({row['eth_total_m2'] / 1_000_000:.2f} km²)\n")
    print(f"{city} Potapov canopy cover: {row['potapov_percent_cover']:.2f}% ({row['potapov_total_m2'] / 1_000_000:.2f} km²)")
    print(f"Fractional Canopy Cover:")
    print(f"{city} GLCF canopy cover: {row['glcf_percent_cover']:.2f}% ({row['glcf_total_m2'] / 1_000_000:.2f} km²)\n")
    print(f"{city} GLOBMAP FTC canopy cover: {row['globmapftc_percent_cover']:.2f}% ({row['globmapftc_total_m2'] / 1_000_000:.2f} km²)\n")
    print(f"Land Cover:")
    print(f"{city} ESRI canopy cover: {row['esri_percent_cover']:.2f}% ({row['esri_total_m2'] / 1_000_000:.2f} km²)")
    print(f"{city} DW canopy cover: {row['dw_10m_percent_cover']:.2f}% ({row['dw_10m_total_m2'] / 1_000_000:.2f} km²)")
    print(f"{city} Terrascope 2020 canopy cover: {row['terrascope 2020_percent_cover']:.2f}% ({row['terrascope 2020_total_m2'] / 1_000_000:.2f} km²)\n")
    print(f"{city} Terrascope 2021 canopy cover: {row['terrascope 2021_percent_cover']:.2f}% ({row['terrascope 2021_total_m2'] / 1_000_000:.2f} km²)\n")

#endregion
