import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_lidar = pd.read_csv('LiDAR.csv')
df_cc = pd.read_csv('All_Cities_Percent_Cover.csv')

## -------------------------------------------- DATASET IMPORT AND HANDLING --------------------------------------------
#region

# ------------ Handle the canopy cover dataset ------------
#region

# Rename the bayan percent canopy cover column
df_cc = df_cc.rename(columns={"pred": "bayan_percent_cover"})

# Select which columns to keep
df_cc = df_cc[['city', 'grid_id', 'bayan_percent_cover', 'source', 'total_m2',
               'percent_cover']]

# Convert column source to lowercase in-place
df_cc['source'] = df_cc['source'].str.strip().str.lower()

# Keep one Bayan_percent_cover and one city per grid_id
bayan = df_cc[['grid_id', 'bayan_percent_cover']].drop_duplicates()
city = df_cc[['grid_id', 'city']].drop_duplicates()

# Pivot Meta/ETH
df_wide = df_cc.pivot(
    index="grid_id",
    columns="source",
    values=["total_m2", "percent_cover"]
)

# Flatten multi-level columns
df_wide.columns = [f"{src}_{val}" for val, src in df_wide.columns]
df_wide = df_wide.reset_index()

# Merge Bayan_percent_cover and city back
df_wide = df_wide.merge(bayan, on="grid_id", how="left")
df_wide = df_wide.merge(city, on="grid_id", how="left")

df_wide = df_wide[['city', 'grid_id', 'bayan_percent_cover', 'meta_total_m2', 'eth_total_m2', 'meta_percent_cover',
                'eth_percent_cover']]

#endregion

# ------------ Handle the LiDAR dataset ------------
#region

# Rename the total m2 and percent canopy cover columns
df_lidar = df_lidar.rename(columns={"total_m2": "lidar_total_m2"})
df_lidar = df_lidar.rename(columns={"percent_cover": "lidar_percent_cover"})

# Drop the city column
df_lidar = df_lidar.drop(columns='city')

# Merge df_lidar with df_wide
df = df_wide.merge(df_lidar, on="grid_id", how="left")

#endregion

#endregion

## ------------------------------------------- CALCULATE TOTAL CANOPY COVER --------------------------------------------
#region

# Number of grids per city
van_grid_count = 8224
win_grid_count = 32411
ott_grid_count = 41539

# Define total area values (replace with your actual values)
area_map = {
    'Vancouver': van_grid_count * (120 * 120),
    'Winnipeg': win_grid_count * (120 * 120),
    'Ottawa': ott_grid_count * (120 * 120)
}

# Create Bayan_total_m2 based on grid cell area
df['bayan_total_m2'] = df['bayan_percent_cover'] / 100 * (120 * 120)

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
    print(f"{city} Meta canopy cover: {row['meta_percent_cover']:.2f}% ({row['meta_total_m2'] / 1_000_000:.2f} km²)")
    print(f"{city} Bayan canopy cover: {row['bayan_percent_cover']:.2f}% ({row['bayan_total_m2'] / 1_000_000:.2f} km²)")
    print(f"{city} ETH canopy cover: {row['eth_percent_cover']:.2f}% ({row['eth_total_m2'] / 1_000_000:.2f} km²)\n")

#endregion

## ------------------------------------------------ FIGURE 1 HISTOGRAM -------------------------------------------------
#region

# Define sources
sources = ['lidar', 'meta', 'bayan']  # skip 'eth' if it's all zeros

# Get list of cities from your dataframe
cities = df['city'].dropna().unique()

line_styles = ['-', '--', '-.', ':']
colors = sns.color_palette("tab10")

for i, city in enumerate(cities):
    city_df = df[df['city'] == city]

    plt.figure(figsize=(10, 6))
    for j, src in enumerate(sources):
        col = f'{src}_percent_cover'
        if col in city_df.columns and city_df[col].notna().any():
            sns.kdeplot(
                city_df[col].dropna(),
                label=src.capitalize(),
                linewidth=2,
                linestyle=line_styles[j % len(line_styles)],
                color=colors[j % len(colors)],
                common_norm=False
            )

    # Add vertical line and text for LiDAR mean
    lidar_mean = sum_by_city.loc[sum_by_city['city'] == city, 'lidar_percent_cover'].values[0]
    plt.axvline(lidar_mean, color='black', linestyle='--', linewidth=1.5, label=f'{lidar_mean:.2f}% LiDAR Mean')
    plt.text(lidar_mean + 1, plt.ylim()[1]*0.9, f'{lidar_mean:.2f}%', color='black', fontsize=10, va='top')

    # Formatting
    plt.title(f'{city} Grid-Level Canopy Cover by Method (KDE)')
    plt.xlabel('Percent Canopy Cover')
    plt.ylabel('Density')
    plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()

#endregion

exit()
