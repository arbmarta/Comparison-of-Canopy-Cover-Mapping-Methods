import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import jensenshannon
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## -------------------------------------------- DATASET IMPORT AND HANDLING --------------------------------------------
#region

df_lidar = pd.read_csv('CSVs/LiDAR_120m_Grid_Canopy_Metrics.csv')
df_cc = pd.read_csv('CSVs/Canopy_120m_Grids.csv')
df_buildings = pd.read_csv('CSVs/Buildings.csv')

print(f"df_lidar columns:")
print(df_lidar.columns)

print(f"\ndf_cc columns:")
print(df_cc.columns)

## Handle the canopy cover dataset
#region

# Select which columns to keep
df_cc = df_cc[['city', 'source', 'grid_size_m', 'grid_id', 'total_m2', 'percent_cover', 'polygon_count']]

# Convert column source to lowercase in-place
df_cc['source'] = df_cc['source'].str.strip().str.lower()

# Keep one Bayan_percent_cover and one city per grid_id
city = df_cc[['grid_id', 'city']].drop_duplicates()

# Pivot Meta/ETH
df_wide = df_cc.pivot(
    index="grid_id",
    columns="source",
    values=["total_m2", "percent_cover", "polygon_count"]
)

# Flatten multi-level columns
df_wide.columns = [f"{src}_{val}" for val, src in df_wide.columns]
df_wide = df_wide.reset_index()

# Merge city name back
df_wide = df_wide.merge(city, on="grid_id", how="left")

#endregion

# ------------ Handle the LiDAR dataset ------------
#region

# Rename the total m2 and percent canopy cover columns
df_lidar = df_lidar.rename(columns={"total_m2": "lidar_total_m2"})
df_lidar = df_lidar.rename(columns={"percent_cover": "lidar_percent_cover"})

# Drop the city column
df_lidar = df_lidar.drop(columns='city')

# Merge df_lidar with df_wide
df = df_wide.merge(df_lidar, left_on="grid_id", right_on="grid_id", how="left")

#endregion

#endregion

## ------------------------------------------- CALCULATE TOTAL CANOPY COVER --------------------------------------------
#region

# Number of grids per city
van_grid_count = 8224
win_grid_count = 32411
ott_grid_count = 41539

# Select where grid_size_m is 120 m
df_120 = df[df["grid_size_m"] == 120]
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
    print(f"{city} ETH canopy cover: {row['eth_percent_cover']:.2f}% ({row['eth_total_m2'] / 1_000_000:.2f} km²)")
    print(f"{city} Potapov canopy cover: {row['potapov_percent_cover']:.2f}% ({row['potapov_total_m2'] / 1_000_000:.2f} km²)")
    print(f"Fractional Canopy Cover:")
    print(f"{city} GLCF canopy cover: {row['glcf_percent_cover']:.2f}% ({row['glcf_total_m2'] / 1_000_000:.2f} km²)")
    print(f"{city} GLOBMAP FTC canopy cover: {row['globmapftc_percent_cover']:.2f}% ({row['globmapftc_total_m2'] / 1_000_000:.2f} km²)")
    print(f"Land Cover:")
    print(f"{city} ESRI canopy cover: {row['esri_percent_cover']:.2f}% ({row['esri_total_m2'] / 1_000_000:.2f} km²)")
    print(f"{city} DW canopy cover: {row['dw_10m_percent_cover']:.2f}% ({row['dw_10m_total_m2'] / 1_000_000:.2f} km²)")
    print(f"{city} Terrascope 2020 canopy cover: {row['terrascope 2020_percent_cover']:.2f}% ({row['terrascope 2020_total_m2'] / 1_000_000:.2f} km²)")
    print(f"{city} Terrascope 2021 canopy cover: {row['terrascope 2021_percent_cover']:.2f}% ({row['terrascope 2021_total_m2'] / 1_000_000:.2f} km²)\n")

#endregion

## ------------------------------------------- PLOT TOTAL CANOPY COVER --------------------------------------------
#region

# Melt dataframe into long format for plotting
percent_cols = [c for c in sum_by_city.columns if c.endswith("_percent_cover")]
plot_df = sum_by_city.melt(
    id_vars="city",
    value_vars=percent_cols,
    var_name="method",
    value_name="percent_cover"
)

# Strip suffix to get method codes
plot_df["method_code"] = plot_df["method"].str.replace("_percent_cover", "", regex=False)

# Define custom order for methods
method_order = [
    "lidar",
    "meta", "eth", "potapov",              # CHM
    "glcf", "globmapftc",                  # Fractional cover
    "esri", "dw_10m", "terrascope 2020", "terrascope 2021"  # Land cover
]

# Clean display names for legend
method_labels = {
    "lidar": "LiDAR",
    "meta": "Meta CHM",
    "eth": "ETH CHM",
    "potapov": "Potapov CHM",
    "glcf": "GLCF Fractional",
    "globmapftc": "GLOBMAP FTC",
    "esri": "ESRI Land Cover",
    "dw_10m": "Dynamic World (10m)",
    "terrascope 2020": "Terrascope 2020",
    "terrascope 2021": "Terrascope 2021"
}

# Apply categorical order
plot_df["method_code"] = plot_df["method_code"].astype(
    pd.CategoricalDtype(categories=method_order, ordered=True)
)

# Plot using method_code
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=plot_df,
    x="city",
    y="percent_cover",
    hue="method_code",
    hue_order=method_order,
    palette="tab20"
)

# Replace legend labels with display names
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, [method_labels[m] for m in method_order],
    title="Method", bbox_to_anchor=(1.05, 1), loc="upper left"
)

plt.ylabel("Canopy Cover (%)")
plt.xlabel("")
plt.title("Comparison of Canopy Cover Estimates Across Methods")
plt.tight_layout()
plt.show()

#endregion

## ---------------------------------------------- CALCULATE RMSE AND JSD -----------------------------------------------
#region

results = []

# Methods to compare against LiDAR
methods_to_compare = [m for m in method_order if m != "lidar"]

for method in methods_to_compare:
    for city in sum_by_city["city"]:
        row = sum_by_city[sum_by_city["city"] == city].iloc[0]

        # LiDAR and method % canopy cover
        lidar_val = row["lidar_percent_cover"]
        method_val = row[f"{method}_percent_cover"]

        # RMSE (here just difference since one value per city, but still useful if extended to grids)
        rmse = mean_squared_error([lidar_val], [method_val])

        # JSD — requires distributions (normalize across the two values)
        dist = np.array([lidar_val, method_val], dtype=float)
        dist = dist / dist.sum() if dist.sum() > 0 else np.array([0.5, 0.5])
        jsd = jensenshannon(dist, np.array([0.5, 0.5])) ** 2  # square to get JSD from JS distance

        results.append({
            "city": city,
            "method": method,
            "rmse": rmse,
            "jsd": jsd
        })

# Convert to DataFrame
comparison_df = pd.DataFrame(results)
print(comparison_df)

#endregion

## ----------------------------------------------- CALCULATE PREDICTORS ------------------------------------------------

#region

# Remove trailing '_120' from grid_id
df_buildings['grid_id'] = df_buildings['grid_id'].str.replace('_120$', '', regex=True)

# Merge building metrics into main dataframe
df_metrics = df_120.merge(df_buildings, on="grid_id", how="left")

#region PEARSON CORRELATION MATRIX OF DEVIATION VS METRICS

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Compute deviation from LiDAR for all methods except 'lidar'
method_cols = [col for col in df_metrics.columns if col.endswith("_percent_cover") and col != "lidar_percent_cover"]
for col in method_cols:
    df_metrics[f"{col}_dev"] = df_metrics[col] - df_metrics["lidar_percent_cover"]

# Metrics columns (y-axis)
metrics_cols = [
    'lidar_total_m2', 'lidar_percent_cover', 'polygon_count', 'mean_patch_size',
    'patch_density', 'total_perimeter', 'area_cv', 'perimeter_cv', 'PAFRAC',
    'nLSI', 'CAI_AM', 'LSI', 'ED',
    'built_area_total_m2', 'number_of_buildings', 'mean_building_size'
]

# Deviation columns (x-axis)
method_cols = [col for col in df_metrics.columns if col.endswith("_percent_cover") and col != "lidar_percent_cover"]
dev_cols = [f"{col}_dev" for col in method_cols]

# Compute pairwise correlation (metrics vs deviation)
corr_matrix = df_metrics[metrics_cols + dev_cols].corr().loc[metrics_cols, dev_cols]

# Plot heatmap with correlation values inside cells
plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    cbar_kws={'label': 'Pearson r'}
)

plt.ylabel("Metrics")
plt.xlabel("Deviation from LiDAR")
plt.title("Pearson Correlation: Landscape & Building Metrics vs Percent Cover Deviations")
plt.tight_layout()
plt.show()

#endregion
