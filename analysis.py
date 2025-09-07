from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
import shap

## -------------------------------------------------- PLOTTING SETUP ---------------------------------------------------
#region

# Define custom city order and display names
city_order = ["Vancouver", "Winnipeg", "Ottawa"]
city_labels = {"Vancouver": "Vancouver, BC", "Winnipeg": "Winnipeg, MB", "Ottawa": "Ottawa, ON"}

# Define method order (exclude lidar)
method_order = [
    "meta", "eth", "potapov",   # CHM
    "glcf", "globmapftc",       # Fractional
    "esri", "dw_10m", "terrascope 2020", "terrascope 2021"  # Land cover
]

# Define colors by group
method_colors = {
    "meta": "#4575b4",       # blue shades
    "eth": "#4575b4",
    "potapov": "#313695",
    "glcf": "#1b7837",       # green shades
    "globmapftc": "#5aae61",
    "esri": "#b2182b",          # darker red
    "dw_10m": "#ef8a62",        # medium orange-red
    "terrascope 2020": "#f59c28", # orange
    "terrascope 2021": "#b2abd2", # purple/lavender
    "lidar": 'black'
}

# Define method groups
method_groups = {
    "CHM": ["meta", "eth", "potapov"],
    "Fractional": ["glcf", "globmapftc"],
    "Land Cover": ["esri", "dw_10m", "terrascope 2020", "terrascope 2021"]
}

# Mapping for legend labels
method_labels = {
    "meta": "Meta", "eth": "ETH", "potapov": "Potapov",
    "glcf": "GLCF", "globmapftc": "GLOBMAP",
    "esri": "ESRI", "dw_10m": "Dynamic World",
    "terrascope 2020": "Terrascope 2020", "terrascope 2021": "Terrascope 2021",
    "lidar": "ALS-Derived UTC"
}

# Line styles for methods (ALS is solid, others can be changed)
method_linestyles = {
    "meta": "-", "eth": "--", "potapov": "-.",
    "glcf": ":", "globmapftc": "-",
    "esri": "--", "dw_10m": "-.", "terrascope 2020": ":", "terrascope 2021": "-",
    "lidar": "-"  # ALS solid
}

# Mapping of metric codes to full names
metric_labels = {
    "R2": "Coefficient of Determination (R²)",
    "ME": "Mean Error",
    "MAE": "Mean Absolute Error",
    "RMSE": "Root Mean Squared Error"
}

# Mapping of metric to marker symbol
metric_markers = {
    "R2": "s",
    "ME": "o",
    "MAE": "v",
    "RMSE": "^"
}

# Define consistent colors for each metric
metric_colors = {
    "R2": "#1f78b4",    # deep blue
    "RMSE": "#ff9f0d",  # bright orange
    "ME": "#33a02c",    # vibrant green
    "MAE": "#e31a1c"    # strong red
}

#endregion

## -------------------------------------------- DATASET IMPORT AND HANDLING --------------------------------------------
#region

df_lidar = pd.read_csv('CSVs/LiDAR_120m_Grid_Canopy_Metrics.csv')
df_cc = pd.read_csv('CSVs/Canopy_120m_Grids.csv')
df_buildings = pd.read_csv('CSVs/Buildings.csv')

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

# Define total area values (replace with your actual values)
area_map = {
    'Vancouver': van_grid_count * (120 * 120),
    'Winnipeg': win_grid_count * (120 * 120),
    'Ottawa': ott_grid_count * (120 * 120)
}

# Print the total m2 of canopy from each method per city
m2_cols = [col for col in df.columns if col.endswith('_total_m2')]
sum_by_city = df.groupby('city')[m2_cols].sum().reset_index()

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

## ----------------------------------------------- CSV: Correlations.csv -----------------------------------------------
#region

# Remove trailing '_120' from grid_id
df_buildings['grid_id'] = df_buildings['grid_id'].str.replace('_120$', '', regex=True)

# Merge building metrics into main dataframe
df_metrics = df_120.merge(df_buildings, on="grid_id", how="left")

# Drop one of the duplicate city columns and rename the other
df_metrics = df_metrics.drop(columns=['city_y'])  # or 'city_x' - check which one has the data
df_metrics = df_metrics.rename(columns={'city_x': 'city'})

# Compute deviation from LiDAR for all methods except 'lidar'
method_cols = [col for col in df_metrics.columns if col.endswith("_percent_cover") and col != "lidar_percent_cover"]
for col in method_cols:
    df_metrics[f"{col}_dev"] = (df_metrics[col] - df_metrics["lidar_percent_cover"]).abs()

# Metrics columns (y-axis)
metrics_cols = [
    'lidar_total_m2', 'polygon_count', 'mean_patch_size',
    'patch_density', 'total_perimeter', 'area_cv', 'perimeter_cv', 'PAFRAC',
    'nLSI', 'CAI_AM', 'LSI', 'ED',
    'built_area_total_m2', 'number_of_buildings', 'mean_building_size'
]

# Deviation columns (x-axis)
method_cols = [col for col in df_metrics.columns if col.endswith("_percent_cover") and col != "lidar_percent_cover"]
dev_cols = [f"{col}_dev" for col in method_cols]

# Reorder deviation columns for plotting
x_order = [
    "meta_percent_cover_dev",
    "eth_percent_cover_dev",
    "potapov_percent_cover_dev",
    "glcf_percent_cover_dev",
    "globmapftc_percent_cover_dev",
    "dw_10m_percent_cover_dev",
    "esri_percent_cover_dev",
    "terrascope 2020_percent_cover_dev",
    "terrascope 2021_percent_cover_dev",
]

dev_cols = [col for col in x_order if col in dev_cols]

# Compute correlation + significance
corr_matrix = pd.DataFrame(index=metrics_cols, columns=dev_cols)
annot_matrix = pd.DataFrame(index=metrics_cols, columns=dev_cols)

for m in metrics_cols:
    for d in dev_cols:
        valid = df_metrics[[m, d]].dropna()
        if valid.shape[0] > 1 and valid[m].nunique() > 1 and valid[d].nunique() > 1:
            r, p = pearsonr(valid[m], valid[d])
            corr_matrix.loc[m, d] = r
            # Add significance stars
            if p <= 0.001:
                annot_matrix.loc[m, d] = f"{r:.2f}***"
            elif p <= 0.01:
                annot_matrix.loc[m, d] = f"{r:.2f}**"
            else:
                annot_matrix.loc[m, d] = "n/s"
        else:
            corr_matrix.loc[m, d] = np.nan
            annot_matrix.loc[m, d] = "n/s"

# Define your preferred order of metrics (y-axis)
y_order = [
    'lidar_total_m2', 'polygon_count', 'mean_patch_size',
    'patch_density', 'total_perimeter', 'area_cv', 'perimeter_cv', 'PAFRAC',
    'nLSI', 'CAI_AM', 'LSI', 'ED', 'number_of_buildings', 'built_area_total_m2',
    'mean_building_size'
]

# Reorder rows of the annotation matrix
annot_matrix = annot_matrix.reindex(y_order)

# Save annotated correlation matrix to working directory
annot_matrix.to_csv("Correlations.csv")
print("Saved annotated correlation matrix to 'Correlations.csv'")

#endregion

## ----------------------------------------------- RANDOM FOREST MODELS ------------------------------------------------
#region

# Predictor variables
model_vars = [
    'lidar_total_m2',

    # Canopy fragmentation variables
    'polygon_count',
    'mean_patch_size',
    'total_perimeter',
    'area_cv',
    'perimeter_cv',
    'LSI',

    # Building variables
    'built_area_total_m2',
    'number_of_buildings',
    'mean_building_size',
    ]

# RF and SHAP, loop across all canopy cover deviation variables
for col in df_metrics.columns:
    if col.endswith("_percent_cover_dev"):
        X = df_metrics[model_vars]
        y = df_metrics[col]

        # Fit Random Forest
        rf = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        y_pred = rf.predict(X)

        print(f"\n=== Random Forest results for {col} ===")
        print(f"R²: {r2_score(y, y_pred):.3f}")
        print(f"MAE: {mean_absolute_error(y, y_pred):.3f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.3f}")

        # Feature importance
        importances = dict(zip(model_vars, rf.feature_importances_))
        print("Feature importances:", importances)

        # --- SHAP values ---
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)

        # Global summary plot (bar chart of mean |SHAP|)
        shap.summary_plot(shap_values, X, plot_type="bar", show=True)

        # Detailed summary plot (beeswarm of all samples)
        shap.summary_plot(shap_values, X, show=True)

        # Example: SHAP force plot for first observation
        shap.initjs()
        display(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))

#endregion

## ------------------------------------------- FIGURE 1: TOTAL CANOPY COVER --------------------------------------------
#region

# Melt dataframe into long format, excluding lidar
percent_cols = [c for c in sum_by_city.columns if c.endswith("_percent_cover") and c != "lidar"]
plot_df = sum_by_city.melt(
    id_vars="city",
    value_vars=percent_cols,
    var_name="method",
    value_name="percent_cover"
)

plot_df["method_code"] = plot_df["method"].str.replace("_percent_cover", "", regex=False)

plot_df["method_code"] = plot_df["method_code"].astype(
    pd.CategoricalDtype(categories=method_order, ordered=True)
)

plt.figure(figsize=(16, 6))  # wider figure for more spacing
ax = sns.barplot(
    data=plot_df,
    x="city",
    y="percent_cover",
    hue="method_code",
    hue_order=method_order,
    palette=[method_colors[m] for m in method_order],
    order=city_order,
    dodge=True
)

# Add ALS (LiDAR) dashed lines and integer labels
for i, city in enumerate(city_order):
    als_value = sum_by_city.loc[sum_by_city["city"] == city, "lidar_percent_cover"].values[0]
    ax.hlines(y=als_value, xmin=i-0.4, xmax=i+0.4, colors="black", linestyles="--", linewidth=2)
    ax.text(i + 0.45, als_value, f"{als_value:.1f}%", va="center", ha="left", color="black", fontsize=12)

# Remove legend frame and add ALS as dashed line
handles, labels = ax.get_legend_handles_labels()
als_handle = mlines.Line2D([], [], color="black", linestyle="--", linewidth=2, label="ALS-derived UTC")

ax.legend(
    handles + [als_handle],
    [method_labels.get(m, m) for m in method_order] + ["ALS-Derived UTC"],
    frameon=False,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=12,
    title_fontsize=14
)

# X-axis labels
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_xticklabels([city_labels.get(c, c) for c in city_order], fontsize=12)

# Y-axis integer ticks
yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels([int(t) for t in yticks], fontsize=12)

# Remove figure frame/box
for spine in ax.spines.values():
    spine.set_visible(False)

plt.xlabel("")
plt.ylabel("Canopy Cover (%)", fontsize=14)

# After all customizations, save the figure
plt.tight_layout()
plt.savefig("Figure 1.png", dpi=600, bbox_inches='tight')
plt.show()

#endregion

## ----------------------------------------- FIGURE 2: CANOPY COVER HISTOGRAMS -----------------------------------------
#region

fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

# Compute JSDs relative to ALS
als_values = sum_by_city["lidar_percent_cover"].dropna().values
x_eval = np.linspace(0, 100, 500)
als_kde = gaussian_kde(als_values)
als_prob = als_kde(x_eval) / als_kde(x_eval).sum()

for ax, (group_name, methods) in zip(axes, method_groups.items()):
    for method in methods + ["lidar"]:
        values = sum_by_city[f"{method}_percent_cover"].dropna().values
        linestyle = method_linestyles.get(method, "-")
        color = method_colors[method]

        if np.unique(values).size > 1:
            kde = gaussian_kde(values)
            y_eval = kde(x_eval)
            ax.plot(x_eval, y_eval, color=color, linestyle=linestyle, linewidth=3)

            # Median line capped at KDE
            median_val = np.median(values)
            kde_at_median = kde(median_val)
            ax.vlines(median_val, 0, kde_at_median, color=color, linestyle='-', linewidth=2)

            # Calculate JSD relative to ALS (skip for ALS itself)
            if method != "lidar":
                prob_dist = y_eval / y_eval.sum()
                jsd_value = jensenshannon(prob_dist, als_prob)**2
                legend_label = f"{method_labels[method]} ({jsd_value:.3f})"
            else:
                legend_label = method_labels[method]

            ax.plot([], [], color=color, linestyle=linestyle, linewidth=2, label=legend_label)

# Axes settings
for ax in axes:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.05)
    ax.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax.set_yticklabels([f"{y:.2f}" for y in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]], fontsize=12)
    ax.set_ylabel("Density", fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=14, loc='center right', bbox_to_anchor=(0.96, 0.5))

# Second y-axis for group labels (right side)
group_labels = ["Canopy Height Models", "Fractional Canopy Cover Datasets", "Land Cover Datasets"]

for i, ax in enumerate(axes):
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([ax.get_ylim()[1]/2])
    ax2.set_yticklabels([group_labels[i]], fontsize=14, rotation=-90, va='center', fontweight='bold')
    ax2.tick_params(length=0)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

axes[-1].set_xlabel("Canopy Cover of Grids (%)", fontsize=15, labelpad=20)
plt.tight_layout()
plt.savefig("Figure 2.png", dpi=600, bbox_inches='tight')

plt.show()

#endregion

## ------------------------------------------ FIGURE 3: DEVIATION STATISTICS -------------------------------------------
#region

# Prepare results for total (all cities combined)
results_total = []

methods_to_compare = [m for m in method_order if m != "lidar"]

for method in methods_to_compare:
    # Get arrays across all cities
    y_true = sum_by_city["lidar_percent_cover"].to_numpy()
    y_pred = sum_by_city[f"{method}_percent_cover"].to_numpy()

    # Metrics
    r2 = r2_score(y_true, y_pred)
    me = np.mean(y_pred - y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    results_total.append({
        "method": method,
        "R2": r2,
        "ME": me,
        "MAE": mae,
        "RMSE": rmse
    })

total_df = pd.DataFrame(results_total)

# Melt for plotting
plot_df = total_df.melt(
    id_vars="method",
    value_vars=["R2", "ME", "MAE", "RMSE"],
    var_name="metric",
    value_name="value"
)

plt.figure(figsize=(12, 10))  # slightly wider to accommodate outside legend
ax = plt.gca()

# Map y-axis positions to methods
y_pos = np.arange(len(total_df))
ax.set_yticks(y_pos)
ax.set_yticklabels([method_labels.get(m, m) for m in total_df["method"]], fontsize=12)
ax.tick_params(axis='y', which='both', pad=15)
ax.set_ylim(-0.5, len(y_pos) - 0.5)

# Remove y-axis tick marks
ax.tick_params(axis='y', length=0)

# Background colors
ax.axhspan(y_pos[0] - 0.5, y_pos[2] + 0.5, color="#cce5ff", alpha=0.2, zorder=0)  # CHM
ax.axhspan(y_pos[3] - 0.5, y_pos[4] + 0.5, color="#d4edda", alpha=0.2, zorder=0)  # Fractional
ax.axhspan(y_pos[5] - 0.5, y_pos[8] + 0.5, color="#f8d7da", alpha=0.2, zorder=0)  # Land Cover

# Plot horizontal dashed lines and metrics
offset = 0.15
for i, method in enumerate(total_df["method"]):
    # Grey dashed lines
    ax.axhline(i + offset, color='grey', linestyle='--', linewidth=1)
    ax.axhline(i - offset, color='grey', linestyle='--', linewidth=1)

    # Bottom line: ME and MAE
    for metric in ["ME", "MAE"]:
        val = plot_df[(plot_df["method"] == method) & (plot_df["metric"] == metric)]["value"].values[0]
        ax.scatter(val, i - offset, s=200, marker=metric_markers[metric], color=metric_colors[metric],
                   label=metric_labels[metric] if i == 0 else "", zorder=3)

    # Top line: R2 and RMSE
    for metric in ["R2", "RMSE"]:
        val = plot_df[(plot_df["method"] == method) & (plot_df["metric"] == metric)]["value"].values[0]
        ax.scatter(val, i + offset, s=200, marker=metric_markers[metric], color=metric_colors[metric],
                   label=metric_labels[metric] if i == 0 else "", zorder=3)

# Vertical line at 0
ax.axvline(0, color="black", linestyle="-", linewidth=1, zorder=1)

# Second y-axis for group labels (rotated -90)
ax_right = ax.twiny()
ax_right.set_xticks([])  # Remove x-axis ticks
ax_right.set_xlabel("")  # Remove x-axis label
group_centers = [(y_pos[0] + y_pos[2]) / 2, (y_pos[3] + y_pos[4]) / 2, (y_pos[5] + y_pos[8]) / 2]
group_labels = ["Canopy Height Models", "Fractional Canopy\nCover Datasets", "Land Cover Datasets"]

for i, label in enumerate(group_labels):
    ax_right.text(
        x=1.05,  # increase from 1.02 to move further right
        y=group_centers[i],
        s=label,
        fontsize=12,
        fontweight='bold',
        rotation=-90,
        va='center',
        ha='center',
        transform=ax_right.get_yaxis_transform()  # uses axis coordinates for x
    )

ax_right.tick_params(length=0)
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.spines['bottom'].set_visible(False)

# Reverse y-axis
ax.invert_yaxis()

# Legend outside, further to the right
ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=10, frameon=False, labelspacing=1.5)

# Remove x/y labels
ax.set_xlabel("")
ax.set_ylabel("")

plt.tight_layout()
plt.savefig("Figure 3.png", dpi=600, bbox_inches='tight')
plt.show()

#endregion
