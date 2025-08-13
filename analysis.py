import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import jensenshannon
import numpy as np
from scipy.stats import pearsonr, spearmanr

## ------------------------------------- IMPORT AND BASIC CANOPY COVER CALCULATION -------------------------------------
#region

# Import the datasets
van_eth = pd.read_csv('Outputs/Vancouver_ETH.csv')
van_meta = pd.read_csv('Outputs/Vancouver_Meta.csv')
van_lidar = pd.read_csv('Outputs/Vancouver_LiDAR.csv')

win_eth = pd.read_csv('Outputs/Winnipeg_ETH.csv')
win_meta = pd.read_csv('Outputs/Winnipeg_Meta.csv')
win_lidar = pd.read_csv('Outputs/Winnipeg_LiDAR.csv')

ott_eth = pd.read_csv('Outputs/Ottawa_ETH.csv')
ott_meta = pd.read_csv('Outputs/Ottawa_Meta.csv')
ott_lidar = pd.read_csv('Outputs/Ottawa_LiDAR.csv')

# Rename columns - percent_cover
van_eth.rename(columns={'percent_cover': 'ETH_percent_cover'}, inplace=True)
van_meta.rename(columns={'percent_cover': 'Meta_percent_cover'}, inplace=True)
van_lidar.rename(columns={'percent_cover': 'LiDAR_percent_cover'}, inplace=True)

win_eth.rename(columns={'percent_cover': 'ETH_percent_cover'}, inplace=True)
win_meta.rename(columns={'percent_cover': 'Meta_percent_cover'}, inplace=True)
win_lidar.rename(columns={'percent_cover': 'LiDAR_percent_cover'}, inplace=True)

ott_eth.rename(columns={'percent_cover': 'ETH_percent_cover'}, inplace=True)
ott_meta.rename(columns={'percent_cover': 'Meta_percent_cover'}, inplace=True)
ott_lidar.rename(columns={'percent_cover': 'LiDAR_percent_cover'}, inplace=True)

# Rename columns - total_m2
van_eth.rename(columns={'total_m2': 'ETH_total_m2'}, inplace=True)
van_meta.rename(columns={'total_m2': 'Meta_total_m2'}, inplace=True)
van_lidar.rename(columns={'total_m2': 'LiDAR_total_m2'}, inplace=True)

win_eth.rename(columns={'total_m2': 'ETH_total_m2'}, inplace=True)
win_meta.rename(columns={'total_m2': 'Meta_total_m2'}, inplace=True)
win_lidar.rename(columns={'total_m2': 'LiDAR_total_m2'}, inplace=True)

ott_eth.rename(columns={'total_m2': 'ETH_total_m2'}, inplace=True)
ott_meta.rename(columns={'total_m2': 'Meta_total_m2'}, inplace=True)
ott_lidar.rename(columns={'total_m2': 'LiDAR_total_m2'}, inplace=True)

# Merge on 'grid_id' for each city and simplify columns
van_merged = van_eth.merge(van_meta, on='grid_id', how='outer').merge(van_lidar, on='grid_id', how='outer')
van_merged = van_merged[[
    'grid_id', 'pred',
    'ETH_total_m2', 'ETH_percent_cover',
    'Meta_total_m2', 'Meta_percent_cover',
    'LiDAR_total_m2', 'LiDAR_percent_cover'
]]

win_merged = win_eth.merge(win_meta, on='grid_id', how='outer').merge(win_lidar, on='grid_id', how='outer')
win_merged = win_merged[[
    'grid_id', 'pred',
    'ETH_total_m2', 'ETH_percent_cover',
    'Meta_total_m2', 'Meta_percent_cover',
    'LiDAR_total_m2', 'LiDAR_percent_cover'
]]

ott_merged = ott_eth.merge(ott_meta, on='grid_id', how='outer').merge(ott_lidar, on='grid_id', how='outer')
ott_merged = ott_merged[[
    'grid_id', 'pred',
    'ETH_total_m2', 'ETH_percent_cover',
    'Meta_total_m2', 'Meta_percent_cover',
    'LiDAR_total_m2', 'LiDAR_percent_cover'
]]

# Rename 'pred' to 'Bayan_percent_cover' in each merged DataFrame
van_merged.rename(columns={'pred': 'Bayan_percent_cover'}, inplace=True)
win_merged.rename(columns={'pred': 'Bayan_percent_cover'}, inplace=True)
ott_merged.rename(columns={'pred': 'Bayan_percent_cover'}, inplace=True)

# Compute city areas (120m x 120m grid cells)
van_area = van_merged['grid_id'].nunique() * (120 * 120)
win_area = win_merged['grid_id'].nunique() * (120 * 120)
ott_area = ott_merged['grid_id'].nunique() * (120 * 120)

# Calculate overall canopy cover - rasters
van_lidar_cc = van_merged['LiDAR_total_m2'].sum() / van_area * 100
win_lidar_cc = win_merged['LiDAR_total_m2'].sum() / win_area * 100
ott_lidar_cc = ott_merged['LiDAR_total_m2'].sum() / ott_area * 100

van_meta_cc = van_merged['Meta_total_m2'].sum() / van_area * 100
win_meta_cc = win_merged['Meta_total_m2'].sum() / win_area * 100
ott_meta_cc = ott_merged['Meta_total_m2'].sum() / ott_area * 100

van_eth_cc = van_merged['ETH_total_m2'].sum() / van_area * 100
win_eth_cc = win_merged['ETH_total_m2'].sum() / win_area * 100
ott_eth_cc = ott_merged['ETH_total_m2'].sum() / ott_area * 100

# Calculate overall canopy cover - Bayan
van_merged['Bayan_total_m2'] = van_merged['Bayan_percent_cover'] / 100 * (120 * 120)
win_merged['Bayan_total_m2'] = win_merged['Bayan_percent_cover'] / 100 * (120 * 120)
ott_merged['Bayan_total_m2'] = ott_merged['Bayan_percent_cover'] / 100 * (120 * 120)

van_bayan_cc = van_merged['Bayan_total_m2'].sum() / van_area * 100
win_bayan_cc = win_merged['Bayan_total_m2'].sum() / win_area * 100
ott_bayan_cc = ott_merged['Bayan_total_m2'].sum() / ott_area * 100

# Print overall canopy cover
print("--- Vancouver Canopy Cover ---")
print(f"Vancouver LiDAR canopy cover: {van_lidar_cc:.2f}%")
print(f"Vancouver Meta canopy cover: {van_meta_cc:.2f}%")
print(f"Vancouver Bayan canopy cover: {van_bayan_cc:.2f}%")
print(f"Vancouver ETH canopy cover: {van_eth_cc:.2f}%\n")

print("--- Winnipeg Canopy Cover ---")
print(f"Winnipeg LiDAR canopy cover: {win_lidar_cc:.2f}%")
print(f"Winnipeg Meta canopy cover: {win_meta_cc:.2f}%")
print(f"Winnipeg Bayan canopy cover: {win_bayan_cc:.2f}%")
print(f"Winnipeg ETH canopy cover: {win_eth_cc:.2f}%\n")

print("--- Ottawa Canopy Cover ---")
print(f"Ottawa LiDAR canopy cover: {ott_lidar_cc:.2f}%")
print(f"Ottawa Meta canopy cover: {ott_meta_cc:.2f}%")
print(f"Ottawa Bayan canopy cover: {ott_bayan_cc:.2f}%")
print(f"Ottawa ETH canopy cover: {ott_eth_cc:.2f}%")

#endregion

## ---------------------------------------------- CALCULATE RMSE AND JSD -----------------------------------------------
#region

def safe_jsd(p, q):
    """Normalize and compute JSD; returns np.nan if sum is 0."""
    p = np.array(p)
    q = np.array(q)
    p = np.nan_to_num(p, nan=0.0)
    q = np.nan_to_num(q, nan=0.0)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return jensenshannon(p, q)

# Set NaN to 0
van_merged[['LiDAR_percent_cover', 'Meta_percent_cover', 'ETH_percent_cover', 'Bayan_percent_cover']] = \
    van_merged[['LiDAR_percent_cover', 'Meta_percent_cover', 'ETH_percent_cover', 'Bayan_percent_cover']].fillna(0)

win_merged[['LiDAR_percent_cover', 'Meta_percent_cover', 'ETH_percent_cover', 'Bayan_percent_cover']] = \
    win_merged[['LiDAR_percent_cover', 'Meta_percent_cover', 'ETH_percent_cover', 'Bayan_percent_cover']].fillna(0)

ott_merged[['LiDAR_percent_cover', 'Meta_percent_cover', 'ETH_percent_cover', 'Bayan_percent_cover']] = \
    ott_merged[['LiDAR_percent_cover', 'Meta_percent_cover', 'ETH_percent_cover', 'Bayan_percent_cover']].fillna(0)

# Vancouver
van_rmse_meta = np.sqrt(mean_squared_error(van_merged['LiDAR_percent_cover'], van_merged['Meta_percent_cover']))
van_rmse_eth = np.sqrt(mean_squared_error(van_merged['LiDAR_percent_cover'], van_merged['ETH_percent_cover']))
van_rmse_bayan = np.sqrt(mean_squared_error(van_merged['LiDAR_percent_cover'], van_merged['Bayan_percent_cover']))

van_jsd_meta = safe_jsd(van_merged['LiDAR_percent_cover'], van_merged['Meta_percent_cover'])
van_jsd_eth = safe_jsd(van_merged['LiDAR_percent_cover'], van_merged['ETH_percent_cover'])
van_jsd_bayan = safe_jsd(van_merged['LiDAR_percent_cover'], van_merged['Bayan_percent_cover'])

# Winnipeg
win_rmse_meta = np.sqrt(mean_squared_error(win_merged['LiDAR_percent_cover'], win_merged['Meta_percent_cover']))
win_rmse_eth = np.sqrt(mean_squared_error(win_merged['LiDAR_percent_cover'], win_merged['ETH_percent_cover']))
win_rmse_bayan = np.sqrt(mean_squared_error(win_merged['LiDAR_percent_cover'], win_merged['Bayan_percent_cover']))

win_jsd_meta = safe_jsd(win_merged['LiDAR_percent_cover'], win_merged['Meta_percent_cover'])
win_jsd_eth = safe_jsd(win_merged['LiDAR_percent_cover'], win_merged['ETH_percent_cover'])
win_jsd_bayan = safe_jsd(win_merged['LiDAR_percent_cover'], win_merged['Bayan_percent_cover'])

# Ottawa
ott_rmse_meta = np.sqrt(mean_squared_error(ott_merged['LiDAR_percent_cover'], ott_merged['Meta_percent_cover']))
ott_rmse_eth = np.sqrt(mean_squared_error(ott_merged['LiDAR_percent_cover'], ott_merged['ETH_percent_cover']))
ott_rmse_bayan = np.sqrt(mean_squared_error(ott_merged['LiDAR_percent_cover'], ott_merged['Bayan_percent_cover']))

ott_jsd_meta = safe_jsd(ott_merged['LiDAR_percent_cover'], ott_merged['Meta_percent_cover'])
ott_jsd_eth = safe_jsd(ott_merged['LiDAR_percent_cover'], ott_merged['ETH_percent_cover'])
ott_jsd_bayan = safe_jsd(ott_merged['LiDAR_percent_cover'], ott_merged['Bayan_percent_cover'])

# Print the summary
print("\n--- RMSE and JSD Statistics ---")
print(f"Vancouver RMSE - Meta: {van_rmse_meta:.2f}, ETH: {van_rmse_eth:.2f}, Bayan: {van_rmse_bayan:.2f}")
print(f"Vancouver JSD  - Meta: {van_jsd_meta:.4f}, ETH: {van_jsd_eth:.4f}, Bayan: {van_jsd_bayan:.4f}\n")

print(f"Winnipeg RMSE  - Meta: {win_rmse_meta:.2f}, ETH: {win_rmse_eth:.2f}, Bayan: {win_rmse_bayan:.2f}")
print(f"Winnipeg JSD   - Meta: {win_jsd_meta:.4f}, ETH: {win_jsd_eth:.4f}, Bayan: {win_jsd_bayan:.4f}\n")

print(f"Ottawa RMSE    - Meta: {ott_rmse_meta:.2f}, ETH: {ott_rmse_eth:.2f}, Bayan: {ott_rmse_bayan:.2f}")
print(f"Ottawa JSD     - Meta: {ott_jsd_meta:.4f}, ETH: {ott_jsd_eth:.4f}, Bayan: {ott_jsd_bayan:.4f}")

#endregion

## --------------------------------- PREDICTORS OF DEVIATION: CORRELATION COEFFICIENT ----------------------------------
#region

print("\n--- Correlation Coefficients ---")

# Calculate deviation from the true canopy cover (LiDAR)
# Vancouver
van_merged['ETH_deviation'] = van_merged['ETH_percent_cover'] - van_merged['LiDAR_percent_cover']
van_merged['ETH_abs_deviation'] = van_merged['ETH_deviation'].abs()

van_merged['Meta_deviation'] = van_merged['Meta_percent_cover'] - van_merged['LiDAR_percent_cover']
van_merged['Meta_abs_deviation'] = van_merged['Meta_deviation'].abs()

van_merged['Bayan_deviation'] = van_merged['Bayan_percent_cover'] - van_merged['LiDAR_percent_cover']
van_merged['Bayan_abs_deviation'] = van_merged['Bayan_deviation'].abs()

# Winnipeg
win_merged['ETH_deviation'] = win_merged['ETH_percent_cover'] - win_merged['LiDAR_percent_cover']
win_merged['ETH_abs_deviation'] = win_merged['ETH_deviation'].abs()

win_merged['Meta_deviation'] = win_merged['Meta_percent_cover'] - win_merged['LiDAR_percent_cover']
win_merged['Meta_abs_deviation'] = win_merged['Meta_deviation'].abs()

win_merged['Bayan_deviation'] = win_merged['Bayan_percent_cover'] - win_merged['LiDAR_percent_cover']
win_merged['Bayan_abs_deviation'] = win_merged['Bayan_deviation'].abs()

# Ottawa
ott_merged['ETH_deviation'] = ott_merged['ETH_percent_cover'] - ott_merged['LiDAR_percent_cover']
ott_merged['ETH_abs_deviation'] = ott_merged['ETH_deviation'].abs()

ott_merged['Meta_deviation'] = ott_merged['Meta_percent_cover'] - ott_merged['LiDAR_percent_cover']
ott_merged['Meta_abs_deviation'] = ott_merged['Meta_deviation'].abs()

ott_merged['Bayan_deviation'] = ott_merged['Bayan_percent_cover'] - ott_merged['LiDAR_percent_cover']
ott_merged['Bayan_abs_deviation'] = ott_merged['Bayan_deviation'].abs()

# Correlation coefficient functions
def format_p(p):
    return "< .001" if p < 0.001 else round(p, 3)


def correlation_summary(df, city_name):
    results = []
    for col in ['ETH_deviation', 'ETH_abs_deviation',
                'Meta_deviation', 'Meta_abs_deviation',
                'Bayan_deviation', 'Bayan_abs_deviation']:
        pearson_corr, pearson_p = pearsonr(df[col], df['LiDAR_percent_cover'])
        spearman_corr, spearman_p = spearmanr(df[col], df['LiDAR_percent_cover'])

        results.append({
            'City': city_name,
            'Variable': col,
            'Pearson_r': round(pearson_corr, 3),
            'Pearson_p': format_p(pearson_p),
            'Spearman_rho': round(spearman_corr, 3),
            'Spearman_p': format_p(spearman_p)
        })
    return results

# Run correlations for each city
van_corr = correlation_summary(van_merged, 'Vancouver')
win_corr = correlation_summary(win_merged, 'Winnipeg')
ott_corr = correlation_summary(ott_merged, 'Ottawa')

corr_df = pd.DataFrame(van_corr + win_corr + ott_corr)

# Display results
print(corr_df)

#endregion

## ---------------------------------------------- PREDICTORS OF DEVIATION ----------------------------------------------
#region



#endregion
