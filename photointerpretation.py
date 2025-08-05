import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Read the photo-interpretation csv
wpg_itree = pd.read_csv('Winnipeg\Winnipeg Photointerpretation.csv')
wpg_itree_lidar = pd.read_csv('Winnipeg\Winnipeg Photointerpretation LiDAR.csv')

print(wpg_itree.columns)

## -------------------------------------------------- CREATE HISTOGRAM -------------------------------------------------
#region

# Clean the dataset (don't shuffle, bootstrap handles randomization)
wpg_itree_clean = wpg_itree.dropna(subset=['Canopy'])

# Convert to percent once to reduce repeated operations
canopy_data = wpg_itree_clean['Canopy'].values * 100

# Bootstrap function
def bootstrap_means(data, sample_size, n_bootstrap=10000, seed=42):
    rng = np.random.default_rng(seed)
    return [np.mean(rng.choice(data, size=sample_size, replace=True)) for _ in range(n_bootstrap)]

# Sample sizes to evaluate
sample_sizes = [500, 750, 1000, 2500, 5000, 10000]

# Full sample mean for reference
full_sample_mean = canopy_data.mean()

# Plot
plt.figure(figsize=(12, 7))

for size in sample_sizes:
    boot_means = bootstrap_means(canopy_data, sample_size=size)
    sns.kdeplot(boot_means, label=f'{size}-point Samples', fill=True, alpha=0.3, linewidth=2)

# Vertical line: full dataset mean
plt.axvline(full_sample_mean, color='black', linestyle='--', linewidth=2, label=f'Full Sample Mean ({full_sample_mean:.2f}%)')

# Final styling
plt.title("Bootstrap Distributions of Estimated Canopy Cover")
plt.xlabel("Estimated Canopy Cover (%)")
plt.ylabel("Density")
plt.legend(title="Sample Size")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# endregion
