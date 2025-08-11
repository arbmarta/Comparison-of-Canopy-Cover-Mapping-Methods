import rasterio

# Access Vancouver datasets
van_eth = rasterio.open('/scratch/arbmarta/ETH/Vancouver ETH.tif')
van_lidar = rasterio.open('/scratch/arbmarta/LiDAR/Vancouver LiDAR.tif')
van_meta = rasterio.open('/scratch/arbmarta/Meta/Vancouver Meta.tif')

# Access Winnipeg datasets
win_eth = rasterio.open('/scratch/arbmarta/ETH/Winnipeg ETH.tif')
win_lidar = rasterio.open('/scratch/arbmarta/LiDAR/Winnipeg LiDAR.tif')
win_meta = rasterio.open('/scratch/arbmarta/Meta/Winnipeg Meta.tif')

# Access Ottawa datasets
ott_eth = rasterio.open('/scratch/arbmarta/ETH/Ottawa ETH.tif')
ott_lidar = rasterio.open('/scratch/arbmarta/LiDAR/Ottawa LiDAR.tif')
ott_meta = rasterio.open('/scratch/arbmarta/Meta/Ottawa Meta.tif')
