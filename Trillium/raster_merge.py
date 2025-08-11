import rasterio
from rasterio.merge import merge

inputs = [
    "/scratch/arbmarta/ETH/Vancouver ETH 1.tif",
    "/scratch/arbmarta/ETH/Vancouver ETH 2.tif",
]
out = "/scratch/arbmarta/ETH/Vancouver ETH new.tif"

srcs = [rasterio.open(p) for p in inputs]
mosaic, transform = merge(srcs, nodata=0)  # simple side-by-side mosaic
profile = srcs[0].profile
profile.update(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
               transform=transform, count=1, compress="LZW", tiled=True, bigtiff="IF_SAFER", nodata=0)

with rasterio.open(out, "w", **profile) as dst:
    dst.write(mosaic[0], 1)

for s in srcs: s.close()
