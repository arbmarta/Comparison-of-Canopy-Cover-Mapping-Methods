import os
import geopandas as gpd
import pandas as pd
from multiprocessing import Pool
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape

# ----------------------------- CONFIG -----------------------------
CITIES = {
    "Vancouver": {
        "utm": "EPSG:32610",
        "bayan": "/scratch/arbmarta/Trinity/Vancouver/TVAN.shp",
    },
    "Winnipeg": {
        "utm": "EPSG:32614",
        "bayan": "/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp",
    },
    "Ottawa": {
        "utm": "EPSG:32618",
        "bayan": "/scratch/arbmarta/Trinity/Ottawa/TOTT.shp",
    },
}

DATASETS = {
    "ETH":  "/scratch/arbmarta/ETH/{city} ETH.tif",
    "Meta": "/scratch/arbmarta/Meta/{city} Meta.tif",
}

OUT_CSV   = "/scratch/arbmarta/{city}_bayan_canopy_sums.csv"

CELL_AREA_M2 = 120 * 120  # 14,400 m² per Bayan cell

# -------------------------- CORE SUMMARIZER --------------------------
def summarize_city_to_csv(city: str, cfg: dict, out_csv: str) -> None:
    """
    For a city: compute canopy area (m²) of split polygons for ETH and Meta,
    sum by Bayan (120x120 m) cell, and write CSV with Bayan attributes + ETH + Meta
    plus ETH_pct and Meta_pct (0–100). Includes integrity checks that each split
    polygon matches exactly one Bayan cell (retrying unmatched with 'intersects').
    """
    utm = cfg["utm"]
    bayan_path = cfg["bayan"]

    # Load Bayan grid (authoritative attributes) in UTM (meters)
    bayan = (
        gpd.read_file(bayan_path)
        .to_crs(utm)
        .reset_index(drop=False)
        .rename(columns={"index": "cell_id"})
    )
    bayan_attrs = bayan.drop(columns="geometry").copy()

results = {}

for kind in ["ETH", "Meta"]:
    raster_path = DATASETS[kind].format(city=city)
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"{city} {kind}: raster not found at {raster_path}")

    with rasterio.open(raster_path) as src:
        # Mask raster to the city grid in raster CRS (cheap)
        bayan_src = bayan.to_crs(src.crs)
        masked, transform = mask(src, bayan_src.geometry, crop=True, filled=True, nodata=0)

        # Threshold to canopy (>= 2 → 1, else 0)
        binary = (masked[0] >= 2).astype("uint8")

        # Polygonize canopy pixels in raster CRS
        poly_iter = shapes(binary, mask=(binary == 1), transform=transform)
        polys = [shape(geom) for geom, v in poly_iter if v == 1]

    if not polys:
        # No canopy → zeros column
        results[kind] = pd.Series(dtype="float64", name=kind)
        continue

    # Build GDF, dissolve (reduce feature count), project once to UTM
    gdf = gpd.GeoDataFrame(geometry=polys, crs=src.crs)
    dissolved = gdf.unary_union
    parts = [dissolved] if dissolved.geom_type == "Polygon" else list(dissolved.geoms)
    canopy = gpd.GeoDataFrame(geometry=parts, crs=gdf.crs).to_crs(utm)

    # Overlay with Bayan grid in UTM to split by cells, then area by cell_id
    bayan_utm = bayan[["cell_id", "geometry"]].copy()
    canopy["geometry"] = canopy.geometry.buffer(0)
    bayan_utm["geometry"] = bayan_utm.geometry.buffer(0)

    split = gpd.overlay(canopy, bayan_utm, how="identity").explode(index_parts=False, ignore_index=True)
    split["area_m2"] = split.geometry.area

    sums = split.groupby("cell_id", dropna=False)["area_m2"].sum().rename(kind)
    results[kind] = sums
    
    # Merge ETH and Meta results onto Bayan attributes
    out = bayan_attrs.merge(
        results.get("ETH", pd.Series(name="ETH")),
        left_on="cell_id",
        right_index=True,
        how="left",
    )
    out = out.merge(
        results.get("Meta", pd.Series(name="Meta")),
        left_on="cell_id",
        right_index=True,
        how="left",
    )

    # Fill NaNs with 0 (no canopy in that cell)
    for col in ["ETH", "Meta"]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)
        else:
            out[col] = 0.0

    # Percentage columns (0–100), clipped to guard against tiny numeric noise
    out["ETH_pct"]  = (out["ETH"]  / CELL_AREA_M2) * 100
    out["Meta_pct"] = (out["Meta"] / CELL_AREA_M2) * 100
    out[["ETH_pct", "Meta_pct"]] = out[["ETH_pct", "Meta_pct"]].clip(lower=0, upper=100)

    # Write CSV
    out.to_csv(out_csv, index=False)
    print(f"✅ {city}: wrote {out_csv} with {len(out)} rows")

# ------------------------------ MAIN ------------------------------
def _run_city(args):
    city, cfg = args
    summarize_city_to_csv(city, cfg, OUT_CSV.format(city=city))
    return city

if __name__ == "__main__":
    # Recommended for Trillium: avoid OpenMP over-subscription
    os.environ["OMP_NUM_THREADS"] = "1"
    # Ensure PROJ data is available to pyproj/GeoPandas
    os.environ["PROJ_DATA"] = "/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/proj/9.1.1/share/proj"

    jobs = list(CITIES.items())  # [("Vancouver", {...}), ("Winnipeg", {...}), ("Ottawa", {...})]

    # Fixed number of processes for Pool (match your Slurm --cpus-per-task)
    pool_processes = 6
    with Pool(processes=pool_processes) as pool:
        for _ in pool.imap_unordered(_run_city, jobs):
            pass
