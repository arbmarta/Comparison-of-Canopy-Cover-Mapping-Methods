import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from multiprocessing import Pool
import pandas as pd 

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

OUT_SPLIT = "/scratch/arbmarta/{kind}/{city} {kind} canopy_SPLIT.shp"
CANOPY_THRESHOLD = 2  # value >= threshold becomes canopy (1)

# Reuse your existing config dicts
# CITIES = {...}  # includes "utm" and "bayan" for each city
# OUT_SPLIT = "/scratch/arbmarta/{kind}/{city} {kind} canopy_SPLIT.shp"

def summarize_city_to_csv(city: str, cfg: dict, out_csv: str) -> None:
    """
    For a city: compute area (m²) of split polygons for ETH and Meta,
    sum by 120x120 Bayan grid cell, and write a CSV with original Bayan columns + ETH + Meta.
    """
    utm = cfg["utm"]
    bayan_path = cfg["bayan"]

    # Load Bayan grid (authoritative attributes)
    bayan = gpd.read_file(bayan_path).to_crs(utm)
    bayan = bayan.reset_index(drop=False).rename(columns={"index": "cell_id"})
    # Keep attributes to return (all original Bayan columns + synthetic cell_id)
    bayan_attrs = bayan.drop(columns="geometry").copy()

    results = {}

    for kind in ["ETH", "Meta"]:
        split_path = OUT_SPLIT.format(kind=kind, city=city)
        gdf = gpd.read_file(split_path)
        if gdf.empty:
            # No canopy polygons for this kind; create empty series aligned later
            results[kind] = pd.Series(dtype="float64", name=kind)
            continue

        gdf = gdf.to_crs(utm)

        # Assign each split polygon to a Bayan cell (robust even if split wasn't perfect)
        # Using 'within' to avoid slivers; fallback to 'intersects' if needed.
        joined = gpd.sjoin(
            gdf.set_geometry("geometry"),
            bayan[["cell_id", "geometry"]],
            predicate="within",
            how="left",
        )

        # Compute area in square meters (UTM is meters)
        joined["area_m2"] = joined.geometry.area

        # Sum area by cell_id
        sums = (
            joined.groupby("cell_id", dropna=False)["area_m2"]
            .sum()
            .rename(kind)
        )
        results[kind] = sums

    # Merge ETH and Meta onto full Bayan attribute table
    out = bayan_attrs.merge(results["ETH"], left_on="cell_id", right_index=True, how="left")
    out = out.merge(results["Meta"], left_on="cell_id", right_index=True, how="left")

    # Replace NaNs (no canopy in that cell) with zeros
    out[["ETH", "Meta"]] = out[["ETH", "Meta"]].fillna(0.0)

    # Write CSV
    out.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv} with {len(out)} rows")

# --- Run for the three cities ---
summarize_city_to_csv("Vancouver", CITIES["Vancouver"], "/scratch/arbmarta/Vancouver_bayan_canopy_sums.csv")
summarize_city_to_csv("Winnipeg",  CITIES["Winnipeg"],  "/scratch/arbmarta/Winnipeg_bayan_canopy_sums.csv")
summarize_city_to_csv("Ottawa",    CITIES["Ottawa"],    "/scratch/arbmarta/Ottawa_bayan_canopy_sums.csv")
