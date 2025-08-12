import os
import geopandas as gpd
import pandas as pd
from multiprocessing import Pool

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

OUT_SPLIT = "/scratch/arbmarta/{kind}/{city} {kind} canopy_SPLIT.shp"
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
        split_path = OUT_SPLIT.format(kind=kind, city=city)
        if not os.path.exists(split_path):
            print(f"⚠️ {city} {kind}: split shapefile not found. Filling zeros.")
            results[kind] = pd.Series(dtype="float64", name=kind)
            continue

        gdf = gpd.read_file(split_path)
        if gdf.empty:
            print(f"⚠️ {city} {kind}: split shapefile is empty. Filling zeros.")
            results[kind] = pd.Series(dtype="float64", name=kind)
            continue

        # Ensure UTM CRS
        gdf = gdf.to_crs(utm)

        # --- Spatial join: expect EXACTLY ONE Bayan cell per split polygon ---
        joined = gpd.sjoin(
            gdf.set_geometry("geometry"),
            bayan[["cell_id", "geometry"]],
            predicate="within",
            how="left",
        )

        # Retry unmatched polygons with 'intersects' (rare precision/edge cases)
        unmatched = joined["cell_id"].isna()
        if unmatched.any():
            fix = gpd.sjoin(
                gdf.loc[unmatched],
                bayan[["cell_id", "geometry"]],
                predicate="intersects",
                how="left",
            )["cell_id"].values
            joined.loc[unmatched, "cell_id"] = fix
            if joined["cell_id"].isna().any():
                missing = int(joined["cell_id"].isna().sum())
                raise RuntimeError(
                    f"{city} {kind}: {missing} polygons didn't match any Bayan cell after retry."
                )

        # Ensure no polygon matched multiple cells
        counts = joined.groupby(joined.index)["cell_id"].nunique(dropna=True)
        if (counts > 1).any():
            n_multi = int((counts > 1).sum())
            raise RuntimeError(
                f"{city} {kind}: {n_multi} polygons matched multiple Bayan cells (split integrity issue)."
            )

        # Compute area (m²) and aggregate by Bayan cell
        joined["area_m2"] = joined.geometry.area
        sums = (
            joined.groupby("cell_id", dropna=False)["area_m2"]
            .sum()
            .rename(kind)
        )
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
