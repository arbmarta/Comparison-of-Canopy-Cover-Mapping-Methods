import geopandas as gpd
import matplotlib.pyplot as plt

## ---------------------------------------- IMPORT CITY BOUNDARY FILES USING AOI ---------------------------------------
#region

# Dictionary of city names, shapefile paths, and their UTM EPSG codes
city_info = {
    "Vancouver": {"path": "City Boundaries/Vancouver.shp", "epsg": 32610},
    "Winnipeg": {"path": "City Boundaries/Winnipeg.shp", "epsg": 32614},
    "Ottawa": {"path": "City Boundaries/Ottawa.shp", "epsg": 32618}
}

# Reproject, print CRS, and plot
for city, info in city_info.items():
    gdf = gpd.read_file(info["path"])
    gdf_utm = gdf.to_crs(epsg=info["epsg"])
    print(f"{city} reprojected to CRS: {gdf_utm.crs}")

    # Plot
    gdf_utm.plot(edgecolor='black', facecolor='lightblue', figsize=(8, 8))
    plt.title(f"{city} City Boundary (UTM Zone)")
    plt.axis('off')
    plt.show()

#endregion
