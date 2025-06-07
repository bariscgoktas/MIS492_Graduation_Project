import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import shape

# === Step 1: Load your CSV ===
csv_path = "Istanbul_UHI_Grid_2024_CSV.csv"  # Change path if needed
df = pd.read_csv(csv_path)

# === Step 2: Convert '.geo' column (GeoJSON string) into shapely geometries ===
geometries = df['.geo'].apply(lambda x: shape(json.loads(x)))

# === Step 3: Create a GeoDataFrame ===
gdf = gpd.GeoDataFrame(df.drop(columns=['.geo']), geometry=geometries)

# === Step 4: Set Coordinate Reference System (WGS 84 is standard for lat/lon) ===
gdf.set_crs(epsg=4326, inplace=True)

# === Step 5: Save as GeoPackage (or .shp if needed) ===
gdf.to_file("Istanbul_UHI_Grid_2024.gpkg", driver="GPKG")

# Save as shapefile instead
# gdf.to_file("Istanbul_UHI_Grid_2000.shp")
