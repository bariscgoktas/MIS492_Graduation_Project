import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from esda.moran import Moran_Local
from libpysal.weights import KNN
import contextily as ctx
import folium
import json
import statsmodels.api as sm

# Step 1: Load GPKG Files and Reproject to a Suitable CRS
gpkg_files = [
    "Istanbul_UHI_Grid_2000.gpkg",
    "Istanbul_UHI_Grid_2005.gpkg",
    "Istanbul_UHI_Grid_2011.gpkg",
    "Istanbul_UHI_Grid_2016.gpkg",
    "Istanbul_UHI_Grid_2020.gpkg",
    "Istanbul_UHI_Grid_2024.gpkg"
]

gdf_list = []
for file in gpkg_files:
    gdf = gpd.read_file(file)
    gdf = gdf[['label', 'year', 'LST', 'NDVI', 'NDBI', 'geometry']]
    # Project to UTM Zone 35N (covers Istanbul, EPSG:32635)
    gdf = gdf.to_crs(epsg=32635)
    gdf_list.append(gdf)

full_gdf = pd.concat(gdf_list, ignore_index=True)
full_gdf = gpd.GeoDataFrame(full_gdf, geometry='geometry', crs='EPSG:32635')

# Step 2: Trend & Correlation Analysis
summary_list = []

for label, group in full_gdf.groupby('label'):
    group_sorted = group.sort_values('year')
    years = group_sorted['year'].values

    def get_linregress(values):
        if len(values) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(years, values)
            return slope, r_value**2, p_value
        else:
            return np.nan, np.nan, np.nan

    lst_slope, lst_r2, lst_p = get_linregress(group_sorted['LST'].values)
    ndvi_slope, _, _ = get_linregress(group_sorted['NDVI'].values)
    ndbi_slope, _, _ = get_linregress(group_sorted['NDBI'].values)

    summary_list.append({
        'label': label,
        'geometry': group_sorted.geometry.iloc[0],
        'LST_trend': lst_slope,
        'LST_R2': lst_r2,
        'LST_pval': lst_p,
        'NDVI_trend': ndvi_slope,
        'NDBI_trend': ndbi_slope,
        'LST_NDVI_corr': group_sorted['LST'].corr(group_sorted['NDVI']),
        'LST_NDBI_corr': group_sorted['LST'].corr(group_sorted['NDBI']),
        'LST_NDVI_pval': pearsonr(group_sorted['LST'], group_sorted['NDVI'])[1] if len(group_sorted) > 1 else np.nan,
        'LST_NDBI_pval': pearsonr(group_sorted['LST'], group_sorted['NDBI'])[1] if len(group_sorted) > 1 else np.nan
    })

summary_gdf = gpd.GeoDataFrame(summary_list, geometry='geometry', crs="EPSG:32635")
summary_gdf.to_file("grid_analysis_summary.gpkg", driver="GPKG")

# Step 3: Spatial Autocorrelation using Local Moran’s I
latest_year_gdf = full_gdf[full_gdf['year'] == 2024].copy()
coords = np.column_stack((latest_year_gdf.geometry.centroid.x, latest_year_gdf.geometry.centroid.y))
weights = KNN.from_array(coords, k=5)

moran_local = Moran_Local(latest_year_gdf['LST'], weights)

latest_year_gdf['Local_I'] = moran_local.Is
latest_year_gdf['Local_p'] = moran_local.p_sim
latest_year_gdf['Cluster'] = moran_local.q

latest_year_gdf.to_file("local_morans_2024.gpkg", driver="GPKG")

# Step 4: Quantile-based Classification for UHI Severity
quantiles = full_gdf['LST'].quantile([0.33, 0.66]).values
low, medium = quantiles[0], quantiles[1]

def classify(lst):
    if lst <= low:
        return 0
    elif lst <= medium:
        return 1
    else:
        return 2

full_gdf['UHI_Label'] = full_gdf['LST'].apply(classify)

# Step 5: Machine Learning Model
ml_df = full_gdf.dropna(subset=['LST', 'NDVI', 'NDBI']).copy()

X = ml_df[['NDVI', 'NDBI']]
y = ml_df['UHI_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))

full_gdf['UHI_Predicted'] = clf.predict(full_gdf[['NDVI', 'NDBI']].fillna(method='ffill'))
full_gdf['UHI_Class'] = full_gdf['UHI_Predicted'].map({0: 'Low', 1: 'Medium', 2: 'High'})

# Step 6: Interactive Map Visualization
m = folium.Map(location=[41.01, 28.97], zoom_start=10, tiles='cartodbpositron')

folium.Choropleth(
    geo_data=latest_year_gdf.to_crs("EPSG:4326"),
    data=latest_year_gdf,
    columns=['label', 'LST'],
    key_on='feature.properties.label',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='LST (°C)'
).add_to(m)

m.save("LST_Map_2024.html")

# Step 7: Export Final Results
full_gdf.to_file("final_grid_yearly_values.gpkg", driver="GPKG")

# Step 8: Citywide Correlation Insight
# This analyzes overall relationships across all grids and all years

global_df = full_gdf.dropna(subset=['LST', 'NDVI', 'NDBI'])

citywide_corr_ndvi, citywide_pval_ndvi = pearsonr(global_df['LST'], global_df['NDVI'])
citywide_corr_ndbi, citywide_pval_ndbi = pearsonr(global_df['LST'], global_df['NDBI'])

print("\n=== Citywide Correlation Summary ===")
print(f"LST vs NDVI: r = {citywide_corr_ndvi:.3f}, p = {citywide_pval_ndvi:.4f}")
print(f"LST vs NDBI: r = {citywide_corr_ndbi:.3f}, p = {citywide_pval_ndbi:.4f}")

# Step 9: Linear Regression Model

reg_df = full_gdf.dropna(subset=['LST', 'NDVI', 'NDBI']).copy()

reg_df['NDVI_x_NDBI'] = reg_df['NDVI'] * reg_df['NDBI']

X = reg_df[['NDVI', 'NDBI', 'NDVI_x_NDBI']]
X = sm.add_constant(X)
y = reg_df['LST']

model = sm.OLS(y, X).fit(cov_type='HC3')

print("\n=== LST ~ NDVI + NDBI + NDVI×NDBI Regression Results ===")
print(model.summary())

regression_summary = {
    'LST_R2': model.rsquared,
    'coef_NDVI': model.params['NDVI'],
    'coef_NDBI': model.params['NDBI'],
    'coef_NDVIxNDBI': model.params['NDVI_x_NDBI'],
    'p_NDVI': f"{model.pvalues['NDVI']:.2e}",
    'p_NDBI': f"{model.pvalues['NDBI']:.2e}",
    'p_NDVIxNDBI': f"{model.pvalues['NDVI_x_NDBI']:.2e}"
}
pd.DataFrame([regression_summary]).to_csv("regression_summary.csv", index=False)
