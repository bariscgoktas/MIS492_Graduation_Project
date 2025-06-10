import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from streamlit_folium import st_folium
import folium
import branca.colormap as cm
import json
from scipy.stats import pearsonr


st.set_page_config(page_title="UHI Dashboard", layout="wide")
st.title("Urban Heat Island Dashboard – Istanbul")

if 'selected_grid' not in st.session_state:
    st.session_state.selected_grid = None

# Load spatial data from GeoPackages
df = gpd.read_file("final_grid_yearly_values.gpkg").to_crs("EPSG:4326")
summary = gpd.read_file("grid_analysis_summary.gpkg").to_crs("EPSG:4326")

# Load Moran's I results
try:
    with open("morans_results.json") as f:
        moran_data = json.load(f)
        morans_i = moran_data["I"]
        morans_p = moran_data["p_sim"]
except FileNotFoundError:
    morans_i = None
    morans_p = None

# Prepare data
latest_year = df['year'].max()
avg_lst = df.groupby('label')['LST'].mean().reset_index().rename(columns={'LST': 'avg_LST'})
geo_df = df[df['year'] == latest_year].copy()
geo_df = geo_df.merge(avg_lst, on='label', how='left')

# Sidebar Filtering
st.sidebar.header("Grid Filtering")

filtered_df = df[df['UHI_Class'].isin(selected_class)]
year_options = sorted(df['year'].unique())
selected_years = st.sidebar.multiselect("Select Year(s)", options=year_options, default=year_options)
df_year_filtered = filtered_df[filtered_df['year'].isin(selected_years)]

# Spatial Autocorrelation Info
st.sidebar.header("Spatial Autocorrelation")
if morans_i is not None and morans_p is not None:
    st.sidebar.markdown(f"**Moran's I (2024):** {morans_i:.3f}")
    st.sidebar.markdown(f"**p-value:** {morans_p:.4f}")
    if morans_p < 0.05:
        st.sidebar.success("Significant spatial clustering")
    else:
        st.sidebar.info("No significant clustering detected")
else:
    st.sidebar.warning("Moran's I data not available")

# Combined Map + Grid Analysis
st.subheader("Interactive UHI Grid Map & Grid-Level Analysis")
col_map, col_analysis = st.columns(2)

color_dict = {
    'High': '#d73027',
    'Medium': '#fc8d59',
    'Low': '#91bfdb'
}

legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
padding: 10px; border: 1px solid grey; border-radius: 5px;">
<p><b>UHI Class</b></p>
<p><i class="fa fa-square" style="color:#d73027"></i> High</p>
<p><i class="fa fa-square" style="color:#fc8d59"></i> Medium</p>
<p><i class="fa fa-square" style="color:#91bfdb"></i> Low</p>
</div>
'''

with col_map:
    m = folium.Map(location=[41.0082, 28.9784], zoom_start=10)

    def style_function(feature):
        uhi_class = feature['properties'].get('UHI_Class', 'None')
        return {
            'fillColor': color_dict.get(uhi_class, '#f0f0f0'),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }

    folium.GeoJson(
        geo_df.to_json(),
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['label', 'UHI_Class'],
            aliases=['Grid ID:', 'UHI Class:'],
            localize=True
        )
    ).add_to(m)

    m.get_root().html.add_child(folium.Element(legend_html))
    map_data = st_folium(m, width=600, height=500)

    # Correlation-Based Recommendation Message
    selected = st.session_state.selected_grid
    if selected is not None:
        row = summary[summary['label'] == selected]
        if not row.empty:
            corr_ndvi = row['LST_NDVI_corr'].values[0]
            corr_ndbi = row['LST_NDBI_corr'].values[0]

            ndvi_valid = pd.notnull(corr_ndvi)
            ndbi_valid = pd.notnull(corr_ndbi)

            # 1) If both are valid, show both messages side by side
            if ndvi_valid and ndbi_valid:
                # Build NDVI message
                if corr_ndvi < -0.5:
                    ndvi_msg = (
                        f"Strong negative correlation (r = {corr_ndvi:.2f}) between LST and NDVI:  \n"
                        " - This suggests that as vegetation cover increases, surface temperature drops significantly.  \n"
                        " - Consider substantially increasing green spaces (parks, tree canopy) to mitigate UHI effects here."
                    )
                elif corr_ndvi < 0:
                    ndvi_msg = (
                        f"Moderate negative correlation (r = {corr_ndvi:.2f}) between LST and NDVI: \n"
                        " - There is some indication that adding vegetation lowers land‐surface temperature.  \n"
                        " - Increasing green cover (e.g. street trees, green roofs) could help reduce heat in this grid."
                    )
                else:
                    ndvi_msg = (
                        f"Vegetation cover does not show a clear inverse relationship with temperature. (r = {corr_ndvi:.2f})  \n"
                    )

                # Build NDBI message
                if corr_ndbi > 0.5:
                    ndbi_msg = (
                        f"Strong positive correlation (r = {corr_ndbi:.2f}) between LST and NDBI:  \n"
                         " - This indicates that more built‐up surfaces correlate strongly with higher temperatures.  \n"
                         " - Consider reducing impervious surfaces (e.g. percent of concrete/asphalt) or using high‐albedo materials."
                    )
                elif corr_ndbi > 0:
                    ndbi_msg = (
                        f"Moderate positive correlation (r = {corr_ndbi:.2f}) between LST and NDBI: \n"
                                " - Built‐up intensity shows some link to higher land‐surface temperatures.  \n"
                                " - Explore methods to limit urban fabric heat absorption—e.g. cool pavements or reflective roofing."
                    )
                else:
                    ndbi_msg = (
                        f"Built‐up index does not show a clear direct relationship with temperature. (r = {corr_ndbi:.2f}) \n"
                    )

                # Display both NDVI and NDBI messages
                st.markdown(
                    f"{ndvi_msg}\n\n---\n\n{ndbi_msg}"
                )

            # 2) If only NDVI is valid, show NDVI-based message
            elif ndvi_valid:
                if corr_ndvi < -0.5:
                    st.markdown(
                        f"Strong negative correlation (r = {corr_ndvi:.2f}) between LST and NDVI:  \n"
                        " - This suggests that as vegetation cover increases, surface temperature drops significantly.  \n"
                        " - Consider substantially increasing green spaces (parks, tree canopy) to mitigate UHI effects here."
                    )
                elif corr_ndvi < 0:
                    st.markdown(
                        f"Moderate negative correlation (r = {corr_ndvi:.2f}) between LST and NDVI: \n"
                        " - There is some indication that adding vegetation lowers land‐surface temperature.  \n"
                        " - Increasing green cover (e.g. street trees, green roofs) could help reduce heat in this grid."
                    )
                else:
                    st.markdown(
                        f"Vegetation cover does not show a clear inverse relationship with temperature. (r = {corr_ndvi:.2f})  \n"
                    )

            # 3) If only NDBI is valid, show NDBI-based message
            elif ndbi_valid:
                if corr_ndbi > 0.5:
                    st.markdown(
                        f"Strong positive correlation (r = {corr_ndbi:.2f}) between LST and NDBI:  \n"
                        " - This indicates that more built‐up surfaces correlate strongly with higher temperatures.  \n"
                        " - Consider reducing impervious surfaces (e.g. percent of concrete/asphalt) or using high‐albedo materials."
                    )
                elif corr_ndbi > 0:
                    st.markdown(
                        f"Moderate positive correlation (r = {corr_ndbi:.2f}) between LST and NDBI: \n"
                        " - Built‐up intensity shows some link to higher land‐surface temperatures.  \n"
                        " - Explore methods to limit urban fabric heat absorption—e.g. cool pavements or reflective roofing."
                    )
                else:
                    st.markdown(
                        f"Built‐up index does not show a clear direct relationship with temperature. (r = {corr_ndbi:.2f}) \n"
                    )

            # 4) If neither correlation is valid, notify user
            else:
                st.markdown(
                    f"Neither NDVI nor NDBI show a strong relationship with LST (r_NDVI = {corr_ndvi:.2f}, r_NDBI = {corr_ndbi:.2f}): \n"
                    " - No clear link between vegetation or built‐up index and temperature in this grid.  \n"
                    " - Other factors (e.g. building density, materials, albedo, anthropogenic heat) might be driving the heat. Further investigation is recommended."
                )
        else:
            st.markdown("**No summary data available for the selected grid.**")


    # Handle clicks on the Folium map to select a grid
    if map_data and map_data.get('last_clicked'):
        click_lat = map_data['last_clicked']['lat']
        click_lng = map_data['last_clicked']['lng']
        click_point = Point(click_lng, click_lat)

        for idx, row in geo_df.iterrows():
            if row['geometry'].buffer(0.01).contains(click_point):
                st.session_state.selected_grid = row['label']
                break

with col_analysis:
    st.markdown(f"### Grid {st.session_state.selected_grid} – Time Series")
    grid_data = df_year_filtered[df_year_filtered['label'] == st.session_state.selected_grid]
    grid_summary = summary[summary['label'] == st.session_state.selected_grid]

    if not grid_data.empty:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(grid_data['year'], grid_data['LST'], color='tab:red', marker='o', label='LST')
        ax2 = ax1.twinx()
        ax2.plot(grid_data['year'], grid_data['NDVI'], color='tab:green', marker='s', label='NDVI')
        ax2.plot(grid_data['year'], grid_data['NDBI'], color='tab:blue', marker='^', label='NDBI')
        ax1.set_xlabel("Year")
        ax1.set_ylabel("LST (°C)", color='tab:red')
        ax2.set_ylabel("NDVI / NDBI", color='tab:blue')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No data available for this grid in the selected years.")

    st.markdown("### Trend & Correlation Summary")
    if not grid_summary.empty:
        st.dataframe(grid_summary)

        ndvi_p = grid_summary['LST_NDVI_pval'].values[0]
        ndbi_p = grid_summary['LST_NDBI_pval'].values[0]
        st.markdown("**Statistical Significance (p-values):**")
        st.markdown(f"- LST vs NDVI p-value: **{ndvi_p:.4f}**" if pd.notnull(ndvi_p) else "- NDVI p-value: N/A")
        st.markdown(f"- LST vs NDBI p-value: **{ndbi_p:.4f}**" if pd.notnull(ndbi_p) else "- NDBI p-value: N/A")
    else:
        st.warning("No summary data found for this grid.")

    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    if not grid_data.empty:
        st.download_button(
            label="Download Grid Data as CSV",
            data=convert_df_to_csv(grid_data),
            file_name=f"grid_{st.session_state.selected_grid}_timeseries.csv",
            mime='text/csv'
        )

# Summary Map
st.subheader("Summary Map: Average LST (2000–2024)")
summary_map = folium.Map(location=[41.0082, 28.9784], zoom_start=10)
min_lst = geo_df['avg_LST'].min()
max_lst = geo_df['avg_LST'].max()
colormap = cm.linear.YlOrRd_09.scale(min_lst, max_lst)
colormap.caption = 'Average LST (°C)'

for _, row in geo_df.iterrows():
    folium.GeoJson(
        row['geometry'],
        style_function=lambda feature, avg=row['avg_LST']: {
            'fillColor': colormap(avg),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7
        },
        tooltip=f"Grid: {row['label']}<br>Avg LST: {row['avg_LST']:.2f}°C"
    ).add_to(summary_map)

colormap.add_to(summary_map)
st_folium(summary_map, width=900, height=500)

# Sidebar: Grid Selection
st.sidebar.header("Grid Selection")
grid_ids = df_year_filtered['label'].unique()
if st.session_state.selected_grid is None and len(grid_ids) > 0:
    st.session_state.selected_grid = grid_ids[0]

dropdown_index = 0
if st.session_state.selected_grid in grid_ids:
    dropdown_index = list(grid_ids).index(st.session_state.selected_grid)

selected_grid_from_dropdown = st.sidebar.selectbox("Select a Grid ID", grid_ids, index=dropdown_index)
if selected_grid_from_dropdown != st.session_state.selected_grid:
    st.session_state.selected_grid = selected_grid_from_dropdown

# Extra Visualizations
st.sidebar.header("Additional Visualizations")
if st.sidebar.checkbox("Show UHI Severity Trends"):
    st.subheader("UHI Severity Trends Over Time")
    col1, col2 = st.columns(2)
    with col1:
        st.image("uhi_severity_trend_barchart.png", caption="Grids by UHI Class")
    with col2:
        st.image("uhi_severity_trend_percentage_barchart.png", caption="Grids by UHI Class (%)")

if st.sidebar.checkbox("Show Citywide LST Correlations"):
    st.subheader("Citywide Correlation Insight")

    # Drop rows with missing values
    clean_df = df.dropna(subset=['LST', 'NDVI', 'NDBI'])

    # Calculate correlation coefficients
    corr_ndvi, pval_ndvi = pearsonr(clean_df['LST'], clean_df['NDVI'])
    corr_ndbi, pval_ndbi = pearsonr(clean_df['LST'], clean_df['NDBI'])

    # Display correlation values
    st.markdown(f"**LST vs NDVI:** r = {corr_ndvi:.3f}, p = {pval_ndvi:.4f}")
    st.markdown(f"**LST vs NDBI:** r = {corr_ndbi:.3f}, p = {pval_ndbi:.4f}")


if st.sidebar.checkbox("Show LST–NDVI Correlation Histogram"):
    st.subheader("Per-Grid LST–NDVI Correlation Distribution")

    summary_df = gpd.read_file("grid_analysis_summary.gpkg")

    ndvi_corrs = summary_df['LST_NDVI_corr'].dropna()

    # Plot histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ndvi_corrs, bins=30, color='green', edgecolor='black')
    ax.set_title("Histogram of LST–NDVI Correlation (Per Grid)")
    ax.set_xlabel("Correlation Coefficient")
    ax.set_ylabel("Number of Grids")
    ax.grid(True)
    st.pyplot(fig)

if st.sidebar.checkbox("Show LST–NDBI Correlation Histogram"):
    st.subheader("Per-Grid LST–NDBI Correlation Distribution")

    # Drop NA values to clean data
    ndvi_corrs = summary_df['LST_NDBI_corr'].dropna()

    # Plot histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ndvi_corrs, bins=30, color='red', edgecolor='black')
    ax.set_title("Histogram of LST–NDBI Correlation (Per Grid)")
    ax.set_xlabel("Correlation Coefficient")
    ax.set_ylabel("Number of Grids")
    ax.grid(True)
    st.pyplot(fig)


show_regression_summary = st.sidebar.checkbox("Show Citywide Regression Summary")
if show_regression_summary:
    st.subheader("Citywide Regression Summary (LST ~ NDVI + NDBI + NDVI×NDBI)")

    try:
        reg_df = pd.read_csv("regression_summary.csv")
        r2 = reg_df['LST_R2'].values[0]
        coef_ndvi = reg_df['coef_NDVI'].values[0]
        coef_ndbi = reg_df['coef_NDBI'].values[0]
        coef_interact = reg_df['coef_NDVIxNDBI'].values[0]
        p_ndvi = reg_df['p_NDVI'].values[0]
        p_ndbi = reg_df['p_NDBI'].values[0]
        p_interact = reg_df['p_NDVIxNDBI'].values[0]

        st.markdown(f"**Model R²:** {r2:.3f}")
        st.markdown(f"- **NDVI coefficient**: {coef_ndvi:.3f} (p = {p_ndvi})")
        st.markdown(f"- **NDBI coefficient**: {coef_ndbi:.3f} (p = {p_ndbi})")
        st.markdown(f"- **NDVI × NDBI interaction**: {coef_interact:.3f} (p = {p_interact})")

    except FileNotFoundError:
        st.warning("Regression summary not found. Please make sure 'regression_summary.csv' is generated.")
