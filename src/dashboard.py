import pandas as pd
import folium
import fiona
from shapely.geometry import shape, mapping
import os

os.makedirs('../dashboard', exist_ok=True)

print('Loading data...')
df = pd.read_csv('../data/processed/final_analysis_dataset.csv')
df = df[df['year'] == 2019].copy()
df = df.rename(columns={'geography code': 'lsoa21cd'})
df = df.drop_duplicates(subset=['lsoa21cd'])
df = df.reset_index(drop=True)

# Build lookup dictionary
data_lookup = {}
for i in range(len(df)):
    code = str(df.loc[i, 'lsoa21cd'])
    data_lookup[code] = {
        'diabetes': round(float(df.loc[i, 'o_diabetes_quantity_per_capita']), 2) if pd.notna(df.loc[i, 'o_diabetes_quantity_per_capita']) else 0,
        'hypertension': round(float(df.loc[i, 'o_hypertension_quantity_per_capita']), 2) if pd.notna(df.loc[i, 'o_hypertension_quantity_per_capita']) else 0,
        'indep_fresh': round(float(df.loc[i, 'dw_independent_fresh_food']), 3) if pd.notna(df.loc[i, 'dw_independent_fresh_food']) else 0,
        'fast_food': round(float(df.loc[i, 'dw_fast_food']), 3) if pd.notna(df.loc[i, 'dw_fast_food']) else 0,
    }

print(f'Lookup built: {len(data_lookup)} LSOAs')

print('Loading LSOA boundaries...')
lsoa_features = []
with fiona.open('../BoundaryData/england_lsoa_2021.shp') as src:
    for feature in src:
        props = feature['properties']
        if 'E09' in str(props.get('label', '')):
            code = props['lsoa21cd']
            data = data_lookup.get(code, {'diabetes': 0, 'hypertension': 0, 'indep_fresh': 0, 'fast_food': 0})
            lsoa_features.append({
                'type': 'Feature',
                'properties': {
                    'lsoa21cd': code,
                    'lsoa21nm': props['lsoa21nm'],
                    'diabetes': data['diabetes'],
                    'hypertension': data['hypertension'],
                    'indep_fresh': data['indep_fresh'],
                    'fast_food': data['fast_food']
                },
                'geometry': mapping(shape(feature['geometry']))
            })

print(f'Features built: {len(lsoa_features)}')
geojson = {'type': 'FeatureCollection', 'features': lsoa_features}

print('Building map...')
m = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles='CartoDB positron')

# Diabetes layer
folium.Choropleth(
    geo_data=geojson,
    data=df,
    columns=['lsoa21cd', 'o_diabetes_quantity_per_capita'],
    key_on='feature.properties.lsoa21cd',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.1,
    legend_name='Diabetes Rate per Capita',
    name='Diabetes Rate'
).add_to(m)

# Tooltips
folium.GeoJson(
    geojson,
    style_function=lambda x: {'fillOpacity': 0, 'weight': 0},
    tooltip=folium.GeoJsonTooltip(
        fields=['lsoa21nm', 'diabetes', 'hypertension', 'indep_fresh', 'fast_food'],
        aliases=['Area:', 'Diabetes Rate:', 'Hypertension Rate:',
                 'Indep Fresh Food Score:', 'Fast Food Score:'],
        localize=True
    ),
    popup=folium.GeoJsonPopup(
        fields=['lsoa21nm', 'diabetes', 'hypertension', 'indep_fresh', 'fast_food'],
        aliases=['Area:', 'Diabetes Rate:', 'Hypertension Rate:',
                 'Indep Fresh Food Score:', 'Fast Food Score:']
    )
).add_to(m)

# Independent fresh food layer
folium.Choropleth(
    geo_data=geojson,
    data=df,
    columns=['lsoa21cd', 'dw_independent_fresh_food'],
    key_on='feature.properties.lsoa21cd',
    fill_color='Greens',
    fill_opacity=0.7,
    line_opacity=0.1,
    legend_name='Independent Fresh Food Access Score',
    name='Independent Fresh Food Access',
    show=False
).add_to(m)

# Fast food layer
folium.Choropleth(
    geo_data=geojson,
    data=df,
    columns=['lsoa21cd', 'dw_fast_food'],
    key_on='feature.properties.lsoa21cd',
    fill_color='Reds',
    fill_opacity=0.7,
    line_opacity=0.1,
    legend_name='Fast Food Access Score',
    name='Fast Food Access',
    show=False
).add_to(m)

# Hypertension layer
folium.Choropleth(
    geo_data=geojson,
    data=df,
    columns=['lsoa21cd', 'o_hypertension_quantity_per_capita'],
    key_on='feature.properties.lsoa21cd',
    fill_color='PuRd',
    fill_opacity=0.7,
    line_opacity=0.1,
    legend_name='Hypertension Rate per Capita',
    name='Hypertension Rate',
    show=False
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

output_path = '../dashboard/london_food_health_dashboard.html'
m.save(output_path)
print(f'Saved to {output_path}')
print('Open in Chrome!')
