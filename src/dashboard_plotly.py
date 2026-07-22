import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
df = df.loc[:, ~df.columns.duplicated()]

borough = pd.read_csv('../data/processed/borough_analysis.csv')

print('Loading LSOA boundaries...')
lsoa_features = []
with fiona.open('../BoundaryData/england_lsoa_2021.shp') as src:
    for feature in src:
        props = feature['properties']
        if 'E09' in str(props.get('label', '')):
            lsoa_features.append({
                'type': 'Feature',
                'id': props['lsoa21cd'],
                'properties': {'lsoa21cd': props['lsoa21cd'], 'lsoa21nm': props['lsoa21nm']},
                'geometry': mapping(shape(feature['geometry']))
            })

geojson = {'type': 'FeatureCollection', 'features': lsoa_features}
print(f'Features built: {len(lsoa_features)}')

df['diabetes_clean'] = df['o_diabetes_quantity_per_capita'].clip(0, 80)
df['hypertension_clean'] = df['o_hypertension_quantity_per_capita'].clip(0, 150)
df['borough'] = df['lsoa21nm'].str.extract(r'^(.*?)\s+\d')

print('Building dashboard...')

fig_diabetes = px.choropleth_map(
    df,
    geojson=geojson,
    locations='lsoa21cd',
    color='diabetes_clean',
    color_continuous_scale='YlOrRd',
    map_style='carto-positron',
    zoom=9,
    center={'lat': 51.5074, 'lon': -0.1278},
    opacity=0.7,
    labels={'diabetes_clean': 'Diabetes Rate'},
    hover_name='lsoa21nm',
    title='Diabetes Prescription Rate per Capita — London LSOAs (2019)'
)
fig_diabetes.update_layout(
    height=600,
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    paper_bgcolor='#1a1a2e',
    font_color='white',
    title_font_size=16
)

fig_food = px.choropleth_map(
    df,
    geojson=geojson,
    locations='lsoa21cd',
    color='dw_independent_fresh_food',
    color_continuous_scale='Greens',
    map_style='carto-positron',
    zoom=9,
    center={'lat': 51.5074, 'lon': -0.1278},
    opacity=0.7,
    labels={'dw_independent_fresh_food': 'Fresh Food Score'},
    hover_name='lsoa21nm',
    title='Independent Fresh Food Access Score — London LSOAs (2019)'
)
fig_food.update_layout(
    height=600,
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    paper_bgcolor='#1a1a2e',
    font_color='white',
    title_font_size=16
)

fig_scatter = px.scatter(
    borough,
    x='mean_indep_fresh',
    y='mean_diabetes',
    size='lsoa_count',
    color='mean_hypertension',
    color_continuous_scale='RdYlGn_r',
    hover_name='borough',
    labels={
        'mean_indep_fresh': 'Mean Independent Fresh Food Access Score',
        'mean_diabetes': 'Mean Diabetes Rate per Capita',
        'mean_hypertension': 'Hypertension Rate',
        'lsoa_count': 'Number of LSOAs'
    },
    title='Borough-Level: Independent Fresh Food Access vs Diabetes Rate'
)
fig_scatter.update_layout(
    height=500,
    paper_bgcolor='#1a1a2e',
    plot_bgcolor='#16213e',
    font_color='white',
    title_font_size=16
)

borough_sorted = borough.sort_values('mean_diabetes', ascending=True)
fig_bar = px.bar(
    borough_sorted,
    x='mean_diabetes',
    y='borough',
    orientation='h',
    color='mean_indep_fresh',
    color_continuous_scale='Greens',
    labels={
        'mean_diabetes': 'Mean Diabetes Rate per Capita',
        'borough': 'Borough',
        'mean_indep_fresh': 'Fresh Food Score'
    },
    title='Diabetes Rate by London Borough (coloured by Independent Fresh Food Access)'
)
fig_bar.update_layout(
    height=800,
    paper_bgcolor='#1a1a2e',
    plot_bgcolor='#16213e',
    font_color='white',
    title_font_size=16
)

html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>London Food Environment & Health Inequalities</title>
    <meta charset="utf-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #1a1a2e; color: white; font-family: 'Segoe UI', sans-serif; }}
        .header {{
            background: linear-gradient(135deg, #16213e, #0f3460);
            padding: 30px 40px;
            border-bottom: 2px solid #e94560;
        }}
        .header h1 {{ font-size: 24px; margin-bottom: 8px; }}
        .header p {{ color: #aaa; font-size: 14px; }}
        .nav {{
            display: flex;
            background: #16213e;
            padding: 0 40px;
            border-bottom: 1px solid #333;
        }}
        .nav button {{
            background: none;
            border: none;
            color: #aaa;
            padding: 15px 20px;
            cursor: pointer;
            font-size: 14px;
            border-bottom: 2px solid transparent;
            transition: all 0.3s;
        }}
        .nav button:hover, .nav button.active {{
            color: #e94560;
            border-bottom-color: #e94560;
        }}
        .tab {{ display: none; padding: 20px 40px; }}
        .tab.active {{ display: block; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: #16213e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .stat-card .value {{ font-size: 28px; font-weight: bold; color: #e94560; }}
        .stat-card .label {{ font-size: 12px; color: #aaa; margin-top: 5px; }}
        h2 {{ margin-bottom: 15px; font-size: 18px; color: #e94560; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>London Food Environment & Health Inequalities</h1>
        <p>MSc Data Science Dissertation — King's College London | Ansh Aggarwal | Supervised by Dr Linus Dietz</p>
    </div>
    <div class="nav">
        <button class="active" onclick="showTab('overview', this)">Overview</button>
        <button onclick="showTab('diabetes', this)">Diabetes Map</button>
        <button onclick="showTab('food', this)">Food Access Map</button>
        <button onclick="showTab('borough', this)">Borough Analysis</button>
    </div>
    <div id="overview" class="tab active">
        <div class="stats">
            <div class="stat-card">
                <div class="value">4,994</div>
                <div class="label">London LSOAs Analysed</div>
            </div>
            <div class="stat-card">
                <div class="value">10,907</div>
                <div class="label">Food Retail POIs Extracted</div>
            </div>
            <div class="stat-card">
                <div class="value">2,186</div>
                <div class="label">Independent Fresh Food Retailers</div>
            </div>
            <div class="stat-card">
                <div class="value">33</div>
                <div class="label">London Boroughs</div>
            </div>
        </div>
        {fig_scatter.to_html(full_html=False, include_plotlyjs='cdn')}
        <br>
        {fig_bar.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    <div id="diabetes" class="tab">
        <h2>Diabetes Prescription Rate per Capita (2019)</h2>
        {fig_diabetes.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    <div id="food" class="tab">
        <h2>Independent Fresh Food Access Score (2019)</h2>
        {fig_food.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    <div id="borough" class="tab">
        <h2>Borough-Level Summary</h2>
        {fig_bar.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    <script>
        function showTab(name, btn) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
            document.getElementById(name).classList.add('active');
            btn.classList.add('active');
        }}
    </script>
</body>
</html>
"""

with open('../dashboard/index.html', 'w') as f:
    f.write(html)

print('Dashboard saved to ../dashboard/index.html')
print('Open index.html in Chrome!')
