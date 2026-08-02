import pandas as pd
import folium
import os

os.makedirs('../dashboard', exist_ok=True)

print('Loading POIs...')
pois = pd.read_csv('../data/processed/pois_llm_classified.csv')
pois = pois[pois['llm_category'].notna()]
pois = pois[~pois['llm_category'].str.startswith('chain_supermarket\n')]

# Sample to keep map fast
pois_sample = pois.sample(n=3000, random_state=42)

colours = {
    'independent_fresh_food': 'green',
    'fast_food': 'red',
    'chain_supermarket': 'blue',
    'convenience_chain': 'orange',
    'market': 'purple',
    'other': 'gray'
}

m = folium.Map(location=[51.5074, -0.1278], zoom_start=10,
               tiles='CartoDB positron')

for _, row in pois_sample.iterrows():
    cat = row['llm_category']
    colour = colours.get(cat, 'gray')
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=3,
        color=colour,
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['name']} ({cat})"
    ).add_to(m)

m.save('../dashboard/food_outlets_map.html')
print('Saved to dashboard/food_outlets_map.html')
print('Open in Chrome and take a screenshot!')