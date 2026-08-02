import pandas as pd
import folium
from folium.plugins import HeatMap
import os
import math

os.makedirs('../dashboard', exist_ok=True)

def bng_to_latlon(easting, northing):
    """Convert British National Grid to WGS84 lat/lon."""
    a = 6377563.396
    b = 6356256.909
    F0 = 0.9996012717
    lat0 = math.radians(49)
    lon0 = math.radians(-2)
    N0 = -100000
    E0 = 400000
    e2 = 1 - (b*b)/(a*a)
    n = (a-b)/(a+b)
    E = easting
    N = northing
    lat = lat0
    M = 0
    while abs(N - N0 - M) >= 0.00001:
        lat = (N - N0 - M)/(a*F0) + lat
        M = b*F0*(
            (1+n+5/4*n**2+5/4*n**3)*(lat-lat0)
            -(3*n+3*n**2+21/8*n**3)*math.sin(lat-lat0)*math.cos(lat+lat0)
            +(15/8*n**2+15/8*n**3)*math.sin(2*(lat-lat0))*math.cos(2*(lat+lat0))
            -(35/24*n**3)*math.sin(3*(lat-lat0))*math.cos(3*(lat+lat0))
        )
    nu = a*F0/math.sqrt(1-e2*math.sin(lat)**2)
    rho = a*F0*(1-e2)/(1-e2*math.sin(lat)**2)**1.5
    eta2 = nu/rho-1
    tanLat = math.tan(lat)
    secLat = 1/math.cos(lat)
    VII = tanLat/(2*rho*nu)
    VIII = tanLat/(24*rho*nu**3)*(5+3*tanLat**2+eta2-9*tanLat**2*eta2)
    IX = tanLat/(720*rho*nu**5)*(61+90*tanLat**2+45*tanLat**4)
    X = secLat/nu
    XI = secLat/(6*nu**3)*(nu/rho+2*tanLat**2)
    XII = secLat/(120*nu**5)*(5+28*tanLat**2+24*tanLat**4)
    XIIA = secLat/(5040*nu**7)*(61+662*tanLat**2+1320*tanLat**4+720*tanLat**6)
    dE = E-E0
    lat = lat - VII*dE**2 + VIII*dE**4 - IX*dE**6
    lon = lon0 + X*dE - XI*dE**3 + XII*dE**5 - XIIA*dE**7
    return math.degrees(lat), math.degrees(lon)

print('Loading data...')
df = pd.read_csv('../data/processed/final_analysis_dataset.csv')
df = df[df['year'] == 2019].copy()
df = df.loc[:, ~df.columns.duplicated()]

centroids = pd.read_csv('../data/raw/2019_spatial_raw_master.csv',
                        usecols=['geography code', 'centroid_x', 'centroid_y'])
df = df.merge(centroids, on='geography code', how='left')
df = df.dropna(subset=['centroid_x', 'centroid_y'])

print('Converting coordinates...')
df['lat'] = df.apply(lambda r: bng_to_latlon(r['centroid_x'], r['centroid_y'])[0], axis=1)
df['lon'] = df.apply(lambda r: bng_to_latlon(r['centroid_x'], r['centroid_y'])[1], axis=1)
print(f'Coordinates converted for {len(df)} LSOAs')

variables = {
    'dw_independent_fresh_food': 'Independent Fresh Food Access',
    'dw_fast_food': 'Fast Food Access',
    'o_diabetes_quantity_per_capita': 'Diabetes Prescription Rate',
    'o_hypertension_quantity_per_capita': 'Hypertension Prescription Rate'
}

for col, title in variables.items():
    print(f'Creating map for {title}...')
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10,
                   tiles='CartoDB positron')
    
    heat_data = [[row['lat'], row['lon'], row[col]] 
                 for _, row in df.iterrows() 
                 if pd.notna(row[col]) and row[col] > 0]
    
    HeatMap(heat_data, min_opacity=0.3, radius=15, blur=10).add_to(m)
    
    filename = f"../dashboard/map_{col}.html"
    m.save(filename)
    print(f'Saved: {filename}')

print('Done! Open each HTML file in Chrome and take screenshots.')