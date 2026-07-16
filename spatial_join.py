import os
import math
import pandas as pd
import fiona
from shapely.geometry import Point, shape

def latlon_to_bng(lat, lon):
    """Convert WGS84 lat/lon to British National Grid easting/northing."""
    a = 6377563.396
    b = 6356256.909
    F0 = 0.9996012717
    lat0 = math.radians(49)
    lon0 = math.radians(-2)
    N0 = -100000
    E0 = 400000
    e2 = 1 - (b*b)/(a*a)
    n = (a-b)/(a+b)

    lat = math.radians(lat)
    lon = math.radians(lon)

    sinLat = math.sin(lat)
    cosLat = math.cos(lat)
    tanLat = math.tan(lat)

    nu = a * F0 / math.sqrt(1 - e2 * sinLat**2)
    rho = a * F0 * (1 - e2) / (1 - e2 * sinLat**2)**1.5
    eta2 = nu/rho - 1

    M = b * F0 * (
        (1 + n + 5/4*n**2 + 5/4*n**3) * (lat - lat0)
        - (3*n + 3*n**2 + 21/8*n**3) * math.sin(lat-lat0) * math.cos(lat+lat0)
        + (15/8*n**2 + 15/8*n**3) * math.sin(2*(lat-lat0)) * math.cos(2*(lat+lat0))
        - (35/24*n**3) * math.sin(3*(lat-lat0)) * math.cos(3*(lat+lat0))
    )

    I = M + N0
    II = nu/2 * sinLat * cosLat
    III = nu/24 * sinLat * cosLat**3 * (5 - tanLat**2 + 9*eta2)
    IIIA = nu/720 * sinLat * cosLat**5 * (61 - 58*tanLat**2 + tanLat**4)
    IV = nu * cosLat
    V = nu/6 * cosLat**3 * (nu/rho - tanLat**2)
    VI = nu/120 * cosLat**5 * (5 - 18*tanLat**2 + tanLat**4 + 14*eta2 - 58*tanLat**2*eta2)

    dlon = lon - lon0
    N = I + II*dlon**2 + III*dlon**4 + IIIA*dlon**6
    E = E0 + IV*dlon + V*dlon**3 + VI*dlon**5

    return E, N

print('Loading POIs...')
pois = pd.read_csv('../data/raw/london_food_pois.csv')
print(f'POIs loaded: {len(pois)}')

print('Converting coordinates...')
eastings, northings = [], []
for _, row in pois.iterrows():
    e, n = latlon_to_bng(row['lat'], row['lon'])
    eastings.append(e)
    northings.append(n)
pois['easting'] = eastings
pois['northing'] = northings
print(f'Sample: {eastings[0]:.0f}, {northings[0]:.0f}')

print('Loading LSOAs...')
lsoa_shapes = []
with fiona.open('../BoundaryData/england_lsoa_2021.shp') as src:
    for feature in src:
        props = feature['properties']
        if 'E09' in str(props.get('label', '')):
            lsoa_shapes.append({
                'lsoa21cd': props['lsoa21cd'],
                'lsoa21nm': props['lsoa21nm'],
                'geometry': shape(feature['geometry'])
            })
print(f'London LSOAs: {len(lsoa_shapes)}')

print('Running spatial join...')
results = []
for i, row in pois.iterrows():
    point = Point(row['easting'], row['northing'])
    matched = False
    for lsoa in lsoa_shapes:
        if lsoa['geometry'].contains(point):
            results.append({**row, 'lsoa21cd': lsoa['lsoa21cd'], 'lsoa21nm': lsoa['lsoa21nm']})
            matched = True
            break
    if not matched:
        results.append({**row, 'lsoa21cd': None, 'lsoa21nm': None})
    if i % 1000 == 0:
        print(f'  Processed {i}/{len(pois)}...')

df = pd.DataFrame(results)
print(f'POIs matched: {df["lsoa21cd"].notna().sum()}')
print(f'POIs unmatched: {df["lsoa21cd"].isna().sum()}')

os.makedirs('../data/processed', exist_ok=True)
df.to_csv('../data/processed/pois_with_lsoa.csv', index=False)
print('Done!')
