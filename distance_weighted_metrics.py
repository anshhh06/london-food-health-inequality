import pandas as pd
import math
import fiona
from shapely.geometry import Point, shape

RADII = {
    'chain_supermarket': 800,
    'convenience_chain': 400,
    'independent_fresh_food': 400,
    'market': 600,
    'fast_food': 300,
    'other': 300
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def latlon_to_bng(lat, lon):
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
pois = pd.read_csv('../data/processed/pois_llm_classified.csv')
pois = pois[pois['llm_category'].notna()]
pois = pois[~pois['llm_category'].str.startswith('chain_supermarket\n')]
print(f'POIs loaded: {len(pois)}')

print('Converting POI coordinates to BNG...')
pois['easting'], pois['northing'] = zip(*pois.apply(
    lambda r: latlon_to_bng(r['lat'], r['lon']), axis=1))

print('Loading LSOA centroids...')
medsat = pd.read_csv('../data/raw/2019_spatial_raw_master.csv',
                     usecols=['geography code', 'centroid_x', 'centroid_y'])
london = pd.read_csv('../data/raw/london_lsoas.csv')
medsat = medsat[medsat['geography code'].isin(london['lsoa21cd'])]
print(f'London LSOAs: {len(medsat)}')

print('Calculating distance-weighted scores...')
results = []
for i, lsoa in medsat.iterrows():
    lsoa_code = lsoa['geography code']
    cx = lsoa['centroid_x']
    cy = lsoa['centroid_y']
    scores = {cat: 0.0 for cat in RADII}
    for _, poi in pois.iterrows():
        cat = poi['llm_category']
        if cat not in RADII:
            continue
        dist = math.sqrt((cx - poi['easting'])**2 + (cy - poi['northing'])**2)
        radius = RADII[cat]
        if dist < radius:
            weight = 1 - (dist / radius)
            scores[cat] += weight
    row = {'lsoa21cd': lsoa_code}
    row.update({f'dw_{k}': round(v, 4) for k, v in scores.items()})
    results.append(row)
    if len(results) % 500 == 0:
        print(f'  Processed {len(results)}/{len(medsat)} LSOAs...')

df = pd.DataFrame(results)
print(f'Done! Shape: {df.shape}')
print(df.describe())
df.to_csv('../data/processed/lsoa_distance_weighted.csv', index=False)
print('Saved to data/processed/lsoa_distance_weighted.csv')