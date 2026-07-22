import pandas as pd
import numpy as np
from esda.moran import Moran
from libpysal.weights import KNN
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('../results', exist_ok=True)

print('Loading data...')
df = pd.read_csv('../data/processed/final_analysis_dataset.csv')
centroids = pd.read_csv('../data/raw/2019_spatial_raw_master.csv',
                        usecols=['geography code', 'centroid_x', 'centroid_y'])
df = df.merge(centroids, on='geography code', how='left')
df = df[df['year'] == 2019].copy()
df = df.dropna(subset=['o_diabetes_quantity_per_capita', 
                        'o_hypertension_quantity_per_capita',
                        'centroid_x', 'centroid_y'])
print(f'LSOAs: {len(df)}')

# Build spatial weights
coords = list(zip(df['centroid_x'], df['centroid_y']))
print('Building spatial weights...')
w = KNN.from_array(coords, k=5)
w.transform = 'r'

results = {}

# Moran's I for diabetes
print('Running Moran\'s I for diabetes...')
moran_d = Moran(df['o_diabetes_quantity_per_capita'].values, w)
results['diabetes'] = {
    'I': moran_d.I,
    'p_value': moran_d.p_sim,
    'z_score': moran_d.z_sim
}
print(f'Diabetes — I: {moran_d.I:.4f}, p: {moran_d.p_sim:.4f}')

# Moran's I for hypertension
print('Running Moran\'s I for hypertension...')
moran_h = Moran(df['o_hypertension_quantity_per_capita'].values, w)
results['hypertension'] = {
    'I': moran_h.I,
    'p_value': moran_h.p_sim,
    'z_score': moran_h.z_sim
}
print(f'Hypertension — I: {moran_h.I:.4f}, p: {moran_h.p_sim:.4f}')

# Moran's I for independent fresh food
print('Running Moran\'s I for independent fresh food...')
moran_f = Moran(df['dw_independent_fresh_food'].values, w)
results['indep_fresh_food'] = {
    'I': moran_f.I,
    'p_value': moran_f.p_sim,
    'z_score': moran_f.z_sim
}
print(f'Indep fresh food — I: {moran_f.I:.4f}, p: {moran_f.p_sim:.4f}')

# Save results
with open('../results/11_morans_i_results.txt', 'w') as f:
    f.write("MORAN'S I — SPATIAL AUTOCORRELATION TESTS\n")
    f.write('='*50 + '\n\n')
    f.write('KNN spatial weights (k=5), 2019 data\n\n')
    for var, res in results.items():
        f.write(f'{var}:\n')
        f.write(f"  Moran's I: {res['I']:.4f}\n")
        f.write(f"  p-value:   {res['p_value']:.4f}\n")
        f.write(f"  z-score:   {res['z_score']:.4f}\n\n")

print('\nSaved to results/11_morans_i_results.txt')
