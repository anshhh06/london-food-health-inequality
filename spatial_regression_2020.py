import pandas as pd
import numpy as np
from spreg import OLS, ML_Lag, ML_Error
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

# Use 2020 data
df = df[df['year'] == 2020].copy()
print(f'LSOAs for analysis: {len(df)}')

# Remove outliers
threshold = df['o_diabetes_quantity_per_capita'].quantile(0.99)
df = df[df['o_diabetes_quantity_per_capita'] < threshold]
print(f'LSOAs after removing outliers: {len(df)}')

key_cols = ['o_diabetes_quantity_per_capita', 'dw_independent_fresh_food',
            'dw_fast_food', 'dw_chain_supermarket', 'centroid_x', 'centroid_y']
df = df.dropna(subset=key_cols)
print(f'LSOAs after dropping NAs: {len(df)}')

imd_cols = [c for c in df.columns if 'imd' in c.lower()]

y = df[['o_diabetes_quantity_per_capita']].values
X = df[['dw_independent_fresh_food', 'dw_fast_food',
        'dw_chain_supermarket', 'dw_convenience_chain']].values
x_names = ['dw_indep_fresh', 'dw_fast_food', 'dw_supermarket', 'dw_convenience']

if imd_cols:
    imd = df[imd_cols[0]].fillna(df[imd_cols[0]].median()).values.reshape(-1, 1)
    X = np.hstack([X, imd])
    x_names.append('imd')

print(f'X shape: {X.shape}, y shape: {y.shape}')

coords = list(zip(df['centroid_x'], df['centroid_y']))
print('Building spatial weights...')
w = KNN.from_array(coords, k=5)
w.transform = 'r'

print('\n--- OLS ---')
ols = OLS(y, X, w=w, spat_diag=True,
          name_y='diabetes_2020', name_x=x_names, name_ds='London 2020')
print(ols.summary)

print('\n--- Spatial Lag ---')
lag = ML_Lag(y, X, w=w, name_y='diabetes_2020', name_x=x_names, name_ds='London 2020')
print(lag.summary)

print('\n--- Spatial Error ---')
err = ML_Error(y, X, w=w, name_y='diabetes_2020', name_x=x_names, name_ds='London 2020')
print(err.summary)

with open('../results/12_regression_full_diabetes_2020.txt', 'w') as f:
    f.write('FULL REGRESSION OUTPUT — DIABETES (2020)\n')
    f.write('='*60 + '\n\n')
    f.write('--- OLS ---\n')
    f.write(ols.summary)
    f.write('\n\n--- SPATIAL LAG ---\n')
    f.write(lag.summary)
    f.write('\n\n--- SPATIAL ERROR ---\n')
    f.write(err.summary)
print('Saved to results/12_regression_full_diabetes_2020.txt')

results_df = pd.DataFrame({
    'Model': ['OLS', 'Spatial Lag', 'Spatial Error'],
    'R2': [ols.r2, lag.pr2, err.pr2],
    'AIC': [ols.aic, lag.aic, err.aic],
    'coef_dw_indep_fresh': [ols.betas[1][0], lag.betas[1][0], err.betas[1][0]]
})
print('\n--- Model Comparison ---')
print(results_df.to_string())
results_df.to_csv('../data/processed/regression_results_2020.csv', index=False)