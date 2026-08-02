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

# Use 2019 only
df = df[df['year'] == 2019].copy()
df = df.loc[:, ~df.columns.duplicated()]
print(f'LSOAs: {len(df)}')

# Remove outliers
threshold = df['o_hypertension_quantity_per_capita'].quantile(0.99)
df = df[df['o_hypertension_quantity_per_capita'] < threshold]

# Drop NAs
key_cols = ['o_hypertension_quantity_per_capita', 'dw_independent_fresh_food',
            'dw_fast_food', 'dw_chain_supermarket', 'dw_convenience_chain',
            'centroid_x', 'centroid_y',
            'c_pop_density', 'c_percent unemployed', 'c_net annual income',
            'c_percent asian', 'c_percent black', 'c_percent Aged 65 to 69 years']
df = df.dropna(subset=key_cols)
print(f'LSOAs after dropping NAs: {len(df)}')

# Define variables
y = df[['o_hypertension_quantity_per_capita']].values

X = df[['dw_independent_fresh_food', 'dw_fast_food',
        'dw_chain_supermarket', 'dw_convenience_chain',
        'c_pop_density', 'c_percent unemployed',
        'c_net annual income',
        'c_percent asian', 'c_percent black',
        'c_percent Aged 65 to 69 years']].values

x_names = ['dw_indep_fresh', 'dw_fast_food', 'dw_supermarket', 'dw_convenience',
           'pop_density', 'pct_unemployed', 'net_income',
           'pct_asian', 'pct_black', 'pct_aged_65_69']

print(f'X shape: {X.shape}, y shape: {y.shape}')

# Build spatial weights
coords = list(zip(df['centroid_x'], df['centroid_y']))
print('Building spatial weights...')
w = KNN.from_array(coords, k=5)
w.transform = 'r'

# OLS
print('\n--- OLS ---')
ols = OLS(y, X, w=w, spat_diag=True,
          name_y='hypertension', name_x=x_names, name_ds='London 2019')
print(ols.summary)

# Spatial Lag
print('\n--- Spatial Lag ---')
lag = ML_Lag(y, X, w=w, name_y='hypertension', name_x=x_names, name_ds='London 2019')
print(lag.summary)

# Spatial Error
print('\n--- Spatial Error ---')
err = ML_Error(y, X, w=w, name_y='hypertension', name_x=x_names, name_ds='London 2019')
print(err.summary)

# Save full output
with open('../results/15_regression_hypertension_with_controls.txt', 'w') as f:
    f.write('REGRESSION WITH CONFOUNDING CONTROLS — HYPERTENSION (2019)\n')
    f.write('='*60 + '\n\n')
    f.write('Controls: population density, unemployment, net income, ethnicity, age 65-69\n\n')
    f.write('--- OLS ---\n')
    f.write(ols.summary)
    f.write('\n\n--- SPATIAL LAG ---\n')
    f.write(lag.summary)
    f.write('\n\n--- SPATIAL ERROR ---\n')
    f.write(err.summary)
print('Saved to results/15_regression_hypertension_with_controls.txt')

# Model comparison
results_df = pd.DataFrame({
    'Model': ['OLS', 'Spatial Lag', 'Spatial Error'],
    'R2': [ols.r2, lag.pr2, err.pr2],
    'AIC': [ols.aic, lag.aic, err.aic],
    'coef_dw_indep_fresh': [ols.betas[1][0], lag.betas[1][0], err.betas[1][0]]
})
print('\n--- Model Comparison ---')
print(results_df.to_string())
results_df.to_csv('../data/processed/regression_results_hypertension_with_controls.csv', index=False)
print('Done!')