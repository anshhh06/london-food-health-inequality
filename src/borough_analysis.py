import pandas as pd
import numpy as np
import os

os.makedirs('../results', exist_ok=True)

print('Loading data...')
df = pd.read_csv('../data/processed/final_analysis_dataset.csv')
df = df[df['year'] == 2019].copy()

# Extract borough name from LSOA name
df['borough'] = df['lsoa21nm'].str.extract(r'^(.*?)\s+\d')

# Borough-level aggregation
borough = df.groupby('borough').agg(
    lsoa_count=('lsoa21cd', 'count'),
    mean_diabetes=('o_diabetes_quantity_per_capita', 'mean'),
    mean_hypertension=('o_hypertension_quantity_per_capita', 'mean'),
    mean_indep_fresh=('dw_independent_fresh_food', 'mean'),
    mean_fast_food=('dw_fast_food', 'mean'),
    mean_supermarket=('dw_chain_supermarket', 'mean'),
    mean_convenience=('dw_convenience_chain', 'mean')
).reset_index()

# Quartile classification by independent fresh food access
borough['fresh_food_quartile'] = pd.qcut(borough['mean_indep_fresh'], 4,
                                          labels=['Q1_lowest', 'Q2', 'Q3', 'Q4_highest'])

print(f'Total boroughs: {len(borough)}')

print(f'\nTop 5 boroughs by independent fresh food access:')
top5 = borough.nlargest(5, 'mean_indep_fresh')
print(top5[['borough', 'mean_indep_fresh', 'mean_fast_food', 
            'mean_diabetes', 'mean_hypertension']].to_string())

print(f'\nBottom 5 boroughs by independent fresh food access:')
bottom5 = borough.nsmallest(5, 'mean_indep_fresh')
print(bottom5[['borough', 'mean_indep_fresh', 'mean_fast_food',
               'mean_diabetes', 'mean_hypertension']].to_string())

print(f'\nMean diabetes by fresh food quartile:')
quartile_summary = borough.groupby('fresh_food_quartile')[
    ['mean_diabetes', 'mean_hypertension', 'mean_indep_fresh', 'mean_fast_food']
].mean()
print(quartile_summary.to_string())

borough.to_csv('../data/processed/borough_analysis.csv', index=False)

with open('../results/13_borough_analysis.txt', 'w') as f:
    f.write('BOROUGH-LEVEL ANALYSIS (RQ3)\n')
    f.write('='*50 + '\n\n')
    f.write(f'Total boroughs analysed: {len(borough)}\n\n')
    f.write('All boroughs sorted by diabetes rate:\n')
    f.write(borough.sort_values('mean_diabetes', ascending=False).to_string())
    f.write('\n\nTop 5 boroughs by independent fresh food access:\n')
    f.write(top5[['borough', 'mean_indep_fresh', 'mean_fast_food',
                  'mean_diabetes', 'mean_hypertension']].to_string())
    f.write('\n\nBottom 5 boroughs by independent fresh food access:\n')
    f.write(bottom5[['borough', 'mean_indep_fresh', 'mean_fast_food',
                     'mean_diabetes', 'mean_hypertension']].to_string())
    f.write('\n\nMean health outcomes by fresh food access quartile:\n')
    f.write(quartile_summary.to_string())

print('\nSaved to results/13_borough_analysis.txt')
