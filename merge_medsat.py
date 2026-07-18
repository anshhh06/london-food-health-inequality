import pandas as pd

print('Loading MedSat 2019...')
med2019 = pd.read_csv('../data/raw/2019_spatial_raw_master.csv')
print(f'Total England LSOAs: {len(med2019)}')

print('Loading MedSat 2020...')
med2020 = pd.read_csv('../data/raw/2020_spatial_raw_master.csv')

# Load London LSOAs
london_lsoas = pd.read_csv('../data/raw/london_lsoas.csv')
london_codes = london_lsoas['lsoa21cd'].tolist()
print(f'London LSOAs: {len(london_codes)}')

# Filter MedSat to London
health_cols = ['geography code', 'o_diabetes_quantity_per_capita',
               'o_hypertension_quantity_per_capita', 'o_depression_quantity_per_capita',
               'o_anxiety_quantity_per_capita', 'o_asthma_quantity_per_capita']

# Keep key sociodemographic controls too
demo_cols = [c for c in med2019.columns if any(x in c.lower() for x in 
             ['percent', 'imd', 'density', 'unemploy', 'income'])]
print(f'Demographic control columns: {len(demo_cols)}')

keep_cols = health_cols + demo_cols
med2019_london = med2019[med2019['geography code'].isin(london_codes)][keep_cols]
med2020_london = med2020[med2020['geography code'].isin(london_codes)][keep_cols]

print(f'London rows in 2019: {len(med2019_london)}')
print(f'London rows in 2020: {len(med2020_london)}')

# Add year column and combine
med2019_london['year'] = 2019
med2020_london['year'] = 2020
medsat = pd.concat([med2019_london, med2020_london], ignore_index=True)
print(f'Combined MedSat rows: {len(medsat)}')

# Load food environment data
food = pd.read_csv('../data/processed/lsoa_llm_food_environment.csv')
print(f'Food environment LSOAs: {len(food)}')

# Merge
merged = medsat.merge(food, left_on='geography code', right_on='lsoa21cd', how='left')
print(f'Final merged dataset: {len(merged)} rows')
print(merged[['geography code', 'o_diabetes_quantity_per_capita', 
              'independent_fresh_food', 'fast_food']].head(5).to_string())

merged.to_csv('../data/processed/final_analysis_dataset.csv', index=False)
print('Saved to data/processed/final_analysis_dataset.csv')
