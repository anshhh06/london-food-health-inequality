import pandas as pd

print('Loading data...')
counts = pd.read_csv('../data/processed/lsoa_food_counts.csv')
lsoas = pd.read_csv('../data/raw/london_lsoas.csv')

print(f'LSOAs total: {len(lsoas)}')
print(f'LSOAs with outlets: {len(counts)}')

# Merge — fill missing with 0
merged = lsoas[['lsoa21cd', 'lsoa21nm']].merge(counts, on=['lsoa21cd', 'lsoa21nm'], how='left')
merged[['fast_food', 'independent_fresh_food', 'market', 'supermarket']] = \
    merged[['fast_food', 'independent_fresh_food', 'market', 'supermarket']].fillna(0).astype(int)

print(f'Final dataset rows: {len(merged)}')
print(merged.describe())

merged.to_csv('../data/processed/lsoa_food_environment.csv', index=False)
print('Saved to data/processed/lsoa_food_environment.csv')
