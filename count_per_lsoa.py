import pandas as pd

print('Loading joined data...')
df = pd.read_csv('../data/processed/pois_with_lsoa.csv')

# Keep only matched POIs
df = df[df['lsoa21cd'].notna()]
print(f'Matched POIs: {len(df)}')

# Count by category per LSOA
counts = df.groupby(['lsoa21cd', 'lsoa21nm', 'category']).size().unstack(fill_value=0).reset_index()
counts.columns.name = None

print('Categories found:', [c for c in counts.columns if c not in ['lsoa21cd', 'lsoa21nm']])
print(f'LSOAs with at least one outlet: {len(counts)}')
print(counts.head(3))

counts.to_csv('../data/processed/lsoa_food_counts.csv', index=False)
print('Saved to data/processed/lsoa_food_counts.csv')
