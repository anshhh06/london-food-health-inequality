import pandas as pd

print('Loading data...')
llm = pd.read_csv('../data/processed/pois_llm_classified.csv')
spatial = pd.read_csv('../data/processed/pois_with_lsoa.csv')
lsoas = pd.read_csv('../data/raw/london_lsoas.csv')

# Merge LLM classifications with spatial join results
merged = spatial.merge(llm[['osm_id', 'llm_category']], on='osm_id', how='left')
print(f'Total POIs: {len(merged)}')
print(f'Classified: {merged["llm_category"].notna().sum()}')
print(f'Errors/unclassified: {(merged["llm_category"] == "error").sum()}')

# Keep only matched to LSOAs and successfully classified
clean = merged[
    (merged['lsoa21cd'].notna()) & 
    (merged['llm_category'] != 'error') &
    (merged['llm_category'].notna())
].copy()

# Fix malformed LLM responses
clean['llm_category'] = clean['llm_category'].str.strip()
clean['llm_category'] = clean['llm_category'].apply(
    lambda x: 'chain_supermarket' if str(x).startswith('chain_supermarket')
    else ('fast_food' if str(x).startswith('fast_food') else x)
)

print(f'Clean POIs for analysis: {len(clean)}')
print('Category counts:')
print(clean['llm_category'].value_counts())

# Count by LLM category per LSOA
counts = clean.groupby(['lsoa21cd', 'lsoa21nm', 'llm_category']).size().unstack(fill_value=0).reset_index()
counts.columns.name = None

# Merge with all LSOAs
final = lsoas[['lsoa21cd', 'lsoa21nm']].merge(counts, on=['lsoa21cd', 'lsoa21nm'], how='left')
cat_cols = [c for c in counts.columns if c not in ['lsoa21cd', 'lsoa21nm']]
final[cat_cols] = final[cat_cols].fillna(0).astype(int)

print(f'\nFinal dataset: {len(final)} LSOAs')
print(final[cat_cols].describe())

final.to_csv('../data/processed/lsoa_llm_food_environment.csv', index=False)
print('Saved to data/processed/lsoa_llm_food_environment.csv')
