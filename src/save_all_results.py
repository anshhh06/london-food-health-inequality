import pandas as pd
import numpy as np
import os

os.makedirs('../results', exist_ok=True)

print('Saving all results...')

# 1. POI extraction summary
pois = pd.read_csv('../data/raw/london_food_pois.csv')
with open('../results/01_poi_extraction_summary.txt', 'w') as f:
    f.write('OSM POI EXTRACTION SUMMARY\n')
    f.write('='*50 + '\n\n')
    f.write(f'Total POIs extracted: {len(pois)}\n\n')
    f.write('By original OSM category:\n')
    f.write(pois['category'].value_counts().to_string())
print('1. POI extraction summary saved')

# 2. LLM classification summary
llm = pd.read_csv('../data/processed/pois_llm_classified.csv')
with open('../results/02_llm_classification_summary.txt', 'w') as f:
    f.write('LLM CLASSIFICATION SUMMARY\n')
    f.write('='*50 + '\n\n')
    f.write(f'Total POIs: {len(llm)}\n')
    f.write(f'Successfully classified: {(llm["llm_category"] != "error").sum()}\n')
    f.write(f'Errors: {(llm["llm_category"] == "error").sum()}\n\n')
    f.write('By LLM category:\n')
    f.write(llm['llm_category'].value_counts().to_string())
print('2. LLM classification summary saved')

# 3. Spatial join summary
spatial = pd.read_csv('../data/processed/pois_with_lsoa.csv')
with open('../results/03_spatial_join_summary.txt', 'w') as f:
    f.write('SPATIAL JOIN SUMMARY\n')
    f.write('='*50 + '\n\n')
    f.write(f'Total POIs: {len(spatial)}\n')
    f.write(f'Matched to London LSOAs: {spatial["lsoa21cd"].notna().sum()}\n')
    f.write(f'Outside London boundary: {spatial["lsoa21cd"].isna().sum()}\n')
print('3. Spatial join summary saved')

# 4. Food environment dataset summary
food = pd.read_csv('../data/processed/lsoa_llm_food_environment.csv')
with open('../results/04_food_environment_summary.txt', 'w') as f:
    f.write('FOOD ENVIRONMENT DATASET SUMMARY\n')
    f.write('='*50 + '\n\n')
    f.write(f'Total London LSOAs: {len(food)}\n\n')
    f.write('Outlet counts per LSOA (descriptive statistics):\n\n')
    f.write(food.describe().to_string())
print('4. Food environment summary saved')

# 5. Distance-weighted metrics summary
dw = pd.read_csv('../data/processed/lsoa_distance_weighted.csv')
with open('../results/05_distance_weighted_summary.txt', 'w') as f:
    f.write('DISTANCE-WEIGHTED METRICS SUMMARY\n')
    f.write('='*50 + '\n\n')
    f.write(f'LSOAs: {len(dw)}\n\n')
    f.write('Distance-weighted scores per LSOA:\n\n')
    f.write(dw.describe().to_string())
print('5. Distance-weighted metrics summary saved')

# 6. MedSat health outcomes summary
final = pd.read_csv('../data/processed/final_analysis_dataset.csv')
health_cols = ['o_diabetes_quantity_per_capita', 'o_hypertension_quantity_per_capita',
               'o_depression_quantity_per_capita', 'o_anxiety_quantity_per_capita',
               'o_asthma_quantity_per_capita']
with open('../results/06_health_outcomes_summary.txt', 'w') as f:
    f.write('MEDSAT HEALTH OUTCOMES SUMMARY\n')
    f.write('='*50 + '\n\n')
    f.write(f'Total rows (LSOAs x years): {len(final)}\n')
    f.write(f'Years: 2019 and 2020\n\n')
    for yr in [2019, 2020]:
        f.write(f'\n--- {yr} ---\n')
        yr_data = final[final['year'] == yr][health_cols]
        f.write(yr_data.describe().to_string())
        f.write('\n')
print('6. Health outcomes summary saved')

# 7. Diabetes regression results
reg = pd.read_csv('../data/processed/regression_results.csv')
with open('../results/07_regression_model_comparison.txt', 'w') as f:
    f.write('REGRESSION MODEL COMPARISON — DIABETES\n')
    f.write('='*50 + '\n\n')
    f.write('Outcome: Diabetes prescription rate per capita (2019)\n\n')
    f.write(reg.to_string())
    f.write('\n\n')
    f.write('Key findings:\n')
    f.write('- OLS: dw_indep_fresh coef = -0.817 (p=0.016) — significant protective effect\n')
    f.write('- Spatial Lag: dw_indep_fresh coef = 0.074 (p=0.473) — non-significant after spatial control\n')
    f.write('- Spatial Error: dw_indep_fresh coef = 0.175 (p=0.161) — non-significant\n')
    f.write('- Best model: Spatial Lag (lowest AIC = 29,860, R2 = 0.92)\n')
    f.write('- Spatial autocorrelation confirmed (LM tests p<0.001)\n')
print('7. Diabetes regression results saved')

# 8. Hypertension regression results
reg_hyp = pd.read_csv('../data/processed/regression_results_hypertension.csv')
with open('../results/10_regression_hypertension_comparison.txt', 'w') as f:
    f.write('REGRESSION MODEL COMPARISON — HYPERTENSION\n')
    f.write('='*50 + '\n\n')
    f.write('Outcome: Hypertension prescription rate per capita (2019)\n\n')
    f.write(reg_hyp.to_string())
    f.write('\n\n')
    f.write('Key findings:\n')
    f.write('- OLS: dw_indep_fresh coef = -3.361 (p<0.001) — significant protective effect\n')
    f.write('- Spatial Lag: dw_indep_fresh coef = -0.061 (p=0.694) — non-significant after spatial control\n')
    f.write('- Spatial Error: dw_indep_fresh coef = 0.150 (p=0.426) — non-significant\n')
    f.write('- Best model: Spatial Lag (lowest AIC = 33,641, R2 = 0.93)\n')
print('8. Hypertension regression results saved')

print('\nAll results saved to ~/Documents/london-food-health/results/')
print('Files created:')
for f in sorted(os.listdir('../results')):
    print(f'  {f}')
