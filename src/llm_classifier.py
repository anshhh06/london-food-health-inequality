import os
import pandas as pd
import requests
import time

print('Loading POIs...')
pois = pd.read_csv('../data/raw/london_food_pois.csv')
print(f'Total POIs: {len(pois)}')

def classify_poi(name, shop, amenity, cuisine):
    prompt = f"""Classify this London food outlet into ONE category:
- independent_fresh_food: independent non-chain shops selling fresh food (greengrocers, butchers, fishmongers, bakeries, delis, ethnic grocers)
- chain_supermarket: Tesco, Sainsburys, Waitrose, M&S, Asda, Morrisons, Lidl, Aldi, Co-op
- convenience_chain: Tesco Express, Sainsburys Local, One Stop, Spar
- fast_food: takeaways, fast food, restaurants
- market: traditional market streets or marketplaces
- other: anything else

Name: {name}
Shop tag: {shop}
Amenity: {amenity}
Cuisine: {cuisine}

Reply with ONLY the category name."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["content"][0]["text"].strip()
        else:
            return "error"
    except Exception as e:
        return "error"

results = []
for i, row in pois.iterrows():
    result = classify_poi(row['name'], row['shop'], row['amenity'], row['cuisine'])
    results.append(result)
    if i % 100 == 0:
        print(f'Processed {i}/{len(pois)}...')
        pois_copy = pois.copy()
        pois_copy['llm_category'] = results + [None] * (len(pois) - len(results))
        pois_copy.to_csv('../data/processed/pois_llm_classified.csv', index=False)
    time.sleep(0.3)

pois['llm_category'] = results
pois.to_csv('../data/processed/pois_llm_classified.csv', index=False)
print('\nDone!')
print(pois['llm_category'].value_counts())
