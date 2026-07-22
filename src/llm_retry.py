import os
import pandas as pd
import requests
import time

print('Loading classified POIs...')
df = pd.read_csv('../data/processed/pois_llm_classified.csv')
errors = df[df['llm_category'] == 'error'].copy()
print(f'Errors to retry: {len(errors)}')

def classify_poi(name, shop, amenity, cuisine):
    prompt = f"""Classify this London food outlet. Reply with EXACTLY one of these words only:
independent_fresh_food
chain_supermarket
convenience_chain
fast_food
market
other

Name: {name}
Shop: {shop}
Amenity: {amenity}
Cuisine: {cuisine}

Your answer (one word only):"""

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
                "max_tokens": 10,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["content"][0]["text"].strip()
        elif response.status_code == 529:
            time.sleep(5)
            return "error"
        else:
            return "error"
    except Exception as e:
        return "error"

for i, (idx, row) in enumerate(errors.iterrows()):
    result = classify_poi(row['name'], row['shop'], row['amenity'], row['cuisine'])
    df.at[idx, 'llm_category'] = result
    
    if i % 100 == 0:
        print(f'Retried {i}/{len(errors)}...')
        df.to_csv('../data/processed/pois_llm_classified.csv', index=False)
    
    time.sleep(0.5)

df.to_csv('../data/processed/pois_llm_classified.csv', index=False)
print('\nDone!')
print(df['llm_category'].value_counts())
