# London Food Environment & Health Inequalities

**MSc Data Science Dissertation — King's College London**  
**Supervisor:** Dr Linus Dietz | **Author:** Ansh Aggarwal

---

## Overview

This project investigates whether access to market streets and independent food retail in London acts as a protective factor against diet-related health inequalities, after controlling for deprivation, fast food density, and ethnicity.

Existing food environment research almost exclusively measures the binary of fast food outlets vs supermarkets — ignoring the ecosystem of independent traders, market streets, and ethnic food retailers that are central to food access in London's most deprived communities. This project fills that gap.

---

## Research Questions

**RQ1 (Health):** Does proximity to market streets and independent fresh food retailers at the LSOA level act as a protective factor against diet-related health outcomes (diabetes, hypertension), after controlling for IMD, fast food density, population density, and ethnicity?

**RQ2 (Measurement):** How can market streets and independent food retail be systematically quantified across London's 4,994 LSOAs using OpenStreetMap and LLM-based classification?

**RQ3 (Spatial Equity):** Are the most deprived LSOAs simultaneously underserved by both formal and informal food retail — constituting compound food environment disadvantage?

---

## Key Results

- 10,907 food retail POIs extracted from OpenStreetMap across Greater London
- LLM annotation (Claude API) used to classify all POIs into independent fresh food, chain supermarket, convenience chain, fast food, market, and other
- 9,980 POIs successfully matched to London's 4,994 LSOAs via spatial join
- Distance-weighted food environment metrics constructed per LSOA
- Moran's I confirmed strong spatial autocorrelation in health outcomes (diabetes I=0.907, hypertension I=0.910, p<0.001)
- OLS regression found significant protective association between independent fresh food access and diabetes (coef=-0.82, p=0.016) and hypertension (coef=-3.36, p<0.001)
- Spatial lag models showed this effect is partly mediated by spatial clustering of deprivation
- Borough analysis showed boroughs with highest independent fresh food access have diabetes rates more than half those of boroughs with lowest access

---

## Data Sources

| Dataset | Source | Purpose |
|---|---|---|
| LSOA Boundaries | ONS Open Geography Portal | Spatial unit of analysis |
| Food retail POIs | OpenStreetMap (Overpass API) | Independent retail + market streets |
| Health outcomes | MedSat (NeurIPS 2023, Scepanovic et al.) | Diabetes + hypertension Rx rates |
| Deprivation | Index of Multiple Deprivation (IMD) | Socioeconomic control |

---

## Project Structure
london-food-health-inequality/
├── src/ # All Python scripts
├── data/
│ ├── raw/ # Raw data (not committed — too large)
│ └── processed/ # Processed LSOA-level datasets
├── results/ # All regression and analysis outputs
├── dashboard/ # Interactive Plotly dashboard (run dashboard_plotly.py to generate)
└── README.md

---

## Methods

1. OSM Extraction — Overpass API, 4 food retail categories
2. LLM Classification — Claude API annotates each POI by outlet type
3. Spatial Join — POIs matched to LSOAs using British National Grid coordinate conversion
4. Distance-Weighted Metrics — outlet influence weighted by distance and outlet size
5. Spatial Regression — OLS, spatial-lag, spatial-error models (PySAL/spreg)
6. Dashboard — Interactive Plotly choropleth maps

---

## Dependencies
pandas
geopandas
fiona
shapely
requests
libpysal
spreg
esda
plotly
folium

Install with:
```bash
pip install pandas geopandas fiona shapely requests libpysal spreg esda plotly folium
```

---

*Department of Informatics, King's College London, 2026*
