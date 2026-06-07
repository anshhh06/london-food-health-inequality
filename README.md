# London Food Environment & Health Inequalities

**MSc Data Science Dissertation — King's College London**
**Supervisor:** Dr Linus Dietz | **Author:** Ansh Aggarwal

## Overview

This project investigates whether access to **market streets and independent food retail** in London acts as a protective factor against diet-related health inequalities, after controlling for deprivation, fast food density, and ethnicity.

Existing food environment research almost exclusively measures the binary of fast food outlets vs. supermarkets — systematically ignoring the rich ecosystem of independent traders, market streets, and ethnic food retailers that are central to food access in London's most deprived communities. This project fills that gap.


## Research Questions

**RQ1 (Health):** Does proximity to market streets and independent fresh food retailers at the LSOA level act as a protective factor against diet-related health outcomes (diabetes, hypertension), after controlling for IMD, fast food density, population density, and ethnicity?

**RQ2 (Measurement):** How can market streets and independent food retail be systematically quantified across London's 4,994 LSOAs using OpenStreetMap and the London Datastore?

**RQ3 (Spatial Equity):** Are the most deprived LSOAs simultaneously underserved by both formal and informal food retail — constituting compound food environment disadvantage?


## Data Sources

| Dataset | Source | Purpose |
|---|---|---|
| LSOA Boundaries | ONS Open Geography Portal | Spatial unit of analysis |
| Food retail POIs | OpenStreetMap (Overpass API) | Independent retail + market streets |
| Food outlet data | FSA Food Hygiene Rating Scheme | Validation + fast food density |
| Health outcomes | MedSat (NeurIPS 2023) | Diabetes + hypertension Rx rates |
| Deprivation | Index of Multiple Deprivation (IMD) | Socioeconomic control |


## Project Structure

```
london-food-health-inequality/
│
├── data/
│   ├── raw/               # Raw downloaded data (not committed)
│   └── processed/         # Cleaned, LSOA-level merged data
│
├── notebooks/
│   ├── 01_osm_extraction.ipynb       # OSM food retail data pull
│   ├── 02_data_cleaning.ipynb        # FSA + MedSat preprocessing
│   ├── 03_metric_construction.ipynb  # LSOA-level food environment metrics
│   ├── 04_spatial_modelling.ipynb    # OLS, spatial-lag, spatial-error models
│   └── 05_visualisation.ipynb        # Dashboard + maps
│
├── src/
│   ├── osm_query.py        # Overpass API query functions
│   ├── metrics.py          # Food environment metric construction
│   └── spatial_models.py   # Spatial regression wrappers
│
├── dashboard/              # Interactive Leaflet.js / React dashboard
│
├── requirements.txt
└── README.md
```

## Methods

1. **Data Collection** — OSM Overpass API, FSA open data, MedSat, ONS boundaries
2. **Metric Construction** — LSOA-level independent food retail density, market street proximity score, composite food environment index
3. **Spatial Modelling** — OLS, spatial-lag, and spatial-error regression (PySAL)
4. **Visualisation** — Interactive geospatial dashboard (React + Leaflet.js)


## Timeline

| Phase | Task | Deadline |
|---|---|---|
| Data Collection | OSM, FSA, MedSat, LSOA boundaries | May 2026 |
| Metric Construction | LSOA-level food environment index | June 2026 |
| Spatial Modelling | Regression analysis | June–July 2026 |
| Dashboard | Interactive visualisation | July 2026 |
| Report Writing | Full dissertation | July–August 2026 |
| Final Submission | Report + video | 6 August 2026 |


## Dependencies

```
geopandas
osmnx
requests
pandas
numpy
matplotlib
libpysal
spreg
esda
folium
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

*Department of Informatics, King's College London, 2026*
