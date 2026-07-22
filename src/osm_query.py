"""
osm_query.py
============
Extracts food retail points of interest from OpenStreetMap
for Greater London using the Overpass API.

Categories extracted:
    - Independent fresh food retail (greengrocers, butchers, delis, etc.)
    - Market streets and marketplaces
    - Supermarkets (for comparison baseline)
    - Fast food outlets (for comparison baseline)

Author: Ansh Aggarwal
Project: London Food Environment & Health Inequalities
Supervisor: Dr Linus Dietz, King's College London
"""

import requests
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import time
import os
import pyproj
pyproj.datadir.set_data_dir("/Users/anshaggarwal/miniconda3/share/proj")

# ── Output directory ──────────────────────────────────────
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── Overpass API endpoint ─────────────────────────────────
OVERPASS_URL = "http://overpass-api.de/api/interpreter"

# ── Greater London bounding box ───────────────────────────
# (south, west, north, east)
LONDON_BBOX = "51.2867,-0.5103,51.6918,0.3340"


def run_overpass_query(query: str) -> dict:
    """Send a query to the Overpass API and return the JSON response."""
    headers = {
        "User-Agent": "london-food-health-research/1.0 (ansh.aggarwal@kcl.ac.uk)"
    }
    for attempt in range(3):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers=headers,
                timeout=180
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}. Retrying in 10s...")
            time.sleep(10)
    raise Exception("Overpass API failed after 3 attempts")


def elements_to_geodataframe(elements: list, category: str) -> gpd.GeoDataFrame:
    """
    Convert a list of Overpass API elements to a GeoDataFrame.
    Only keeps nodes (point features).
    """
    records = []
    for el in elements:
        if el["type"] == "node":
            tags = el.get("tags", {})
            records.append({
                "osm_id":   el["id"],
                "lat":      el["lat"],
                "lon":      el["lon"],
                "category": category,
                "name":     tags.get("name", "Unknown"),
                "shop":     tags.get("shop", ""),
                "amenity":  tags.get("amenity", ""),
                "cuisine":  tags.get("cuisine", ""),
                "geometry": Point(el["lon"], el["lat"])
            })
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


# ── Query definitions ─────────────────────────────────────

def query_independent_fresh_food() -> gpd.GeoDataFrame:
    """
    Extract independent fresh food retailers:
    greengrocers, butchers, fishmongers, delis, bakeries,
    farm shops, health food shops.
    """
    print("Querying independent fresh food retailers...")
    query = f"""
    [out:json][timeout:120];
    (
      node["shop"="greengrocer"]({LONDON_BBOX});
      node["shop"="butcher"]({LONDON_BBOX});
      node["shop"="fishmonger"]({LONDON_BBOX});
      node["shop"="deli"]({LONDON_BBOX});
      node["shop"="bakery"]({LONDON_BBOX});
      node["shop"="farm"]({LONDON_BBOX});
      node["shop"="health_food"]({LONDON_BBOX});
      node["shop"="food"]({LONDON_BBOX});
    );
    out body;
    """
    data = run_overpass_query(query)
    gdf = elements_to_geodataframe(data["elements"], "independent_fresh_food")
    print(f"  Found {len(gdf)} independent fresh food retailers")
    return gdf


def query_markets() -> gpd.GeoDataFrame:
    """
    Extract market streets and marketplaces.
    """
    print("Querying markets and market streets...")
    query = f"""
    [out:json][timeout:120];
    (
      node["amenity"="marketplace"]({LONDON_BBOX});
      node["shop"="market"]({LONDON_BBOX});
    );
    out body;
    """
    data = run_overpass_query(query)
    gdf = elements_to_geodataframe(data["elements"], "market")
    print(f"  Found {len(gdf)} market locations")
    return gdf


def query_supermarkets() -> gpd.GeoDataFrame:
    """
    Extract supermarkets and convenience stores (baseline comparison).
    """
    print("Querying supermarkets...")
    query = f"""
    [out:json][timeout:120];
    (
      node["shop"="supermarket"]({LONDON_BBOX});
      node["shop"="convenience"]({LONDON_BBOX});
    );
    out body;
    """
    data = run_overpass_query(query)
    gdf = elements_to_geodataframe(data["elements"], "supermarket")
    print(f"  Found {len(gdf)} supermarkets/convenience stores")
    return gdf


def query_fast_food() -> gpd.GeoDataFrame:
    """
    Extract fast food outlets (baseline comparison).
    """
    print("Querying fast food outlets...")
    query = f"""
    [out:json][timeout:120];
    (
      node["amenity"="fast_food"]({LONDON_BBOX});
    );
    out body;
    """
    data = run_overpass_query(query)
    gdf = elements_to_geodataframe(data["elements"], "fast_food")
    print(f"  Found {len(gdf)} fast food outlets")
    return gdf


# ── Main extraction pipeline ──────────────────────────────

def extract_all(save=True) -> gpd.GeoDataFrame:
    """
    Run all four queries, combine results, and optionally save to file.
    Returns a single GeoDataFrame with all food retail POIs.
    """
    gdfs = []

    # Run each query with a small pause between them
    # to be respectful of the Overpass API rate limits
    for query_fn in [
        query_independent_fresh_food,
        query_markets,
        query_supermarkets,
        query_fast_food
    ]:
        gdf = query_fn()
        gdfs.append(gdf)
        time.sleep(3)

    # Combine all into one GeoDataFrame
    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs="EPSG:4326"
    )

    print(f"\nTotal POIs extracted: {len(combined)}")
    print(combined["category"].value_counts())

    if save:
      csv_path = "data/raw/london_food_pois.csv"
      combined.drop(columns="geometry").to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}") 

    return combined


# ── Spatial join to LSOAs ─────────────────────────────────

def assign_pois_to_lsoas(
    pois: gpd.GeoDataFrame,
    lsoa_path: str = "data/raw/london_lsoas.geojson"
) -> gpd.GeoDataFrame:
    """
    Spatially join food retail POIs to London LSOA boundaries.
    Each POI gets assigned the LSOA it falls within.

    Parameters
    ----------
    pois : GeoDataFrame of food retail POIs
    lsoa_path : path to London LSOA boundary GeoJSON

    Returns
    -------
    GeoDataFrame with LSOA code added to each POI
    """
    print("\nLoading LSOA boundaries...")
    lsoas = gpd.read_file(lsoa_path)

    # Ensure both are in the same CRS
    lsoas = lsoas.to_crs("EPSG:4326")
    pois = pois.to_crs("EPSG:4326")

    print("Performing spatial join...")
    joined = gpd.sjoin(pois, lsoas[["LSOA21CD", "LSOA21NM", "geometry"]],
                       how="left", predicate="within")

    print(f"POIs successfully assigned to LSOAs: {joined['LSOA21CD'].notna().sum()}")
    print(f"POIs outside LSOA boundaries: {joined['LSOA21CD'].isna().sum()}")

    return joined


# ── Count POIs per LSOA ───────────────────────────────────

def count_pois_per_lsoa(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Count the number of each food retail category per LSOA.
    Returns a wide-format DataFrame ready for merging with MedSat.
    """
    counts = (
        joined
        .groupby(["LSOA21CD", "category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    counts.columns.name = None

    # Rename for clarity
    rename_map = {
        "independent_fresh_food": "count_independent_fresh",
        "market":                 "count_market",
        "supermarket":            "count_supermarket",
        "fast_food":              "count_fast_food"
    }
    counts = counts.rename(columns=rename_map)

    # Save
    out_path = "data/processed/lsoa_poi_counts.csv"
    counts.to_csv(out_path, index=False)
    print(f"\nLSOA POI counts saved to {out_path}")
    print(counts.head())

    return counts


# ── Entry point ───────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("London Food Environment — OSM Data Extraction")
    print("=" * 55)

    # Step 1: Extract all POIs from OSM
    pois = extract_all(save=True)

    # Step 2: Assign to LSOAs (requires LSOA boundaries downloaded)
    # Comment this out if you haven't downloaded LSOA boundaries yet
    # joined = assign_pois_to_lsoas(pois)
    # counts = count_pois_per_lsoa(joined)

    print("\nDone. Next step: download LSOA boundaries and run assign_pois_to_lsoas()")
