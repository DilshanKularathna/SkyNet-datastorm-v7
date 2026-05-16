"""
=============================================================================
Pipeline Step 04 — POI Data Extraction (OpenStreetMap / Local PBF)
=============================================================================

Two modes (choose based on your environment):

  Mode A — Local PBF (RECOMMENDED, fastest, no rate limits):
    1. Download: https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf
    2. Place in:  data/external/sri-lanka-latest.osm.pbf
    3. Run:       python pipeline/04_poi_scraping.py

  Mode B — Overpass API (fallback, slower, network-dependent):
    1. Set USE_LOCAL_PBF = False below
    2. Run:       python pipeline/04_poi_scraping.py
       Add --resume flag to continue an interrupted run:
       python pipeline/04_poi_scraping.py --resume

FIXES vs original:
  - Removed duplicate if __name__ == "__main__" block.
  - Added resume parameter to run_poi_scraping() (was called with resume=True
    but the function signature didn't accept it — caused TypeError).
  - Improved progress logging and partial-save logic.
  - Added is_urban_score (continuous, not binary) as a richer feature.
=============================================================================
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# ── Mode switch ───────────────────────────────────────────────────────────────
USE_LOCAL_PBF = True   # Set False to use Overpass API instead

PBF_PATH = ROOT / "data" / "external" / "sri-lanka-latest.osm.pbf"

SILVER_DIR = ROOT / "data" / "silver"
EXTERNAL   = ROOT / "data" / "external"
EXTERNAL.mkdir(parents=True, exist_ok=True)

# ── Overpass endpoints (rotated on failure) ───────────────────────────────────
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# ── POI definitions: (label, osm_key, osm_value) ─────────────────────────────
POI_TYPES = [
    ("schools",            "amenity",  "school"),
    ("universities",       "amenity",  "university"),
    ("bus_stops",          "highway",  "bus_stop"),
    ("hospitals",          "amenity",  "hospital"),
    ("clinics",            "amenity",  "clinic"),
    ("hotels",             "tourism",  "hotel"),
    ("guesthouses",        "tourism",  "guest_house"),
    ("markets",            "amenity",  "marketplace"),
    ("supermarkets",       "shop",     "supermarket"),
    ("places_of_worship",  "amenity",  "place_of_worship"),
    ("pharmacies",         "amenity",  "pharmacy"),
    ("fuel_stations",      "amenity",  "fuel"),
    ("restaurants",        "amenity",  "restaurant"),
    ("banks",              "amenity",  "bank"),
    ("construction_sites", "landuse",  "construction"),
]

FOOTFALL_WEIGHTS = {
    "schools":            3.0,
    "universities":       4.0,
    "bus_stops":          4.5,
    "hospitals":          3.5,
    "clinics":            2.0,
    "hotels":             3.0,
    "guesthouses":        2.0,
    "markets":            5.0,
    "supermarkets":       3.0,
    "places_of_worship":  2.5,
    "pharmacies":         2.0,
    "fuel_stations":      2.0,
    "restaurants":        1.5,
    "banks":              2.0,
    "construction_sites": 2.5,
}

RADIUS_M          = 500
REQUEST_DELAY_SEC = 1.2
MAX_RETRIES       = 3
TIMEOUT_SEC       = 20
BATCH_SAVE_EVERY  = 50


# =============================================================================
# HTTP session
# =============================================================================
def _make_session() -> requests.Session:
    session = requests.Session()
    retry   = Retry(total=MAX_RETRIES, backoff_factor=1.5,
                    status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

SESSION = _make_session()


# =============================================================================
# Overpass API — single POI count
# =============================================================================
def query_poi_count(lat: float, lon: float, key: str, value: str,
                    radius: int = RADIUS_M, endpoint_idx: int = 0) -> int:
    endpoint = OVERPASS_ENDPOINTS[endpoint_idx % len(OVERPASS_ENDPOINTS)]
    query = (
        f'[out:json][timeout:15];'
        f'(node["{key}"="{value}"](around:{radius},{lat},{lon});'
        f' way["{key}"="{value}"](around:{radius},{lat},{lon});'
        f' relation["{key}"="{value}"](around:{radius},{lat},{lon}););'
        f'out count;'
    )
    try:
        resp  = SESSION.get(endpoint, params={"data": query}, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        data  = resp.json()
        count = int(data.get("elements", [{}])[0].get("tags", {}).get("total", 0))
        return count
    except Exception as exc:
        log.debug("  Overpass error (%s=%s @ %.4f,%.4f): %s", key, value, lat, lon, exc)
        return 0


# =============================================================================
# Overpass API — all POIs for one outlet
# =============================================================================
def get_all_pois_for_outlet(outlet_id: str, lat: float, lon: float) -> dict:
    row = {"Outlet_ID": outlet_id}
    for label, key, value in POI_TYPES:
        row[f"poi_{label}"] = query_poi_count(lat, lon, key, value)
        time.sleep(0.1)

    row["poi_footfall_score"] = sum(
        row.get(f"poi_{label}", 0) * FOOTFALL_WEIGHTS.get(label, 1.0)
        for label, _, _ in POI_TYPES
    )
    row["poi_total_count"] = sum(
        row.get(f"poi_{label}", 0) for label, _, _ in POI_TYPES
    )
    # Continuous urban score (0–1) — richer than binary is_urban
    row["is_urban"]       = int(row["poi_total_count"] > 10)
    row["is_urban_score"] = min(row["poi_total_count"] / 30.0, 1.0)  # saturates at 30 POIs
    return row


# =============================================================================
# Overpass mode — batch run with resume
# =============================================================================
def run_overpass_mode(coords_df: pd.DataFrame, resume: bool = False):
    output_path   = EXTERNAL / "poi_raw.csv"
    features_path = EXTERNAL / "poi_features.csv"

    already_done = set()
    results      = []

    if resume and output_path.exists():
        existing = pd.read_csv(output_path)
        already_done = set(existing["Outlet_ID"].astype(str).unique())
        results      = existing.to_dict("records")
        log.info("  Resuming — %d outlets already scraped.", len(already_done))

    pending = coords_df[~coords_df["Outlet_ID"].astype(str).isin(already_done)]
    total   = len(pending)
    log.info("  Outlets to scrape: %d", total)

    for i, (_, row) in enumerate(pending.iterrows(), 1):
        outlet_id = str(row["Outlet_ID"])
        lat, lon  = float(row["Latitude"]), float(row["Longitude"])
        poi_row   = get_all_pois_for_outlet(outlet_id, lat, lon)
        results.append(poi_row)

        if i % 10 == 0:
            log.info("  Progress: %d / %d (%.1f%%)", i, total, i / total * 100)

        if i % BATCH_SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
            log.info("  Partial save — %d rows written.", len(results))

        time.sleep(REQUEST_DELAY_SEC)

    df = pd.DataFrame(results)
    df.to_csv(output_path,   index=False)
    df.to_csv(features_path, index=False)
    log.info("  Overpass scraping complete — %d outlets saved.", len(df))
    return df


# =============================================================================
# Local PBF mode
# =============================================================================
def run_local_poi_extraction(coords_df: pd.DataFrame, pbf_path: Path):
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        log.error("Local extraction requires geopandas and shapely.")
        log.info("Install with: pip install geopandas shapely pyogrio")
        return None

    log.info("Starting LOCAL extraction from: %s", pbf_path.name)

    outlets_gdf = gpd.GeoDataFrame(
        coords_df,
        geometry=gpd.points_from_xy(coords_df.Longitude, coords_df.Latitude),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    outlets_gdf["buffer"] = outlets_gdf.geometry.buffer(RADIUS_M)

    all_pois = []
    for layer in ["points", "multipolygons"]:
        log.info("  Reading layer '%s'...", layer)
        try:
            gdf = gpd.read_file(pbf_path, layer=layer, engine="pyogrio")
            keys_to_match = list({t[1] for t in POI_TYPES})
            available     = [k for k in keys_to_match if k in gdf.columns]
            if not available:
                continue
            mask   = gdf[available].notnull().any(axis=1)
            subset = gdf[mask].copy()
            if layer == "multipolygons":
                subset = subset.copy()
                subset.geometry = subset.geometry.centroid
            all_pois.append(subset)
        except Exception as exc:
            log.warning("  Could not read layer %s: %s", layer, exc)

    if not all_pois:
        log.error("No relevant POIs found in PBF layers.")
        return None

    pois_gdf = pd.concat(all_pois).to_crs("EPSG:3857")
    log.info("  Total candidate POIs: %d", len(pois_gdf))

    results = coords_df[["Outlet_ID"]].copy()
    for label, key, value in POI_TYPES:
        log.info("  Counting %s...", label)
        if key not in pois_gdf.columns:
            results[f"poi_{label}"] = 0
            continue
        cat = pois_gdf[pois_gdf[key] == value]
        if cat.empty:
            results[f"poi_{label}"] = 0
            continue
        joined = gpd.sjoin(
            outlets_gdf.set_geometry("buffer"), cat,
            how="inner", predicate="intersects"
        )
        counts = (
            joined.groupby("Outlet_ID").size()
            .reindex(results["Outlet_ID"], fill_value=0)
        )
        results[f"poi_{label}"] = counts.values

    results["poi_footfall_score"] = sum(
        results[f"poi_{label}"] * FOOTFALL_WEIGHTS.get(label, 1.0)
        for label, _, _ in POI_TYPES
    )
    results["poi_total_count"] = sum(
        results[f"poi_{label}"] for label, _, _ in POI_TYPES
    )
    results["is_urban"]       = (results["poi_total_count"] > 10).astype(int)
    results["is_urban_score"] = (results["poi_total_count"] / 30.0).clip(0, 1)

    return results


# =============================================================================
# Main orchestrator  (FIXED: single __main__ block, resume parameter added)
# =============================================================================
def run_poi_scraping(resume: bool = False):
    """
    Entry point for POI data extraction.

    Parameters
    ----------
    resume : bool
        If True, continue a previously interrupted Overpass API run.
        Not applicable in local PBF mode (always processes all outlets).
    """
    log.info("=" * 70)
    log.info("POI DATA EXTRACTION — mode=%s",
             "LOCAL PBF" if USE_LOCAL_PBF else "Overpass API")
    log.info("=" * 70)

    coords_df = pd.read_csv(SILVER_DIR / "coordinates_clean.csv", low_memory=False)
    coords_df = coords_df.dropna(subset=["Latitude", "Longitude"])
    log.info("Outlets with valid coordinates: %d", len(coords_df))

    output_path   = EXTERNAL / "poi_raw.csv"
    features_path = EXTERNAL / "poi_features.csv"

    if USE_LOCAL_PBF:
        if not PBF_PATH.exists():
            log.error("CRITICAL: PBF file not found at: %s", PBF_PATH)
            log.info("Download from: https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf")
            log.info("Falling back to Overpass API mode...")
            result_df = run_overpass_mode(coords_df, resume=resume)
        else:
            result = run_local_poi_extraction(coords_df, PBF_PATH)
            if result is None:
                log.error("Local extraction failed — falling back to Overpass API.")
                result_df = run_overpass_mode(coords_df, resume=resume)
            else:
                result_df = result if isinstance(result, pd.DataFrame) \
                            else pd.DataFrame(result)
    else:
        result_df = run_overpass_mode(coords_df, resume=resume)

    if result_df is not None and not result_df.empty:
        result_df.to_csv(output_path,   index=False)
        result_df.to_csv(features_path, index=False)
        log.info("=" * 70)
        log.info("POI features saved → %s  (%d outlets)", features_path, len(result_df))
        log.info("Top-line stats:")
        for col in ["poi_total_count", "poi_footfall_score", "is_urban"]:
            if col in result_df.columns:
                log.info("  %-25s  mean=%.2f  max=%.0f",
                         col, result_df[col].mean(), result_df[col].max())
    else:
        log.error("Extraction returned no results.")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POI scraping for Data Storm v7.0")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted Overpass API run")
    args = parser.parse_args()
    run_poi_scraping(resume=args.resume)