"""
POI Proximity Features — query OpenStreetMap's Overpass API for nearby
Points of Interest and compute count + nearest-distance features.

Strategy:
  1. One single Overpass API call fetches ALL POIs for the entire city bbox.
  2. POIs are stored in a local cache at ``cache/poi_raw_{city}.json``.
  3. For each listing, we compute Haversine distances to all POIs and
     derive count-within-radius and nearest-distance features.

This is far more efficient than per-listing queries: ~1 API call vs 75K+.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import CACHE_DIR, POI_CATEGORIES, POI_RADIUS_M, DEFAULT_CITY

# Overpass API endpoint (public, rate-limited)
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# =========================================================================
# Overpass query — fetch ALL POIs for a city bounding box in one call
# =========================================================================
def _build_city_query(
    south: float,
    west: float,
    north: float,
    east: float,
    categories: dict[str, str] | None = None,
) -> str:
    """Build a single Overpass QL query for all POI categories."""
    categories = categories or POI_CATEGORIES
    bbox = f"{south},{west},{north},{east}"

    stmts = []
    for tag_filter in categories.values():
        stmts.append(f"  node{tag_filter}({bbox});")
        stmts.append(f"  way{tag_filter}({bbox});")

    return (
        "[out:json][timeout:180];\n"
        "(\n"
        + "\n".join(stmts)
        + "\n);\n"
        "out center;"
    )


def _call_overpass(query: str, retries: int = 3, backoff: float = 10.0) -> list[dict]:
    """Send a query to the Overpass API with retry + back-off."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                _OVERPASS_URL,
                data={"data": query},
                timeout=180,
            )
            if resp.status_code == 429:
                wait = backoff * (2 ** attempt)
                print(f"    Rate-limited, waiting {wait:.0f}s ...")
                time.sleep(wait)
                continue
            if resp.status_code == 504:
                wait = backoff * (2 ** attempt)
                print(f"    Gateway timeout, waiting {wait:.0f}s ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except requests.RequestException as exc:
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"    Request error ({exc}), retrying in {wait:.0f}s ...")
                time.sleep(wait)
            else:
                raise
    return []


# =========================================================================
# Classify an OSM element into our POI categories
# =========================================================================
def _classify_element(tags: dict) -> list[str]:
    """Return matching POI category names for a set of OSM tags."""
    matched = []
    if tags.get("highway") == "bus_stop":
        matched.append("bus_stop")
    if tags.get("amenity") == "hospital":
        matched.append("hospital")
    tourism = tags.get("tourism", "")
    if tourism in ("attraction", "museum", "artwork", "gallery"):
        matched.append("tourist_attraction")
    if tags.get("shop") == "supermarket":
        matched.append("supermarket")
    return matched


# =========================================================================
# Cache helpers — raw POI data
# =========================================================================
def _cache_path(city: str) -> Path:
    """Return the cache file path for a city's raw POI data."""
    return CACHE_DIR / f"poi_raw_{city.lower().replace(' ', '_')}.json"


def _load_poi_cache(city: str) -> list[dict] | None:
    """Load cached POI data if available."""
    path = _cache_path(city)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_poi_cache(city: str, pois: list[dict]) -> None:
    """Save POI data to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(city)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pois, f)
    size_kb = path.stat().st_size / 1024
    print(f"  Saved POI cache: {path.name} ({len(pois)} POIs, {size_kb:.0f} KB)")


# =========================================================================
# Fetch or load city POIs
# =========================================================================
def _get_city_pois(
    df: pd.DataFrame,
    city: str,
) -> list[dict]:
    """Get all POIs for a city — from cache or Overpass API.

    Returns a list of dicts: ``[{"lat": ..., "lon": ..., "cat": ...}, ...]``
    """
    # Try cache first
    cached = _load_poi_cache(city)
    if cached is not None:
        print(f"  Loaded POI cache: {_cache_path(city).name} ({len(cached)} POIs)")
        return cached

    # Compute bounding box from listing coordinates + padding
    lat_min = df["latitude"].min() - 0.01  # ~1.1 km padding
    lat_max = df["latitude"].max() + 0.01
    lon_min = df["longitude"].min() - 0.015
    lon_max = df["longitude"].max() + 0.015

    print(f"  Querying Overpass API for {city} POIs ...")
    print(f"  Bounding box: ({lat_min:.4f}, {lon_min:.4f}) to "
          f"({lat_max:.4f}, {lon_max:.4f})")

    query = _build_city_query(lat_min, lon_min, lat_max, lon_max)
    elements = _call_overpass(query)
    print(f"  Received {len(elements)} raw elements from Overpass")

    # Parse into our format
    pois = []
    for el in elements:
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue
        cats = _classify_element(el.get("tags", {}))
        for cat in cats:
            pois.append({"lat": lat, "lon": lon, "cat": cat})

    print(f"  Classified {len(pois)} POIs across {len(POI_CATEGORIES)} categories:")
    for cat in POI_CATEGORIES:
        n = sum(1 for p in pois if p["cat"] == cat)
        print(f"    {cat:30s} {n:,}")

    # Save to cache
    _save_poi_cache(city, pois)
    return pois


# =========================================================================
# Compute features for each listing
# =========================================================================
def _compute_distances(
    listing_coords: np.ndarray,
    poi_coords: np.ndarray,
    radius_m: int,
    chunk_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised count + nearest distance for one POI category.

    Processes listings in chunks to avoid memory issues with large
    (N × M) distance matrices.

    Parameters
    ----------
    listing_coords : (N, 2) array of (lat, lon)
    poi_coords : (M, 2) array of (lat, lon)
    radius_m : search radius in metres
    chunk_size : number of listings per batch (default 2000)

    Returns
    -------
    counts : (N,) int array
    nearest : (N,) float array (metres, capped at radius_m)
    """
    N = listing_coords.shape[0]
    M = poi_coords.shape[0]

    if M == 0:
        return np.zeros(N, dtype=int), np.full(N, float(radius_m))

    all_counts = np.zeros(N, dtype=int)
    all_nearest = np.full(N, float(radius_m))

    # Pre-compute POI radians (shared across chunks)
    poi_lat_rad = np.radians(poi_coords[:, 0:1].T)   # (1, M)
    poi_lon_rad = np.radians(poi_coords[:, 1:2].T)    # (1, M)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = listing_coords[start:end]  # (chunk, 2)

        lat1 = np.radians(chunk[:, 0:1])   # (chunk, 1)
        lon1 = np.radians(chunk[:, 1:2])   # (chunk, 1)

        dlat = poi_lat_rad - lat1          # (chunk, M)
        dlon = poi_lon_rad - lon1          # (chunk, M)

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(poi_lat_rad) * np.sin(dlon / 2) ** 2
        )
        dists = 6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        all_counts[start:end] = np.sum(dists <= radius_m, axis=1).astype(int)
        all_nearest[start:end] = np.minimum(np.min(dists, axis=1), float(radius_m))

    return all_counts, np.round(all_nearest, 1)


# =========================================================================
# Public API
# =========================================================================
def compute_poi_features(
    df: pd.DataFrame,
    city: str = DEFAULT_CITY,
    radius_m: int = POI_RADIUS_M,
) -> pd.DataFrame:
    """Add POI proximity features to a listings DataFrame.

    For each POI category, creates two columns:
        - ``{cat}_count``      — number of POIs within *radius_m*
        - ``{cat}_nearest_m``  — distance to nearest POI (metres)

    Uses a disk cache at ``cache/poi_raw_{city}.json``.  The first run
    queries the Overpass API (~1 call); subsequent runs load from cache.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``latitude`` and ``longitude`` columns.
    city : str
        City name (used for cache filename).
    radius_m : int
        Search radius in metres (default 650).
    """
    df = df.copy()

    # 1. Get all POIs for the city (cached or fresh)
    pois = _get_city_pois(df, city)

    # 2. Prepare listing coordinates
    listing_coords = df[["latitude", "longitude"]].values  # (N, 2)

    # 3. Compute features per category
    for cat in POI_CATEGORIES:
        cat_pois = [(p["lat"], p["lon"]) for p in pois if p["cat"] == cat]
        poi_coords = np.array(cat_pois) if cat_pois else np.empty((0, 2))

        counts, nearest = _compute_distances(listing_coords, poi_coords, radius_m)
        df[f"{cat}_count"] = counts
        df[f"{cat}_nearest_m"] = nearest

    total_new = len(POI_CATEGORIES) * 2
    print(f"  Added {total_new} POI feature columns")

    return df
