"""
Data Loader — read Airbnb listings and reviews CSVs with relative paths.
"""

from __future__ import annotations

import pandas as pd
from .config import DATA_DIR, DEFAULT_CITY, AVAILABLE_CITIES


def load_listings(city: str = DEFAULT_CITY) -> pd.DataFrame:
    """Load the detailed listings CSV for a given city.

    Tries ``listings.csv.gz`` first, then falls back to
    ``listings (2).csv`` (the uncompressed variant some cities have).

    Parameters
    ----------
    city : str
        City name matching a subfolder in ``Data Files/``.
        Defaults to :pydata:`config.DEFAULT_CITY`.
    """
    city_dir = DATA_DIR / city
    gz_path = city_dir / "listings.csv.gz"
    csv_path = city_dir / "listings (2).csv"

    if gz_path.exists():
        return pd.read_csv(gz_path, low_memory=False)
    elif csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    else:
        raise FileNotFoundError(
            f"No listings file found in {city_dir}. "
            "Expected 'listings.csv.gz' or 'listings (2).csv'."
        )


def load_reviews(city: str = DEFAULT_CITY) -> pd.DataFrame:
    """Load the reviews CSV for a given city.

    Tries ``reviews.csv.gz`` first, then ``reviews (1).csv``.

    Parameters
    ----------
    city : str
        City name matching a subfolder in ``Data Files/``.
        Defaults to :pydata:`config.DEFAULT_CITY`.
    """
    city_dir = DATA_DIR / city
    gz_path = city_dir / "reviews.csv.gz"
    csv_path = city_dir / "reviews (1).csv"

    if gz_path.exists():
        return pd.read_csv(gz_path, low_memory=False)
    elif csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    else:
        raise FileNotFoundError(
            f"No reviews file found in {city_dir}. "
            "Expected 'reviews.csv.gz' or 'reviews (1).csv'."
        )


def load_all_listings(
    cities: list[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate listings from multiple cities.

    Adds a ``city`` column to identify each listing's origin.

    Parameters
    ----------
    cities : list[str] or None
        Cities to load.  Defaults to all :pydata:`AVAILABLE_CITIES`.
    """
    cities = cities or AVAILABLE_CITIES
    frames = []
    for city in cities:
        try:
            df = load_listings(city)
            df["city"] = city
            frames.append(df)
        except FileNotFoundError:
            print(f"⚠  Skipping {city} — no listings file found.")
    if not frames:
        raise FileNotFoundError("No listings data found for any city.")
    return pd.concat(frames, ignore_index=True)
