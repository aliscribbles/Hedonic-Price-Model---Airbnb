"""
Feature Engineering — amenity dummies, VADER sentiment, Haversine distance,
neighbourhood encoding, host feature extraction, and date-derived features.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .config import (
    AMENITY_KEYWORDS,
    CITY_CENTERS,
    DEFAULT_CITY,
    PROPERTY_TYPES_TO_KEEP,
    TEXT_COLUMNS_TO_DROP_FINAL,
    URL_AND_REDUNDANT_COLUMNS,
)


# ==========================================================================
# VADER Sentiment Analysis
# ==========================================================================
def compute_vader_sentiment(reviews: pd.DataFrame) -> pd.DataFrame:
    """Add VADER polarity columns (neg, pos, neu, compound) to reviews.

    Requires ``nltk`` and the ``vader_lexicon`` resource.
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk.download("vader_lexicon", quiet=True)
    sid = SentimentIntensityAnalyzer()

    reviews = reviews.dropna(subset=["comments"]).copy()

    scores = reviews["comments"].apply(sid.polarity_scores).apply(pd.Series)
    reviews[["neg", "neu", "pos", "compound"]] = scores
    reviews["polarity_value"] = scores["compound"]

    return reviews


def aggregate_sentiment_by_listing(reviews: pd.DataFrame) -> pd.DataFrame:
    """Return the mean polarity value per ``listing_id``.

    Returns a DataFrame with columns ``listing_id`` and ``polarity_value``.
    """
    return (
        reviews.groupby("listing_id")["polarity_value"]
        .mean()
        .reset_index()
    )


def merge_sentiment(
    listings: pd.DataFrame,
    sentiment_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Merge aggregated sentiment scores into listings on ``id``."""
    return pd.merge(
        listings,
        sentiment_agg,
        left_on="id",
        right_on="listing_id",
        how="left",
    ).drop(columns=["listing_id"], errors="ignore")


# ==========================================================================
# Amenity Dummy Variables
# ==========================================================================
def create_amenity_dummies(
    df: pd.DataFrame,
    amenity_keywords: dict[str, list[str]] | None = None,
    amenity_col: str = "amenities",
) -> pd.DataFrame:
    """Create binary dummy columns for each amenity category.

    For every category in *amenity_keywords*, a new column is set to ``1``
    if any of the category's keywords appear (case-insensitive) in the
    listing's amenities string.
    """
    df = df.copy()
    amenity_keywords = amenity_keywords or AMENITY_KEYWORDS

    amenities_lower = df[amenity_col].astype(str).str.lower()

    for col_name, keywords in amenity_keywords.items():
        pattern = "|".join(re.escape(kw.lower()) for kw in keywords)
        df[col_name] = amenities_lower.str.contains(pattern, regex=True).astype(int)

    return df


# ==========================================================================
# Host Features
# ==========================================================================
def encode_host_verifications(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode ``host_verifications`` into email / phone columns."""
    df = df.copy()
    verif = df["host_verifications"].astype(str).str.lower()

    df["email_verified"] = verif.str.contains("email|work_email", regex=True).astype(int)
    df["phone_verified"] = verif.str.contains("phone").astype(int)

    df = df.drop(columns=["host_verifications"], errors="ignore")
    return df


def encode_host_response_time(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode ``host_response_time`` into four binary columns."""
    df = df.copy()
    rt = df["host_response_time"].astype(str).str.lower()

    df["host_response_within_hour"] = rt.str.contains("within an hour").astype(int)
    df["host_response_few_hours"]   = rt.str.contains("within a few hours").astype(int)
    df["host_response_within_day"]  = rt.str.contains("within a day").astype(int)
    df["host_response_few_days"]    = rt.str.contains("a few days or more").astype(int)

    df = df.drop(columns=["host_response_time"], errors="ignore")
    return df


# ==========================================================================
# Property Type Grouping
# ==========================================================================
def group_rare_property_types(
    df: pd.DataFrame,
    keep: list[str] | None = None,
) -> pd.DataFrame:
    """Replace infrequent property types with ``"Other"``."""
    df = df.copy()
    keep = keep or PROPERTY_TYPES_TO_KEEP
    df.loc[~df["property_type"].isin(keep), "property_type"] = "Other"
    return df


# ==========================================================================
# One-Hot Encoding (room type, property type, neighbourhood)
# ==========================================================================
def one_hot_encode_categoricals(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """One-hot encode categorical columns and clean resulting column names."""
    df = df.copy()
    columns = columns or ["room_type", "property_type", "neighbourhood_cleansed"]
    existing = [c for c in columns if c in df.columns]
    df = pd.get_dummies(df, columns=existing)
    # Clean column names: replace spaces and slashes with underscores
    df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]
    return df


# ==========================================================================
# Date-Derived Features
# ==========================================================================
def compute_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hosting_duration_days and joining_to_first_review_duration."""
    df = df.copy()
    for col in ["first_review", "last_review", "host_since"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "first_review" in df.columns and "last_review" in df.columns:
        df["hosting_duration_days"] = (
            df["last_review"] - df["first_review"]
        ).dt.days

    if "first_review" in df.columns and "host_since" in df.columns:
        df["joining_to_first_review_duration"] = (
            df["first_review"] - df["host_since"]
        ).dt.days

    df = df.drop(
        columns=["first_review", "last_review", "host_since"],
        errors="ignore",
    )
    return df


# ==========================================================================
# City-Centre Distance (Haversine)
# ==========================================================================
def _haversine(coord1: tuple, coord2: tuple) -> float:
    """Haversine distance in km between two (lat, lon) pairs."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_city_center_distance(
    df: pd.DataFrame,
    city: str = DEFAULT_CITY,
) -> pd.DataFrame:
    """Add ``distance_to_city_center`` (km) and drop lat/lon."""
    df = df.copy()
    center = CITY_CENTERS[city]
    df["distance_to_city_center"] = df.apply(
        lambda row: _haversine((row["latitude"], row["longitude"]), center),
        axis=1,
    )
    df = df.drop(columns=["latitude", "longitude"], errors="ignore")
    return df


# ==========================================================================
# Log-Transform Price
# ==========================================================================
def log_transform_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``log_price`` column and drop the original ``price``."""
    df = df.copy()
    df["log_price"] = np.log(df["price"])
    df = df.drop(columns=["price"])
    return df


# ==========================================================================
# Drop Final Text / URL Columns
# ==========================================================================
def drop_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop free-text and URL columns that are not needed for modelling."""
    cols = [c for c in TEXT_COLUMNS_TO_DROP_FINAL if c in df.columns]
    return df.drop(columns=cols)


# ==========================================================================
# Master Pipeline
# ==========================================================================
def engineer_features(
    listings: pd.DataFrame,
    city: str = DEFAULT_CITY,
    use_poi: bool = True,
) -> pd.DataFrame:
    """Run the full feature-engineering pipeline.

    Steps:
        1. Drop URL / redundant columns
        2. Filter valid listings
        3. Create amenity dummy variables
        4. Encode host verifications and response time
        5. Group rare property types
        6. One-hot encode categoricals
        7. Log-transform price
        8. Compute date-derived features
        9. Compute POI proximity features (Overpass API, cached)
        10. Compute city-centre distance (drops lat/lon)
        11. Drop remaining text columns
        12. Clean percentage columns

    Parameters
    ----------
    listings : pd.DataFrame
        Cleaned listings DataFrame (output of ``clean_listings``).
    city : str
        City name, used for POI queries and distance calculation.
    use_poi : bool
        If True (default), compute POI proximity features via Overpass.
        Set to False to skip POI features (faster, no API calls).
    """
    from .data_cleaning import (
        clean_percentage_columns,
        filter_valid_listings,
    )

    df = listings.copy()

    # Remove URL / media columns
    cols_to_drop = [c for c in URL_AND_REDUNDANT_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df = filter_valid_listings(df)
    df = create_amenity_dummies(df)
    df = encode_host_verifications(df)
    df = encode_host_response_time(df)
    df = group_rare_property_types(df)
    df = one_hot_encode_categoricals(df)
    df = log_transform_price(df)
    df = compute_date_features(df)

    # POI proximity (needs lat/lon — must come before city-centre distance)
    if use_poi:
        from .poi import compute_poi_features
        df = compute_poi_features(df, city=city)

    df = compute_city_center_distance(df, city=city)
    df = drop_text_columns(df)
    df = clean_percentage_columns(df)

    # Safety net: fill any remaining NaN with 0 for numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df.reset_index(drop=True)
