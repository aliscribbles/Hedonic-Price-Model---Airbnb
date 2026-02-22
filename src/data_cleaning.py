"""
Data Cleaning — price parsing, NaN handling, column dropping, T/F encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    COLUMNS_TO_DROP_INITIAL,
    URL_AND_REDUNDANT_COLUMNS,
    REDUNDANT_TEXT_COLUMNS,
    TF_COLUMNS,
)

# Columns that are always 100% NaN in recent Inside Airbnb scrapes
_ALWAYS_EMPTY = ["calendar_updated"]


# --------------------------------------------------------------------------
# Price cleaning
# --------------------------------------------------------------------------
def clean_prices(df: pd.DataFrame, col: str = "price") -> pd.DataFrame:
    """Parse ``$1,234.00``-style price strings into floats."""
    df = df.copy()
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    return df


# --------------------------------------------------------------------------
# Boolean / T-F columns
# --------------------------------------------------------------------------
def convert_tf_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convert ``t`` / ``f`` string values to ``1`` / ``0``."""
    df = df.copy()
    columns = columns or TF_COLUMNS
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda s: 1 if s == "t" else 0)
    return df


# --------------------------------------------------------------------------
# Percentage columns
# --------------------------------------------------------------------------
def clean_percentage_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Strip trailing ``%`` and convert to float.  NaN values are filled with 0."""
    df = df.copy()
    columns = columns or ["host_response_rate", "host_acceptance_rate"]
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.rstrip("%")
                .replace("nan", np.nan)
                .astype(float)
                .fillna(0)
            )
    return df


# --------------------------------------------------------------------------
# Drop helpers
# --------------------------------------------------------------------------
def drop_initial_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely NaN in raw listings (e.g. license)."""
    cols = [c for c in COLUMNS_TO_DROP_INITIAL + _ALWAYS_EMPTY if c in df.columns]
    return df.drop(columns=cols)


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove URL columns, host media, and free-text fields."""
    cols = [
        c
        for c in URL_AND_REDUNDANT_COLUMNS + REDUNDANT_TEXT_COLUMNS
        if c in df.columns
    ]
    return df.drop(columns=cols)


# --------------------------------------------------------------------------
# Filter valid listings
# --------------------------------------------------------------------------
def filter_valid_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only listings with positive price, bedrooms, accommodates, etc."""
    df = df.copy()
    for col in ["price", "accommodates"]:
        if col in df.columns:
            df = df[df[col] > 0]
    return df


# --------------------------------------------------------------------------
# Datetime conversion
# --------------------------------------------------------------------------
def convert_date_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convert string date columns to datetime."""
    df = df.copy()
    columns = columns or ["first_review", "last_review", "host_since"]
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# --------------------------------------------------------------------------
# Master pipeline
# --------------------------------------------------------------------------

# Columns that are critical for modelling — rows missing these are dropped
_REQUIRED_COLUMNS = [
    "price", "accommodates", "room_type", "property_type",
    "neighbourhood_cleansed", "latitude", "longitude",
    "host_since", "amenities",
]

# Numeric columns where NaN can safely be filled with 0
_FILL_ZERO_COLUMNS = [
    "bedrooms", "beds", "host_response_rate", "host_acceptance_rate",
    "review_scores_rating", "review_scores_accuracy",
    "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location",
    "review_scores_value", "reviews_per_month", "number_of_reviews",
    "host_total_listings_count", "host_listings_count",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
]


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on raw listings data.

    Steps:
        1. Drop sparse / entirely-NaN columns (license, calendar_updated, etc.)
        2. Drop redundant text / URL columns (host_about, host_url, etc.)
        3. Parse prices to float
        4. Clean percentage columns (fill NaN → 0)
        5. Convert date columns
        6. Convert T/F columns to 1/0
        7. Fill numeric NaN with 0 for safe columns (bedrooms, review scores)
        8. Drop rows missing *critical* columns only (price, location, etc.)
        9. Filter invalid listings (price ≤ 0, accommodates ≤ 0)
    """
    df = drop_initial_empty_columns(df)
    df = drop_redundant_columns(df)
    df = clean_prices(df)
    df = clean_percentage_columns(df)
    df = convert_date_columns(df)
    df = convert_tf_columns(df)

    # Fill safe numeric columns with 0
    fill_cols = [c for c in _FILL_ZERO_COLUMNS if c in df.columns]
    df[fill_cols] = df[fill_cols].fillna(0)

    # Drop only rows missing critical modelling columns
    req_cols = [c for c in _REQUIRED_COLUMNS if c in df.columns]
    df = df.dropna(subset=req_cols)

    df = filter_valid_listings(df)
    return df.reset_index(drop=True)
