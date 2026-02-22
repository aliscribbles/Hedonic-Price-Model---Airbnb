"""
Configuration — paths, constants, random seeds, and amenity keyword mappings.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data Files"

# ---------------------------------------------------------------------------
# City selection  —  change DEFAULT_CITY to switch the entire pipeline
# ---------------------------------------------------------------------------
AVAILABLE_CITIES = [
    "Amsterdam", "Barcelona", "Berlin", "Edinburgh", "Istanbul",
    "London", "Los Angeles", "Newyork", "Paris", "Rome",
]

DEFAULT_CITY = "London"  # ← change this one value to analyse a different city

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
KFOLD_SPLITS = 10

# ---------------------------------------------------------------------------
# City-centre coordinates  (used for Haversine distance feature)
# ---------------------------------------------------------------------------
CITY_CENTERS = {
    "Amsterdam":   (52.3676, 4.9041),
    "Barcelona":   (41.3874, 2.1686),
    "Berlin":      (52.5200, 13.4050),
    "Edinburgh":   (55.9533, -3.1883),
    "Istanbul":    (41.0082, 28.9784),
    "London":      (51.5074, -0.1278),
    "Los Angeles": (34.0522, -118.2437),
    "Newyork":     (40.7128, -74.0060),
    "Paris":       (48.8566, 2.3522),
    "Rome":        (41.9028, 12.4964),
}

# ---------------------------------------------------------------------------
# POI Proximity Features  (Overpass API)
# ---------------------------------------------------------------------------
CACHE_DIR = PROJECT_ROOT / "cache"
POI_RADIUS_M = 650  # search radius in metres

# OSM tag filters for each POI category
POI_CATEGORIES: dict[str, str] = {
    "bus_stop":             '["highway"="bus_stop"]',
    "hospital":             '["amenity"="hospital"]',
    "tourist_attraction":   '["tourism"~"attraction|museum|artwork|gallery"]',
    "supermarket":          '["shop"="supermarket"]',
}

# ---------------------------------------------------------------------------
# Columns to drop at various stages
# ---------------------------------------------------------------------------
COLUMNS_TO_DROP_INITIAL = [
    "license", "neighbourhood_group_cleansed", "bathrooms",
]

URL_AND_REDUNDANT_COLUMNS = [
    "calendar_last_scraped", "host_picture_url", "host_thumbnail_url",
    "host_url", "host_id", "picture_url", "listing_url", "source",
]

TEXT_COLUMNS_TO_DROP_FINAL = [
    "name", "description", "host_name", "bathrooms_text",
    "amenities", "host_location",
]

REDUNDANT_TEXT_COLUMNS = [
    "host_about", "neighborhood_overview", "neighbourhood",
    "host_neighbourhood", "last_scraped",
]

# ---------------------------------------------------------------------------
# True/False columns to encode as 1/0
# ---------------------------------------------------------------------------
TF_COLUMNS = [
    "instant_bookable", "host_is_superhost", "host_has_profile_pic",
    "host_identity_verified", "has_availability",
]

# ---------------------------------------------------------------------------
# Property types to keep (all others → "Other")
# ---------------------------------------------------------------------------
PROPERTY_TYPES_TO_KEEP = [
    "Entire rental unit", "Entire condo", "Private room in home",
    "Private room in rental unit", "Entire home",
    "Private room in condo", "Private room in townhouse",
    "Entire serviced apartment", "Entire townhouse",
    "Private room in bed and breakfast", "Entire loft",
    "Room in hotel", "Private room in guesthouse",
    "Entire guest suite", "Entire guesthouse",
    "Room in boutique hotel", "Room in aparthotel",
]

# ---------------------------------------------------------------------------
# Amenity keyword dictionaries
# Each key is the dummy column name; values are keyword lists.
# A listing gets a 1 if *any* keyword appears in its amenities text.
# ---------------------------------------------------------------------------
AMENITY_KEYWORDS: dict[str, list[str]] = {
    "cooking_range": [
        "stove", "oven", "hob", "cooker", "range",
        "induction", "gas stove", "electric stove",
    ],
    "sound_system": [
        "sound system", "Bluetooth speaker", "speaker", "Sonos",
        "Bose", "Bang & Olufsen", "Echo dot",
    ],
    "toiletries": [
        "shampoo", "conditioner", "body wash", "soap",
        "shower gel", "toiletries",
    ],
    "tv_streaming": [
        "HDTV", "TV", "Netflix", "Amazon Prime", "Disney+",
        "Apple TV", "Chromecast", "Roku",
    ],
    "coffee_machine": [
        "coffee maker", "Nespresso", "espresso", "Keurig",
        "French press", "coffee machine",
    ],
    "pool_hot_tub": [
        "pool", "hot tub", "jacuzzi", "plunge pool",
    ],
    "sauna": [
        "sauna",
    ],
    "fridge_freezer": [
        "refrigerator", "fridge", "freezer",
    ],
    "high_chair": [
        "high chair", "booster seat",
    ],
    "microwave": [
        "microwave",
    ],
    "internet": [
        "wifi", "Wifi", "internet", "ethernet",
    ],
    "air_conditioning": [
        "AC", "air conditioning", "Window AC",
        "Central air", "portable air",
    ],
    "laundry": [
        "washer", "dryer", "laundry", "washing machine",
    ],
    "white_goods": [
        "kettle", "dishwasher", "hair dryer", "iron",
        "toaster", "blender",
    ],
    "children_friendly": [
        "children's", "crib", "baby", "child",
        "pack 'n play", "baby bath",
    ],
    "clothes_storage": [
        "closet", "wardrobe", "dresser",
        "clothing storage", "hangers",
    ],
    "parking_space": [
        "parking", "garage", "carport",
    ],
    "self_check_in": [
        "self check-in", "lockbox", "keypad", "smart lock",
    ],
    "luggage_dropoff": [
        "luggage dropoff",
    ],
    "pets_allowed": [
        "pets allowed",
    ],
    "smoking_allowed": [
        "smoking allowed",
    ],
    "bed_linens": [
        "bed linens",
    ],
    "views_and_scenery": [
        "view", "courtyard", "city skyline", "garden view",
        "lake view", "mountain view", "ocean view",
    ],
    "host_greets_you": [
        "host greets you",
    ],
    "extra_pillows_blankets": [
        "extra pillows", "blankets",
    ],
    "patio_balcony": [
        "patio", "balcony",
    ],
    "cooking_essentials": [
        "dishes", "silverware", "cooking basics",
        "pots", "pans", "baking sheet",
    ],
    "elevator": [
        "elevator",
    ],
    "heating": [
        "heating", "central heating", "radiant heating",
    ],
    "gaming_console": [
        "game console", "Xbox", "PlayStation", "Nintendo",
    ],
    "gym_access": [
        "gym", "exercise equipment", "free weights",
    ],
    "bbq_grill": [
        "BBQ", "grill", "fireplace", "outdoor dining",
    ],
    "safety_features": [
        "fire extinguisher", "first aid kit", "smoke alarm",
        "carbon monoxide alarm", "keypad", "security cameras",
    ],
}

# ---------------------------------------------------------------------------
# Model hyperparameter grids (used in grid search)
# ---------------------------------------------------------------------------
RF_PARAM_GRID = {
    "n_estimators": [2000],
    "max_features": [2, 5],
    "max_depth": [40, 70],
    "min_samples_split": [40, 50],
    "max_leaf_nodes": [50, 70],
}

RF_BEST_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 40,
    "min_samples_split": 40,
    "max_leaf_nodes": 70,
    "max_features": 5,
}

XGB_PARAM_GRID = {
    "colsample_bytree": [0.3, 0.5],
    "learning_rate": [0.1, 0.5],
    "alpha": [10, 12],
    "max_depth": [3, 5],
    "n_estimators": [2000],
}

XGB_BEST_PARAMS = {
    "objective": "reg:squarederror",
    "colsample_bytree": 0.5,
    "learning_rate": 0.1,
    "max_depth": 5,
    "alpha": 10,
    "n_estimators": 2000,
}

CATBOOST_PARAM_GRID = {
    "depth": [3, 5],
    "learning_rate": [0.1, 0.5],
    "iterations": [2000],
}

CATBOOST_BEST_PARAMS = {
    "iterations": 2000,
    "depth": 5,
    "learning_rate": 0.1,
    "loss_function": "RMSE",
}
