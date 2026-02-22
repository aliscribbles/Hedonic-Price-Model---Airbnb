# Airbnb Hedonic Price Model — London

A hedonic price model identifying price determinants for Airbnb listings in London, built as part of a Business Analytics & Data Science dissertation.

## Abstract

This study identifies the price determinants and develops a hedonic price model with a prime focus on the influence **amenities**, **reviews**, **location proximity**, **POI proximity**, and **neighbourhoods** have on Airbnb pricing. The analysis covers **75,000+ listings** in London and their associated **1,352,432 reviews** (processed with VADER sentiment analysis) to create **147 explanatory variables**. Prices are predicted using six regression models:

| Model | Description |
|---|---|
| **MLR** | Multiple Linear Regression (baseline) |
| **Ridge** | L2-regularised regression |
| **Lasso** | L1-regularised regression (feature selection) |
| **Random Forest** | Ensemble of decision trees |
| **XGBoost** | Gradient-boosted trees |
| **CatBoost** | Open-source ensemble (best performer) |

> **Key Finding:** CatBoost demonstrated the strongest performance, explaining the highest amount of variance with the smallest MSE. Listing characteristics, certain amenities, and proximity to the city centre had a positive high-to-moderate effect on Airbnb price.

## Methodology

```
Raw Data ──► Data Cleaning ──► Feature Engineering ──► Modelling ──► Evaluation
  │               │                    │                    │             │
  │          Price parsing        VADER Sentiment       6 Models     Train/Test
  │          NaN handling         Amenity Dummies       Grid Search   + K-Fold CV
  │          T/F encoding         Haversine Distance
  │          Filtering            One-Hot Encoding
  │                               Property Grouping
  │                               Host Features
  │                               POI Proximity *
  └── Inside Airbnb (London)
```

\* **POI Proximity**: Listing latitude/longitude is used to query the OpenStreetMap Overpass API for nearby Points of Interest (bus stops, hospitals, tourist attractions, supermarkets) within a 650m radius. For each POI type, the model receives both a **count** and a **nearest-distance** feature.

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── src/                          # Reusable Python modules
│   ├── config.py                 # Paths, constants, hyperparameters
│   ├── data_loader.py            # Load listings & reviews CSVs
│   ├── data_cleaning.py          # Price parsing, NaN handling
│   ├── feature_engineering.py    # Amenities, VADER, distance, encoding
│   ├── poi.py                    # POI proximity via Overpass API
│   ├── modeling.py               # 6 regression models + evaluation
│   └── visualization.py          # EDA & model comparison plots
├── notebooks/                    # Clean Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_ablation_study.ipynb
├── cache/                        # Auto-generated POI cache (gitignored)
└── Data Files/                   # NOT committed (see Data section below)
    └── London/
```

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/aliscribbles/hedonic-price-model-airbnb.git
cd hedonic-price-model-airbnb

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (required for VADER sentiment analysis)
python -c "import nltk; nltk.download('vader_lexicon')"
```

> **Note:** The first time you run the feature engineering pipeline, POI data is automatically fetched from the OpenStreetMap Overpass API and cached locally in `cache/`. This takes ~2 minutes per city and only happens once.

## Data

The data is sourced from [Inside Airbnb](http://insideairbnb.com/get-the-data/) and is **not included** in this repository due to file size. To reproduce the analysis:

1. Visit [Inside Airbnb — London](http://insideairbnb.com/get-the-data/)
2. Download the **detailed listings** (`listings.csv.gz`) and **reviews** (`reviews.csv.gz`)
3. Place them in `Data Files/London/`

The pipeline supports 10 cities (Amsterdam, Barcelona, Berlin, Edinburgh, Istanbul, London, Los Angeles, New York, Paris, Rome). Change `DEFAULT_CITY` in `src/config.py` to switch.

## Usage

The analysis is split across four notebooks in `notebooks/`:

| Notebook | Description |
|---|---|
| `01_data_exploration.ipynb` | Dataset overview, missing values, price distributions, geographic analysis |
| `02_feature_engineering.ipynb` | VADER sentiment, amenity extraction, location features, encoding |
| `03_modeling.ipynb` | Train all 6 models, evaluate, compare, cross-validate |
| `04_ablation_study.ipynb` | Measure incremental impact of each feature group |

All shared logic lives in `src/` and is imported by the notebooks:

```python
from src.data_loader import load_listings, load_reviews
from src.data_cleaning import clean_listings
from src.feature_engineering import engineer_features
from src.modeling import train_catboost, evaluate_model
from src.visualization import plot_model_comparison
```

## Tech Stack

- **Python 3.10+**
- **pandas** / **NumPy** — data manipulation
- **scikit-learn** — preprocessing, linear models, Random Forest, evaluation
- **XGBoost** / **CatBoost** — gradient boosting
- **NLTK (VADER)** — sentiment analysis
- **requests** — Overpass API queries for POI data
- **Matplotlib** / **Seaborn** / **Plotly** — visualisation
- **FuzzyWuzzy** — amenity string matching

## License

The source code in this repository is licensed under the [Apache License 2.0](LICENSE).

> **Please note:** This project is for academic and portfolio purposes. The datasets used in this analysis are sourced from and owned by [Inside Airbnb](http://insideairbnb.com/) and are subject to their specific terms of use.


