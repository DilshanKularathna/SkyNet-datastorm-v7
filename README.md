# SkyNet — Data Storm v7.0

This repository contains the complete data engineering and modeling pipeline for the Data Storm v7.0 competition. Our approach focuses on **Latent Demand Estimation** using a left-censored demand uncaping methodology.

## 🚀 Execution Flow (End-to-End)

### Environment Setup
1. Ensure all Python dependencies are installed (`pandas`, `numpy`, `xgboost`, `geopandas`, `pyogrio`, etc.).
2. Ensure you have the local Sri Lanka OSM PBF file at `data/external/sri-lanka-latest.osm.pbf` for POI scraping.

### Phase 1: Data Engineering Pipeline
Run these Python scripts sequentially from the project root or the `pipeline/` directory:

1.  **`python pipeline/01_bronze_ingestion.py`**
    - Ingests raw CSVs from `data/bronze/` with zero transformations.
2.  **`python pipeline/02_silver_cleaning.py`**
    - Runs rigorous DQ checks. Discovers allowed values dynamically to prevent valid data rejection. Clean data goes to `data/silver/`, while anomalies are quarantined in `data/rejected/`.
3.  **`python pipeline/04_poi_scraping.py`**
    - Extracts Point-of-Interest (POI) data locally from the OSM PBF file (schools, hospitals, urban scoring) for fast processing (no API rate limits).
4.  **`python pipeline/03_gold_enrichment.py`**
    - Joins cleaned datasets, engineers features (January baselines, price efficiency, SKU growth), classifies constraint archetypes, and computes peer-group benchmarks. Outputs `model_input_final.csv`.

### Phase 2: Modeling & Estimation
Open and run the Jupyter notebooks in the `modeling/` directory in this exact order:

1.  **`01_eda.ipynb`**: Exploratory analysis of volumes, trends, and geographic distributions.
2.  **`02_constraint_analysis.ipynb`**: Validates constraint classifications, visualizes volume profiles by type, analyzes peer benchmarks, and saves `outlet_features_with_constraints.csv`.
3.  **`03_potential_model.ipynb`**: Core potential estimation implementing the **Two-Stage Uncapping Framework**:
    - *Stage 1*: XGBoost predicts `max_monthly_volume` as a proxy lower bound.
    - *Stage 2*: Applies `constraint_multiplier` and `jan26_seasonality_index` as post-prediction multipliers. Includes protective bounds (never predict below observed max, capped at 2x peer 90th percentile).
4.  **`04_final_predictions.ipynb`**: Generates the final submission CSV (`outputs/SkyNet_predictions.csv`) covering all outlets with the target column `Maximum_Monthly_Liters`.

## 🛠 Project Structure

```
datastorm-v7/
├── data/
│   ├── bronze/     # Raw input files
│   ├── silver/     # Cleaned, validated datasets
│   ├── gold/       # Feature-engineered model inputs
│   ├── rejected/   # Quarantine store (critical for DQ points)
│   └── external/   # Scraped POI features
├── pipeline/       # Automation scripts
├── dq_checks/      # Reusable quality functions
├── modeling/       # Analysis and model development
├── outputs/        # Final submission files
└── reports/        # Generated charts and analysis
```

## 💎 Key Methodologies

*   **DQ Forensics**: Automated detection of "Ghost Transactions", coordinate anomalies, and implausible volume spikes.
*   **Constraint Uncapping**: Classification-based multipliers to estimate true demand potential from observed sales.
*   **Geospatial Footfall**: Integration of OpenStreetMap data (Schools, Bus Stops, Hospitals) to model impulse demand.
*   **Seasonality Calibration**: Distributor-specific January seasonality index application for 2026 forecasting.

## ⚖️ GenAI Transparency
Used for:
- Drafting boilerplate DQ function templates.
- Optimizing Overpass API query logic for OSM scraping.
- Debugging Tobit-style regression implementations.

