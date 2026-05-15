# SkyNet — Data Storm v7.0 Winning Strategy

This repository contains the complete data engineering and modeling pipeline for the Data Storm v7.0 competition. Our approach focuses on **Latent Demand Estimation** using a left-censored demand uncaping methodology.

## 🚀 Execution Flow (End-to-End)

To achieve the best results, follow this execution order:

### Phase 1: Data Engineering Pipeline
Run these scripts sequentially from the `pipeline/` directory:

1.  **`01_bronze_ingestion.py`**: Ingests raw CSVs from `data/bronze/` with zero transformations.
2.  **`02_silver_cleaning.py`**: Runs 12+ rigorous DQ checks. Clean data goes to `data/silver/`, while anomalies are quarantined in `data/rejected/` with failure reasons.
3.  **`04_poi_scraping.py`**: (Recommended) Scrapes Point-of-Interest data from OpenStreetMap (Overpass API) to enrich the feature set.
4.  **`03_gold_enrichment.py`**: Joins cleaned datasets, engineers transaction-based features, constraint signals, and merges POI data.

### Phase 2: Modeling & Estimation
Open and run the Jupyter notebooks in the `modeling/` directory:

1.  **`01_eda.ipynb`**: Exploratory analysis of volumes, trends, and geographic distributions.
2.  **`02_constraint_analysis.ipynb`**: Classifies every outlet into 4 constraint archetypes (Credit-hit, Stockout, Delivery-capped, Low-demand).
3.  **`03_potential_model.ipynb`**: Core potential estimation. Uses XGBoost with constraint-based uplift factors and peer-group 90th percentile benchmarks.
4.  **`04_final_predictions.ipynb`**: Generates the final submission CSV for January 2026.

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

*Note: All logic was manually reviewed, tested, and calibrated against Sri Lankan FMCG market dynamics.*