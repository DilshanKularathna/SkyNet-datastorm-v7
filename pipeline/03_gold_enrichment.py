"""
=============================================================================
Pipeline Step 03 — Gold Enrichment & Feature Engineering
=============================================================================
Joins all silver datasets, engineers features, merges POI data.
Output: data/gold/outlet_features.csv  +  data/gold/model_input_final.csv
=============================================================================
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import variation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

SILVER_DIR  = ROOT / "data" / "silver"
GOLD_DIR    = ROOT / "data" / "gold"
EXTERNAL    = ROOT / "data" / "external"
GOLD_DIR.mkdir(parents=True, exist_ok=True)

# January 2026 is the prediction month
PRED_YEAR, PRED_MONTH = 2026, 1

# Seasonality index numeric mapping (for modelling)
SEASON_MAP = {"Favorable": 1.15, "Moderate": 1.00, "Un-Favorable": 0.85}


# =============================================================================
# 1. Load silver datasets
# =============================================================================
def load_silver():
    log.info("Loading silver datasets...")
    tx       = pd.read_csv(SILVER_DIR / "transactions_clean.csv",  low_memory=False)
    master   = pd.read_csv(SILVER_DIR / "outlet_master_clean.csv", low_memory=False)
    coords   = pd.read_csv(SILVER_DIR / "coordinates_clean.csv",   low_memory=False)
    season   = pd.read_csv(SILVER_DIR / "seasonality_clean.csv",   low_memory=False)
    holidays = pd.read_csv(SILVER_DIR / "holidays_clean.csv",      low_memory=False,
                           parse_dates=["Date"])

    for col in ["Year", "Month", "Volume_Liters", "Total_Bill_Value"]:
        if col in tx.columns:
            tx[col] = pd.to_numeric(tx[col], errors="coerce")

    log.info("  tx=%d  master=%d  coords=%d  season=%d  holidays=%d",
             len(tx), len(master), len(coords), len(season), len(holidays))
    return tx, master, coords, season, holidays


# =============================================================================
# 2. Monthly aggregate (outlet × year × month)
# =============================================================================
def build_monthly_agg(tx):
    log.info("Building monthly outlet aggregates...")
    agg = (
        tx.groupby(["Outlet_ID", "Year", "Month", "Distributor_ID"])
        .agg(
            monthly_volume=("Volume_Liters",    "sum"),
            monthly_value =("Total_Bill_Value", "sum"),
            sku_count      =("SKU_ID",           "nunique"),
            order_lines    =("SKU_ID",           "count"),
        )
        .reset_index()
    )
    log.info("  Monthly agg rows: %d", len(agg))
    return agg


# =============================================================================
# 3. Outlet-level behavioural features
# =============================================================================
def build_outlet_features(monthly_agg):
    log.info("Engineering outlet-level features...")

    feats = {}

    grouped = monthly_agg.groupby("Outlet_ID")

    feats["mean_monthly_volume"]  = grouped["monthly_volume"].mean()
    feats["median_monthly_volume"]= grouped["monthly_volume"].median()
    feats["max_monthly_volume"]   = grouped["monthly_volume"].max()
    feats["std_monthly_volume"]   = grouped["monthly_volume"].std().fillna(0)
    feats["cv_volume"]            = (
        feats["std_monthly_volume"] / (feats["mean_monthly_volume"] + 1e-9)
    )
    feats["months_active"]        = grouped["monthly_volume"].count()
    feats["total_volume"]         = grouped["monthly_volume"].sum()
    feats["total_value"]          = grouped["monthly_value"].sum()

    # Zero-order months: months where volume == 0
    zero_months = (
        monthly_agg[monthly_agg["monthly_volume"] == 0]
        .groupby("Outlet_ID")["monthly_volume"].count()
    )
    feats["zero_order_months"] = zero_months

    # Order frequency: proportion of months with positive volume
    total_months_in_dataset = monthly_agg["Year"].nunique() * 12
    feats["order_frequency"] = feats["months_active"] / total_months_in_dataset

    # Peak-to-average ratio (high => constrained most of the time)
    feats["peak_to_avg_ratio"] = (
        feats["max_monthly_volume"] / (feats["mean_monthly_volume"] + 1e-9)
    )

    # Dominant Distributor_ID per outlet
    dom_dist = (
        monthly_agg.groupby(["Outlet_ID", "Distributor_ID"])["monthly_volume"]
        .sum()
        .reset_index()
        .sort_values("monthly_volume", ascending=False)
        .drop_duplicates("Outlet_ID")
        .set_index("Outlet_ID")["Distributor_ID"]
    )
    feats["dominant_distributor"] = dom_dist

    # ── Constraint signals ────────────────────────────────────────────────────
    # Credit-limit hit rate: % months where volume >= 0.95 × outlet max
    # (proxy for hitting a credit ceiling)
    max_vol = grouped["monthly_volume"].max()
    def credit_hit_rate(grp):
        outlet_max = grp["monthly_volume"].max()
        if outlet_max == 0:
            return 0.0
        return (grp["monthly_volume"] >= 0.95 * outlet_max).mean()

    feats["credit_limit_hit_rate"] = grouped.apply(credit_hit_rate)

    # Stockout flag: outlet had at least one zero-volume month mid-series
    def stockout_flag(grp):
        sorted_grp = grp.sort_values(["Year", "Month"])
        vols = sorted_grp["monthly_volume"].values
        if len(vols) < 3:
            return 0
        # Zero sandwiched between non-zeros
        for i in range(1, len(vols) - 1):
            if vols[i] == 0 and vols[i - 1] > 0 and vols[i + 1] > 0:
                return 1
        return 0

    feats["stockout_flag"] = grouped.apply(stockout_flag)

    # Trend: simple slope from linear regression over monthly volumes
    def volume_trend(grp):
        if len(grp) < 3:
            return 0.0
        x = np.arange(len(grp))
        y = grp.sort_values(["Year", "Month"])["monthly_volume"].values
        if y.std() == 0:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return slope

    feats["volume_trend"] = grouped.apply(volume_trend)

    feature_df = pd.DataFrame(feats).reset_index()
    feature_df = feature_df.fillna(0)
    log.info("  Outlet feature matrix: %d rows × %d cols", *feature_df.shape)
    return feature_df


# =============================================================================
# 4. Seasonality lookup for January 2026
# =============================================================================
def build_seasonality_lookup(season_df):
    log.info("Building January 2026 seasonality lookup...")
    season_df["season_numeric"] = season_df["Seasonality_Index"].map(SEASON_MAP)

    # January 2026 seasonality per distributor
    jan26 = season_df[
        (season_df["Year"] == PRED_YEAR) & (season_df["Month"] == PRED_MONTH)
    ][["Distributor_ID", "season_numeric"]].copy()
    jan26.columns = ["dominant_distributor", "jan26_seasonality_index"]

    # Fallback: use January across all known years
    if jan26.empty:
        log.warning("  No Jan 2026 seasonality — falling back to Jan average across years.")
        jan26 = (
            season_df[season_df["Month"] == PRED_MONTH]
            .groupby("Distributor_ID")["season_numeric"]
            .mean()
            .reset_index()
        )
        jan26.columns = ["dominant_distributor", "jan26_seasonality_index"]

    log.info("  Seasonality lookup: %d distributors", len(jan26))
    return jan26


# =============================================================================
# 5. Holiday density feature for January 2026
# =============================================================================
def compute_holiday_density(holidays_df):
    jan_holidays = holidays_df[
        (holidays_df["Date"].dt.year  == PRED_YEAR) &
        (holidays_df["Date"].dt.month == PRED_MONTH)
    ]
    poya_count    = (jan_holidays["Holiday_Type"] == "Poya Day").sum()
    public_count  = (jan_holidays["Holiday_Type"] == "Public").sum()
    total_holidays = len(jan_holidays)
    log.info(
        "  Jan %d holidays: total=%d  poya=%d  public=%d",
        PRED_YEAR, total_holidays, poya_count, public_count,
    )
    return {
        "jan_total_holidays": total_holidays,
        "jan_poya_days":      poya_count,
        "jan_public_holidays": public_count,
    }


# =============================================================================
# 6. Merge everything into Gold
# =============================================================================
def build_gold(outlet_features, master_df, coords_df, jan_season_lkp, holiday_meta, poi_df=None):
    log.info("Building Gold layer...")

    gold = outlet_features.copy()

    # Merge outlet master (outlet size, type, cooler count)
    gold = gold.merge(
        master_df[["Outlet_ID", "Outlet_Size", "Outlet_Type", "Cooler_Count"]],
        on="Outlet_ID", how="left",
    )

    # Merge coordinates
    gold = gold.merge(
        coords_df[["Outlet_ID", "Latitude", "Longitude"]],
        on="Outlet_ID", how="left",
    )

    # Merge January 2026 seasonality
    gold = gold.merge(jan_season_lkp, on="dominant_distributor", how="left")
    gold["jan26_seasonality_index"] = gold["jan26_seasonality_index"].fillna(1.0)

    # Add global holiday metadata as columns
    for k, v in holiday_meta.items():
        gold[k] = v

    # Merge POI data if available
    if poi_df is not None and not poi_df.empty:
        log.info("  Merging POI data...")
        gold = gold.merge(poi_df, on="Outlet_ID", how="left")
        poi_cols = [c for c in poi_df.columns if c != "Outlet_ID"]
        gold[poi_cols] = gold[poi_cols].fillna(0)

    # ── One-hot encode categoricals ────────────────────────────────────────────
    gold = pd.get_dummies(gold, columns=["Outlet_Size", "Outlet_Type"], drop_first=False)

    # ── Outlet size ordinal ────────────────────────────────────────────────────
    size_ord = {"Small": 1, "Medium": 2, "Large": 3, "Extra Large": 4}
    if "Outlet_Size" in master_df.columns:
        gold["outlet_size_ord"] = (
            master_df.set_index("Outlet_ID")["Outlet_Size"]
            .map(size_ord)
            .reindex(gold["Outlet_ID"])
            .values
        )

    log.info("  Gold shape: %d rows × %d cols", *gold.shape)
    return gold


# =============================================================================
# Main
# =============================================================================
def run_gold_enrichment():
    log.info("=" * 70)
    log.info("GOLD ENRICHMENT START — %s", datetime.now().isoformat())
    log.info("=" * 70)

    tx, master, coords, season, holidays = load_silver()

    monthly_agg     = build_monthly_agg(tx)
    outlet_features = build_outlet_features(monthly_agg)
    jan_season      = build_seasonality_lookup(season)
    holiday_meta    = compute_holiday_density(holidays)

    # Load POI data if already scraped
    poi_path = EXTERNAL / "poi_features.csv"
    poi_df   = pd.read_csv(poi_path, low_memory=False) if poi_path.exists() else None
    if poi_df is None:
        log.warning("POI features not found — run 04_poi_scraping.py first for best results.")

    gold = build_gold(outlet_features, master, coords, jan_season, holiday_meta, poi_df)

    # ── Save outputs ──────────────────────────────────────────────────────────
    feat_path  = GOLD_DIR / "outlet_features.csv"
    model_path = GOLD_DIR / "model_input_final.csv"
    gold.to_csv(feat_path,  index=False)
    gold.to_csv(model_path, index=False)

    log.info("Saved outlet_features.csv  → %s", feat_path)
    log.info("Saved model_input_final.csv → %s", model_path)
    log.info("GOLD ENRICHMENT COMPLETE")
    return gold


if __name__ == "__main__":
    run_gold_enrichment()
