"""
=============================================================================
Pipeline Step 03 — Gold Enrichment & Feature Engineering
=============================================================================

Output:
  data/gold/outlet_features.csv      — all engineered features per outlet
  data/gold/model_input_final.csv    — same (ready for modeling notebook)
=============================================================================
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR   = ROOT / "data" / "gold"
EXTERNAL   = ROOT / "data" / "external"
GOLD_DIR.mkdir(parents=True, exist_ok=True)

PRED_YEAR, PRED_MONTH = 2026, 1

# Seasonality numeric mapping (Favorable = +15%, Un-Favorable = -15%)
SEASON_MAP = {"Favorable": 1.15, "Moderate": 1.00, "Un-Favorable": 0.85}

# Province derived from Distributor_ID
PROVINCE_MAP = {
    "DIST_W_01":  "Western",    "DIST_W_02":  "Western",    "DIST_W_03":  "Western",
    "DIST_C_01":  "Central",    "DIST_C_02":  "Central",    "DIST_C_03":  "Central",
    "DIST_NW_01": "NorthWest",  "DIST_NW_02": "NorthWest",
    "DIST_S_01":  "Southern",   "DIST_S_02":  "Southern",
}

# Outlet size ordinal (used before one-hot encoding)
SIZE_ORDINAL = {"Small": 1, "Medium": 2, "Large": 3, "Extra Large": 4}

# Constraint multipliers applied AFTER XGBoost base prediction
CONSTRAINT_MULTIPLIERS = {
    "Stockout_Constrained":    1.35,
    "Credit_Constrained":      1.25,
    "Delivery_Capped":         1.15,
    "Moderately_Constrained":  1.10,
    "Genuinely_Low_Demand":    1.00,
}


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
    for col in ["Year", "Month"]:
        if col in season.columns:
            season[col] = pd.to_numeric(season[col], errors="coerce")

    log.info("  tx=%d  master=%d  coords=%d  season=%d  holidays=%d",
             len(tx), len(master), len(coords), len(season), len(holidays))
    return tx, master, coords, season, holidays


# =============================================================================
# 2. Monthly aggregate  (outlet × year × month)
# =============================================================================
def build_monthly_agg(tx: pd.DataFrame) -> pd.DataFrame:
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
def build_outlet_features(monthly_agg: pd.DataFrame) -> pd.DataFrame:
    log.info("Engineering outlet-level behavioural features...")
    feats   = {}
    grouped = monthly_agg.groupby("Outlet_ID")

    feats["mean_monthly_volume"]   = grouped["monthly_volume"].mean()
    feats["median_monthly_volume"] = grouped["monthly_volume"].median()
    feats["max_monthly_volume"]    = grouped["monthly_volume"].max()
    feats["std_monthly_volume"]    = grouped["monthly_volume"].std().fillna(0)
    feats["cv_volume"]             = (
        feats["std_monthly_volume"] / (feats["mean_monthly_volume"] + 1e-9)
    )
    feats["months_active"]  = grouped["monthly_volume"].count()
    feats["total_volume"]   = grouped["monthly_volume"].sum()
    feats["total_value"]    = grouped["monthly_value"].sum()

    # Zero-order months (proxy for stockout / credit freeze periods)
    zero_months = (
        monthly_agg[monthly_agg["monthly_volume"] == 0]
        .groupby("Outlet_ID")["monthly_volume"].count()
    )
    feats["zero_order_months"] = zero_months.reindex(grouped.groups.keys(), fill_value=0)

    total_possible_months = monthly_agg["Year"].nunique() * 12
    feats["order_frequency"] = feats["months_active"] / total_possible_months

    # Peak-to-average ratio (high = constrained most months, spikes occasionally)
    feats["peak_to_avg_ratio"] = (
        feats["max_monthly_volume"] / (feats["mean_monthly_volume"] + 1e-9)
    )

    # Dominant distributor per outlet (by volume)
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

    # Credit-limit hit rate: % months where volume ≥ 95% of outlet max
    def credit_hit_rate(grp):
        outlet_max = grp["monthly_volume"].max()
        if outlet_max == 0:
            return 0.0
        return (grp["monthly_volume"] >= 0.95 * outlet_max).mean()

    feats["credit_limit_hit_rate"] = grouped.apply(credit_hit_rate)

    # Stockout flag (FIXED): only flag zero-volume months WITHIN the active window.
    # Gaps before the first sale or after the last sale are NOT flagged (new/exited outlets).
    def stockout_flag(grp):
        if len(grp) < 2:
            return 0
        sgrp = grp.sort_values(["Year", "Month"]).copy()
        sgrp["ym"] = sgrp["Year"] * 12 + sgrp["Month"]

        # Any explicit zero within the recorded window?
        if (sgrp["monthly_volume"] == 0).any():
            return 1

        # Gaps within the active window (first sale → last sale)?
        expected = sgrp["ym"].max() - sgrp["ym"].min() + 1
        if len(sgrp) < expected:
            return 1
        return 0

    feats["stockout_flag"] = grouped.apply(stockout_flag)

    # Volume trend (linear slope over time; positive = growing outlet)
    def volume_trend(grp):
        if len(grp) < 3:
            return 0.0
        y = grp.sort_values(["Year", "Month"])["monthly_volume"].values
        if y.std() == 0:
            return 0.0
        return np.polyfit(np.arange(len(y)), y, 1)[0]

    feats["volume_trend"] = grouped.apply(volume_trend)

    feature_df = pd.DataFrame(feats).reset_index().fillna(0)
    log.info("  Outlet feature matrix: %d rows × %d cols", *feature_df.shape)
    return feature_df


# =============================================================================
# 4. January-specific historical baseline  (NEW — critical missing feature)
# =============================================================================
def build_january_features(monthly_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Extract January-specific volume statistics from historical data.
    January 2023 and 2024 are the strongest predictors for January 2026.
    """
    log.info("Building January-specific historical features...")
    jan = monthly_agg[monthly_agg["Month"] == 1].copy()

    if jan.empty:
        log.warning("  No January records found in monthly_agg!")
        return pd.DataFrame(columns=["Outlet_ID"])

    jan_grouped = jan.groupby("Outlet_ID")

    jan_feats = pd.DataFrame({
        "jan_mean_volume":  jan_grouped["monthly_volume"].mean(),
        "jan_max_volume":   jan_grouped["monthly_volume"].max(),
        "jan_min_volume":   jan_grouped["monthly_volume"].min(),
        "jan_data_points":  jan_grouped["monthly_volume"].count(),
    }).reset_index()

    # Year-on-year January growth (2024 vs 2023, 2025 vs 2024)
    jan_pivot = jan.pivot_table(
        index="Outlet_ID", columns="Year", values="monthly_volume", aggfunc="sum"
    ).reset_index()

    years = sorted(jan["Year"].unique())
    yoy_cols = []
    for i in range(1, len(years)):
        prev_yr, curr_yr = years[i - 1], years[i]
        if prev_yr in jan_pivot.columns and curr_yr in jan_pivot.columns:
            col = f"jan_yoy_{prev_yr}_{curr_yr}"
            jan_pivot[col] = (
                (jan_pivot[curr_yr] - jan_pivot[prev_yr]) /
                (jan_pivot[prev_yr] + 1e-9)
            ).clip(-1, 5)  # clip extreme growth rates
            yoy_cols.append(col)

    if yoy_cols:
        jan_feats = jan_feats.merge(
            jan_pivot[["Outlet_ID"] + yoy_cols], on="Outlet_ID", how="left"
        )

    log.info("  January features: %d outlets, %d cols", len(jan_feats), len(jan_feats.columns))
    return jan_feats


# =============================================================================
# 5. SKU diversity trend  (NEW)
# =============================================================================
def build_sku_trend(monthly_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Measure how SKU breadth (product variety) is changing per outlet.
    Expanding SKU range = outlet is growing / developing.
    """
    log.info("Building SKU diversity trend features...")
    sku_by_year = (
        monthly_agg.groupby(["Outlet_ID", "Year"])["sku_count"]
        .mean()
        .reset_index()
        .rename(columns={"sku_count": "avg_sku_count"})
    )
    sku_pivot = sku_by_year.pivot(index="Outlet_ID", columns="Year",
                                   values="avg_sku_count").reset_index()

    years = sorted(monthly_agg["Year"].unique())
    sku_trend_df = sku_pivot[["Outlet_ID"]].copy()

    if len(years) >= 2:
        first_yr, last_yr = years[0], years[-1]
        if first_yr in sku_pivot.columns and last_yr in sku_pivot.columns:
            sku_trend_df["sku_growth"] = (
                (sku_pivot[last_yr] - sku_pivot[first_yr]) /
                (sku_pivot[first_yr] + 1e-9)
            ).clip(-1, 5)
        else:
            sku_trend_df["sku_growth"] = 0.0
    else:
        sku_trend_df["sku_growth"] = 0.0

    sku_trend_df["max_annual_sku_count"] = sku_pivot[
        [c for c in sku_pivot.columns if c != "Outlet_ID"]
    ].max(axis=1)

    log.info("  SKU trend features: %d outlets", len(sku_trend_df))
    return sku_trend_df


# =============================================================================
# 6. Seasonality lookup for January 2026
# =============================================================================
def build_seasonality_lookup(season_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Building January 2026 seasonality lookup...")
    season_df = season_df.copy()
    season_df["season_numeric"] = season_df["Seasonality_Index"].map(SEASON_MAP)

    jan26 = season_df[
        (season_df["Year"] == PRED_YEAR) & (season_df["Month"] == PRED_MONTH)
    ][["Distributor_ID", "season_numeric"]].copy()
    jan26.columns = ["dominant_distributor", "jan26_seasonality_index"]

    if jan26.empty:
        log.warning("  No Jan 2026 seasonality — averaging across all historical Januaries.")
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
# 7. Holiday density for January 2026
# =============================================================================
def compute_holiday_density(holidays_df: pd.DataFrame) -> dict:
    jan = holidays_df[
        (holidays_df["Date"].dt.year  == PRED_YEAR) &
        (holidays_df["Date"].dt.month == PRED_MONTH)
    ]
    poya   = (jan["Holiday_Type"] == "Poya Day").sum()
    public = (jan["Holiday_Type"] == "Public").sum()
    total  = len(jan)
    log.info("  Jan %d holidays: total=%d  poya=%d  public=%d",
             PRED_YEAR, total, poya, public)
    return {
        "jan_total_holidays":   int(total),
        "jan_poya_days":        int(poya),
        "jan_public_holidays":  int(public),
    }


# =============================================================================
# 8. Peer group 90th percentile benchmarking  (NEW — core uncapping mechanism)
# =============================================================================
def compute_peer_benchmarks(gold_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each outlet, find comparable peers (same Type × Province × Size) and
    set potential benchmarks at the 75th and 90th percentile of max monthly volume.

    The 90th percentile peer represents what a similar, unconstrained outlet sells.
    The uncapping_factor = how far below that ceiling the current outlet is.

    NOTE: Must be called BEFORE one-hot encoding (needs original categorical columns).
    """
    log.info("Computing peer group benchmarks...")
    peer_cols = ["Outlet_Type", "province", "Outlet_Size"]

    # Ensure all peer cols exist (fill missing with 'Unknown')
    for col in peer_cols:
        if col not in gold_df.columns:
            log.warning("  Column '%s' missing — using 'Unknown' for peer grouping.", col)
            gold_df[col] = "Unknown"

    gold_df["peer_group"] = (
        gold_df["Outlet_Type"].astype(str) + "|" +
        gold_df["province"].astype(str)    + "|" +
        gold_df["Outlet_Size"].astype(str)
    )

    gold_df["peer_90th_potential"] = (
        gold_df.groupby("peer_group")["max_monthly_volume"]
        .transform(lambda x: x.quantile(0.90))
    )
    gold_df["peer_75th_potential"] = (
        gold_df.groupby("peer_group")["max_monthly_volume"]
        .transform(lambda x: x.quantile(0.75))
    )
    gold_df["peer_group_size"] = (
        gold_df.groupby("peer_group")["Outlet_ID"]
        .transform("count")
    )
    gold_df["peer_percentile_rank"] = (
        gold_df.groupby("peer_group")["max_monthly_volume"]
        .rank(pct=True)
    )

    # Uncapping factor: how many times above current max could this outlet go?
    # Clipped at 3× — prevents absurd predictions for severely constrained outlets
    gold_df["uncapping_factor"] = (
        gold_df["peer_90th_potential"] /
        (gold_df["max_monthly_volume"] + 1e-9)
    ).clip(1.0, 3.0)

    log.info("  Peer groups: %d unique", gold_df["peer_group"].nunique())
    log.info("  Uncapping factor — mean=%.2f  median=%.2f  max=%.2f",
             gold_df["uncapping_factor"].mean(),
             gold_df["uncapping_factor"].median(),
             gold_df["uncapping_factor"].max())
    return gold_df


# =============================================================================
# 9. Constraint classification  (FIXED — now produces Genuinely_Low_Demand)
# =============================================================================
def classify_constraints(gold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each outlet by the dominant constraint suppressing its demand.
    Assigns constraint_type and constraint_multiplier for the two-stage model.

    FIXED: Tightened thresholds so genuinely low-demand outlets are not
    misclassified as constrained (original code left ~0 in that bucket).

    Apply the multiplier AFTER the XGBoost base prediction in the notebook:
        final_prediction = xgb_prediction × constraint_multiplier × jan26_seasonality_index
    """
    log.info("Classifying outlet constraint types...")

    def _classify(row):
        cv        = row.get("cv_volume", 0)
        max_vol   = row.get("max_monthly_volume", 0)
        mean_vol  = row.get("mean_monthly_volume", 0)
        hit_rate  = row.get("credit_limit_hit_rate", 0)
        stockout  = row.get("stockout_flag", 0)
        zero_mths = row.get("zero_order_months", 0)
        peak_avg  = row.get("peak_to_avg_ratio", 1)
        months    = row.get("months_active", 0)

        # Genuinely low demand:
        # stable (low CV), small volume, no constraint signals, active long enough
        if (mean_vol < 30 and max_vol < 50 and
                cv < 0.30 and stockout == 0 and
                hit_rate < 0.20 and peak_avg < 1.5 and months >= 6):
            return "Genuinely_Low_Demand"

        # Credit constrained: repeatedly ordering near the same ceiling
        if hit_rate > 0.50 and cv < 0.30:
            return "Credit_Constrained"

        # Stockout constrained: actual zero-volume months within active window
        if stockout == 1 and zero_mths >= 1:
            return "Stockout_Constrained"

        # Delivery capped: high peak-to-average, high variability
        if peak_avg > 2.5 and cv > 0.35:
            return "Delivery_Capped"

        # Default: moderate constraint (some suppression but no clear signal)
        return "Moderately_Constrained"

    gold_df["constraint_type"] = gold_df.apply(_classify, axis=1)
    gold_df["constraint_multiplier"] = gold_df["constraint_type"].map(CONSTRAINT_MULTIPLIERS)

    dist = gold_df["constraint_type"].value_counts()
    log.info("  Constraint distribution:\n%s", dist.to_string())
    return gold_df


# =============================================================================
# 10. Build Gold (FIXED ordering — one-hot encoding last)
# =============================================================================
def build_gold(outlet_features: pd.DataFrame,
               jan_features: pd.DataFrame,
               sku_trend: pd.DataFrame,
               master_df: pd.DataFrame,
               coords_df: pd.DataFrame,
               jan_season_lkp: pd.DataFrame,
               holiday_meta: dict,
               poi_df=None) -> pd.DataFrame:
    log.info("Building Gold layer...")
    gold = outlet_features.copy()

    # ── Merge January historical features ─────────────────────────────────────
    gold = gold.merge(jan_features, on="Outlet_ID", how="left")

    # ── Merge SKU trend ────────────────────────────────────────────────────────
    gold = gold.merge(sku_trend, on="Outlet_ID", how="left")

    # ── Merge outlet master (keep original categorical columns for now) ────────
    gold = gold.merge(
        master_df[["Outlet_ID", "Outlet_Size", "Outlet_Type", "Cooler_Count"]],
        on="Outlet_ID", how="left",
    )

    # ── FIX: Outlet size ordinal BEFORE one-hot encoding ─────────────────────
    gold["outlet_size_ord"] = gold["Outlet_Size"].map(SIZE_ORDINAL).fillna(0).astype(int)

    # ── Province feature from dominant distributor ─────────────────────────────
    gold["province"] = gold["dominant_distributor"].map(PROVINCE_MAP).fillna("Unknown")

    # ── Price per liter (revenue efficiency signal) ────────────────────────────
    gold["price_per_liter"] = (
        gold["total_value"] / (gold["total_volume"] + 1e-9)
    ).round(2)

    # ── Merge coordinates ──────────────────────────────────────────────────────
    gold = gold.merge(
        coords_df[["Outlet_ID", "Latitude", "Longitude"]],
        on="Outlet_ID", how="left",
    )

    # ── Peer group benchmarks (BEFORE one-hot encoding) ───────────────────────
    gold = compute_peer_benchmarks(gold)

    # ── Constraint classification (BEFORE one-hot encoding) ───────────────────
    gold = classify_constraints(gold)

    # ── Seasonality lookup ────────────────────────────────────────────────────
    gold = gold.merge(jan_season_lkp, on="dominant_distributor", how="left")
    gold["jan26_seasonality_index"] = gold["jan26_seasonality_index"].fillna(1.0)

    # ── Global holiday metadata ────────────────────────────────────────────────
    for k, v in holiday_meta.items():
        gold[k] = v

    # ── Merge POI features ────────────────────────────────────────────────────
    if poi_df is not None and not poi_df.empty:
        log.info("  Merging POI features...")
        gold = gold.merge(poi_df, on="Outlet_ID", how="left")
        poi_cols = [c for c in poi_df.columns if c != "Outlet_ID"]
        gold[poi_cols] = gold[poi_cols].fillna(0)

    # ── One-hot encoding — LAST STEP so ordinal/peer columns are already set ──
    gold = pd.get_dummies(
        gold,
        columns=["Outlet_Size", "Outlet_Type", "province"],
        drop_first=False,
        dtype=int,
    )

    # ── Fill remaining nulls ───────────────────────────────────────────────────
    numeric_cols = gold.select_dtypes(include=[np.number]).columns
    gold[numeric_cols] = gold[numeric_cols].fillna(0)

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
    jan_features    = build_january_features(monthly_agg)
    sku_trend       = build_sku_trend(monthly_agg)
    jan_season      = build_seasonality_lookup(season)
    holiday_meta    = compute_holiday_density(holidays)

    # Load POI data if already scraped
    poi_path = EXTERNAL / "poi_features.csv"
    poi_df   = pd.read_csv(poi_path, low_memory=False) if poi_path.exists() else None
    if poi_df is None:
        log.warning("POI features not found — run 04_poi_scraping.py first for best results.")

    gold = build_gold(
        outlet_features, jan_features, sku_trend,
        master, coords, jan_season, holiday_meta, poi_df,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    feat_path  = GOLD_DIR / "outlet_features.csv"
    model_path = GOLD_DIR / "model_input_final.csv"
    gold.to_csv(feat_path,  index=False)
    gold.to_csv(model_path, index=False)

    log.info("Saved outlet_features.csv   → %s", feat_path)
    log.info("Saved model_input_final.csv → %s", model_path)

    # ── Quick sanity print ────────────────────────────────────────────────────
    log.info("\nConstraint type distribution (should now have Genuinely_Low_Demand):")
    if "constraint_type" in gold.columns:
        for ct, cnt in gold["constraint_type"].value_counts().items():
            log.info("  %-30s %d", ct, cnt)

    log.info("\nKey feature stats:")
    key_cols = [
        "mean_monthly_volume", "max_monthly_volume", "peer_90th_potential",
        "uncapping_factor", "constraint_multiplier",
        "jan_mean_volume", "price_per_liter",
    ]
    existing = [c for c in key_cols if c in gold.columns]
    log.info("\n%s", gold[existing].describe().to_string())

    log.info("\nGOLD ENRICHMENT COMPLETE")
    return gold


if __name__ == "__main__":
    run_gold_enrichment()