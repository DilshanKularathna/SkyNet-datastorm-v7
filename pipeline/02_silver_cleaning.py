"""
=============================================================================
Pipeline Step 02 — Silver Cleaning & Data Quality
=============================================================================
Runs all parameterised DQ checks from dq_checks/quality_functions.py.

FIX vs original:
  - Discovers actual unique values from Bronze BEFORE applying allowed-values
    checks so no valid records are silently quarantined due to a hardcoded list.
  - Adds Volume_Liters range check (min=0) to catch negative volumes.
  - Total_Bill_Value range check now also catches zero-value ghost entries.

Produces:
  data/silver/<dataset>_clean.csv
  data/rejected/<dataset>_rejected.csv   (includes failure_reason column)
  data/silver/_dq_summary.json
=============================================================================
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dq_checks.quality_functions import (
    check_nulls, check_duplicates, check_referential_integrity,
    check_value_range, check_ghost_transactions, check_volume_spikes,
    check_future_dates, check_coordinates, check_allowed_values,
    discover_unique_values, dq_summary,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

BRONZE_DIR   = ROOT / "data" / "bronze"
SILVER_DIR   = ROOT / "data" / "silver"
REJECTED_DIR = ROOT / "data" / "rejected"
SILVER_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)
DQ_SUMMARY_FILE = SILVER_DIR / "_dq_summary.json"

# ── Fallback allowed values (used ONLY if discovery finds nothing unexpected) ──
# After running Bronze ingestion, check _ingestion_log.json unique_value_profile
# to verify these match your actual data. Add any new values you discover.
FALLBACK_OUTLET_SIZES    = ["Small", "Medium", "Large", "Extra Large"]
FALLBACK_OUTLET_TYPES    = ["Grocery", "Eatery", "Pharmacy", "Kade",
                             "Supermarket", "Hotel", "Restaurant"]
FALLBACK_SEASONALITY     = ["Favorable", "Moderate", "Un-Favorable"]
FALLBACK_HOLIDAY_TYPES   = ["Public", "Bank", "Poya Day", "Mercantile"]


def _discover_allowed(df: pd.DataFrame, col: str, fallback: list,
                       dataset_name: str) -> list:
    """
    Discover actual unique values in the Bronze data.
    If Bronze has more values than the fallback list, we ADD them (no rejection
    of data just because we didn't anticipate a value).
    Log the diff so analysts can decide whether to add or reject new categories.
    """
    discovered = discover_unique_values(df, col, dataset_name)
    fallback_lower = {v.lower() for v in fallback}
    new_values     = [v for v in discovered if v.lower() not in fallback_lower]
    if new_values:
        log.warning("  [%s] NEW/UNEXPECTED values in '%s': %s "
                    "— adding to allowed list. Review and quarantine if invalid.",
                    dataset_name, col, new_values)
    return fallback + new_values   # keep everything; analysts decide later


def _save(df: pd.DataFrame, path: Path, label: str):
    df.to_csv(path, index=False)
    log.info("  Saved %-35s → %s  (%d rows)", label, path.name, len(df))


def _concat_rejected(buckets: list, columns: list) -> pd.DataFrame:
    filled = [r for r in buckets if not r.empty]
    if not filled:
        return pd.DataFrame(columns=columns + ["failure_reason"])
    return pd.concat(filled, ignore_index=True)


# =============================================================================
# 1. outlet_master
# =============================================================================
def clean_outlet_master():
    ds  = "outlet_master"
    src = BRONZE_DIR / "outlet_master.csv"
    log.info("\n[%s] Cleaning...", ds)
    df  = pd.read_csv(src, low_memory=False)
    orig = len(df)

    df["Cooler_Count"] = pd.to_numeric(df["Cooler_Count"], errors="coerce")

    # Discover actual values before applying check
    allowed_sizes = _discover_allowed(df, "Outlet_Size", FALLBACK_OUTLET_SIZES, ds)
    allowed_types = _discover_allowed(df, "Outlet_Type", FALLBACK_OUTLET_TYPES, ds)

    q = []
    df, r = check_nulls(df, ["Outlet_ID", "Outlet_Size", "Outlet_Type"], ds); q.append(r)
    df, r = check_duplicates(df, ["Outlet_ID"], ds); q.append(r)
    df, r = check_allowed_values(df, "Outlet_Size", allowed_sizes, ds); q.append(r)
    df, r = check_allowed_values(df, "Outlet_Type", allowed_types, ds); q.append(r)
    df, r = check_value_range(df, "Cooler_Count", 0, 100, ds); q.append(r)

    rej = _concat_rejected(q, df.columns.tolist())
    _save(df,  SILVER_DIR   / "outlet_master_clean.csv",    "outlet_master_clean")
    _save(rej, REJECTED_DIR / "outlet_master_rejected.csv", "outlet_master_rejected")
    return df, dq_summary(ds, orig, len(df), len(rej))


# =============================================================================
# 2. transactions_history
# =============================================================================
def clean_transactions(master_df: pd.DataFrame):
    ds  = "transactions_history"
    src = BRONZE_DIR / "transactions_history_final.csv"
    log.info("\n[%s] Cleaning...", ds)
    df  = pd.read_csv(src, low_memory=False)
    orig = len(df)

    for col in ["Year", "Month", "Volume_Liters", "Total_Bill_Value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    q = []
    df, r = check_nulls(df, ["Outlet_ID", "Year", "Month", "Distributor_ID",
                              "SKU_ID", "Volume_Liters", "Total_Bill_Value"], ds); q.append(r)
    df, r = check_duplicates(df, ["Outlet_ID", "Year", "Month", "SKU_ID"], ds); q.append(r)
    df, r = check_referential_integrity(df, "Outlet_ID", master_df, "Outlet_ID", ds); q.append(r)
    df, r = check_value_range(df, "Year",  2023, 2025, ds); q.append(r)
    df, r = check_value_range(df, "Month", 1,    12,   ds); q.append(r)

    # FIX: catch ghost entries first, THEN check Volume_Liters range
    df, r = check_ghost_transactions(df, "Volume_Liters", "Total_Bill_Value", ds); q.append(r)

    # FIX: now reject any remaining negative volumes (credit note miscodings)
    df, r = check_value_range(df, "Volume_Liters",    0, None, ds); q.append(r)
    df, r = check_value_range(df, "Total_Bill_Value", 0, None, ds); q.append(r)

    df, r = check_volume_spikes(df, "Outlet_ID", "Volume_Liters", 10, ds); q.append(r)
    df, r = check_future_dates(df, "Year", "Month", ds); q.append(r)

    rej = _concat_rejected(q, df.columns.tolist())
    _save(df,  SILVER_DIR   / "transactions_clean.csv",    "transactions_clean")
    _save(rej, REJECTED_DIR / "transactions_rejected.csv", "transactions_rejected")
    return df, dq_summary(ds, orig, len(df), len(rej))


# =============================================================================
# 3. outlet_coordinates
# =============================================================================
def clean_coordinates(master_df: pd.DataFrame):
    ds  = "outlet_coordinates"
    src = BRONZE_DIR / "outlet_coordinates.csv"
    log.info("\n[%s] Cleaning...", ds)
    df  = pd.read_csv(src, low_memory=False)
    orig = len(df)

    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    q = []
    df, r = check_nulls(df, ["Outlet_ID", "Latitude", "Longitude"], ds); q.append(r)
    df, r = check_duplicates(df, ["Outlet_ID"], ds); q.append(r)
    df, r = check_referential_integrity(df, "Outlet_ID", master_df, "Outlet_ID", ds); q.append(r)
    df, r = check_coordinates(df, "Latitude", "Longitude", ds); q.append(r)

    rej = _concat_rejected(q, df.columns.tolist())
    _save(df,  SILVER_DIR   / "coordinates_clean.csv",    "coordinates_clean")
    _save(rej, REJECTED_DIR / "coordinates_rejected.csv", "coordinates_rejected")
    return df, dq_summary(ds, orig, len(df), len(rej))


# =============================================================================
# 4. distributor_seasonality
# =============================================================================
def clean_seasonality():
    ds  = "distributor_seasonality"
    src = BRONZE_DIR / "distributor_seasonality_details.csv"
    log.info("\n[%s] Cleaning...", ds)
    df  = pd.read_csv(src, low_memory=False)
    orig = len(df)

    df["Year"]  = pd.to_numeric(df["Year"],  errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")

    # Discover actual seasonality values before checking
    allowed_season = _discover_allowed(df, "Seasonality_Index", FALLBACK_SEASONALITY, ds)

    q = []
    df, r = check_nulls(df, ["Distributor_ID", "Year", "Month", "Seasonality_Index"], ds); q.append(r)
    df, r = check_duplicates(df, ["Distributor_ID", "Year", "Month"], ds); q.append(r)
    # Allow up to 2026 (Jan 2026 may be present for prediction)
    df, r = check_value_range(df, "Year",  2023, 2026, ds); q.append(r)
    df, r = check_value_range(df, "Month", 1,    12,   ds); q.append(r)
    df, r = check_allowed_values(df, "Seasonality_Index", allowed_season, ds); q.append(r)

    rej = _concat_rejected(q, df.columns.tolist())
    _save(df,  SILVER_DIR   / "seasonality_clean.csv",    "seasonality_clean")
    _save(rej, REJECTED_DIR / "seasonality_rejected.csv", "seasonality_rejected")
    return df, dq_summary(ds, orig, len(df), len(rej))


# =============================================================================
# 5. holiday_list
# =============================================================================
def clean_holidays():
    ds  = "holiday_list"
    src = BRONZE_DIR / "holiday_list.csv"
    log.info("\n[%s] Cleaning...", ds)
    df  = pd.read_csv(src, low_memory=False, parse_dates=["Date"])
    orig = len(df)

    # Discover actual holiday types
    allowed_htype = _discover_allowed(df, "Holiday_Type", FALLBACK_HOLIDAY_TYPES, ds)

    q = []
    df, r = check_nulls(df, ["Date", "Holiday_Name", "Holiday_Type"], ds); q.append(r)
    df, r = check_duplicates(df, ["Date"], ds); q.append(r)
    df, r = check_allowed_values(df, "Holiday_Type", allowed_htype, ds); q.append(r)

    rej = _concat_rejected(q, df.columns.tolist())
    _save(df,  SILVER_DIR   / "holidays_clean.csv",    "holidays_clean")
    _save(rej, REJECTED_DIR / "holidays_rejected.csv", "holidays_rejected")
    return df, dq_summary(ds, orig, len(df), len(rej))


# =============================================================================
# Main
# =============================================================================
def run_silver_cleaning():
    log.info("=" * 70)
    log.info("SILVER CLEANING START — %s", datetime.now().isoformat())
    log.info("=" * 70)

    summaries = []
    master_df, s = clean_outlet_master();            summaries.append(s)
    _,         s = clean_transactions(master_df);    summaries.append(s)
    _,         s = clean_coordinates(master_df);     summaries.append(s)
    _,         s = clean_seasonality();              summaries.append(s)
    _,         s = clean_holidays();                 summaries.append(s)

    with open(DQ_SUMMARY_FILE, "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, default=str)

    log.info("\nSILVER CLEANING COMPLETE — DQ report → %s", DQ_SUMMARY_FILE)
    log.info("\nRetention summary:")
    for s in summaries:
        log.info("  %-30s original=%-7d clean=%-7d rejected=%-5d retention=%.1f%%",
                 s["dataset"], s["original_count"], s["clean_count"],
                 s["rejected_count"], s["retention_pct"])
    return summaries


if __name__ == "__main__":
    run_silver_cleaning()