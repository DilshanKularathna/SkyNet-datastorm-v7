"""
=============================================================================
Pipeline Step 02 — Silver Cleaning & Data Quality
=============================================================================
Runs all parameterised DQ checks from dq_checks/quality_functions.py.
Produces:
  data/silver/<dataset>_clean.csv
  data/rejected/<dataset>_rejected.csv  (with failure_reason column)
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
    check_future_dates, check_coordinates, check_allowed_values, dq_summary,
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

ALLOWED_OUTLET_SIZES  = ["Small", "Medium", "Large", "Extra Large"]
ALLOWED_OUTLET_TYPES  = ["Grocery", "Eatery", "Pharmacy", "Kade", "Supermarket", "Hotel"]
ALLOWED_SEASONALITY   = ["Favorable", "Moderate", "Un-Favorable"]
ALLOWED_HOLIDAY_TYPES = ["Public", "Bank", "Poya Day", "Mercantile"]


def _save(df, path, label):
    df.to_csv(path, index=False)
    log.info("  Saved %s → %s  (%d rows)", label, path.name, len(df))


def _concat_rejected(buckets, columns):
    filled = [r for r in buckets if not r.empty]
    return pd.concat(filled, ignore_index=True) if filled else pd.DataFrame(columns=columns)


# =============================================================================
# 1. outlet_master
# =============================================================================
def clean_outlet_master():
    ds, src = "outlet_master", BRONZE_DIR / "outlet_master.csv"
    log.info("\n[%s] Cleaning...", ds)
    df = pd.read_csv(src, low_memory=False)
    orig = len(df)
    df["Cooler_Count"] = pd.to_numeric(df["Cooler_Count"], errors="coerce")
    q = []
    df, r = check_nulls(df, ["Outlet_ID", "Outlet_Size", "Outlet_Type"], ds); q.append(r)
    df, r = check_duplicates(df, ["Outlet_ID"], ds); q.append(r)
    df, r = check_allowed_values(df, "Outlet_Size", ALLOWED_OUTLET_SIZES, ds); q.append(r)
    df, r = check_allowed_values(df, "Outlet_Type", ALLOWED_OUTLET_TYPES, ds); q.append(r)
    df, r = check_value_range(df, "Cooler_Count", 0, 100, ds); q.append(r)
    rej = _concat_rejected(q, df.columns.tolist())
    _save(df,  SILVER_DIR   / "outlet_master_clean.csv",    "outlet_master_clean")
    _save(rej, REJECTED_DIR / "outlet_master_rejected.csv", "outlet_master_rejected")
    return df, dq_summary(ds, orig, len(df), len(rej))


# =============================================================================
# 2. transactions
# =============================================================================
def clean_transactions(master_df):
    ds, src = "transactions_history", BRONZE_DIR / "transactions_history_final.csv"
    log.info("\n[%s] Cleaning...", ds)
    df = pd.read_csv(src, low_memory=False)
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
    df, r = check_ghost_transactions(df, "Volume_Liters", "Total_Bill_Value", ds); q.append(r)
    df, r = check_value_range(df, "Total_Bill_Value", 0, None, ds); q.append(r)
    df, r = check_volume_spikes(df, "Outlet_ID", "Volume_Liters", 10, ds); q.append(r)
    df, r = check_future_dates(df, "Year", "Month", ds); q.append(r)
    rej = _concat_rejected(q, df.columns.tolist())
    _save(df,  SILVER_DIR   / "transactions_clean.csv",    "transactions_clean")
    _save(rej, REJECTED_DIR / "transactions_rejected.csv", "transactions_rejected")
    return df, dq_summary(ds, orig, len(df), len(rej))


# =============================================================================
# 3. coordinates
# =============================================================================
def clean_coordinates(master_df):
    ds, src = "outlet_coordinates", BRONZE_DIR / "outlet_coordinates.csv"
    log.info("\n[%s] Cleaning...", ds)
    df = pd.read_csv(src, low_memory=False)
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
# 4. seasonality
# =============================================================================
def clean_seasonality():
    ds, src = "distributor_seasonality", BRONZE_DIR / "distributor_seasonality_details.csv"
    log.info("\n[%s] Cleaning...", ds)
    df = pd.read_csv(src, low_memory=False)
    orig = len(df)
    df["Year"]  = pd.to_numeric(df["Year"],  errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    q = []
    df, r = check_nulls(df, ["Distributor_ID", "Year", "Month", "Seasonality_Index"], ds); q.append(r)
    df, r = check_duplicates(df, ["Distributor_ID", "Year", "Month"], ds); q.append(r)
    df, r = check_value_range(df, "Year",  2023, 2026, ds); q.append(r)
    df, r = check_value_range(df, "Month", 1,    12,   ds); q.append(r)
    df, r = check_allowed_values(df, "Seasonality_Index", ALLOWED_SEASONALITY, ds); q.append(r)
    rej = _concat_rejected(q, df.columns.tolist())
    _save(df,  SILVER_DIR   / "seasonality_clean.csv",    "seasonality_clean")
    _save(rej, REJECTED_DIR / "seasonality_rejected.csv", "seasonality_rejected")
    return df, dq_summary(ds, orig, len(df), len(rej))


# =============================================================================
# 5. holidays
# =============================================================================
def clean_holidays():
    ds, src = "holiday_list", BRONZE_DIR / "holiday_list.csv"
    log.info("\n[%s] Cleaning...", ds)
    df = pd.read_csv(src, low_memory=False, parse_dates=["Date"])
    orig = len(df)
    q = []
    df, r = check_nulls(df, ["Date", "Holiday_Name", "Holiday_Type"], ds); q.append(r)
    df, r = check_duplicates(df, ["Date"], ds); q.append(r)
    df, r = check_allowed_values(df, "Holiday_Type", ALLOWED_HOLIDAY_TYPES, ds); q.append(r)
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

    master_df, s = clean_outlet_master();   summaries.append(s)
    _,         s = clean_transactions(master_df); summaries.append(s)
    _,         s = clean_coordinates(master_df);  summaries.append(s)
    _,         s = clean_seasonality();    summaries.append(s)
    _,         s = clean_holidays();       summaries.append(s)

    with open(DQ_SUMMARY_FILE, "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, default=str)

    log.info("\nSILVER CLEANING COMPLETE — DQ report → %s", DQ_SUMMARY_FILE)
    return summaries


if __name__ == "__main__":
    run_silver_cleaning()
