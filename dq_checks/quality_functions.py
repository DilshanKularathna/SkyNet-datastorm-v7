"""
=============================================================================
dq_checks/quality_functions.py
Reusable, parameterisable Data Quality check functions.

Each check returns (clean_df, rejected_df).
Every rejected record carries a 'failure_reason' column — never silently dropped.
=============================================================================
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Sri Lanka geographic bounding box
SL_LAT_MIN, SL_LAT_MAX = 5.9,  9.8
SL_LON_MIN, SL_LON_MAX = 79.6, 81.9


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _tag_rejected(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    """Stamp failure_reason onto a subset of rejected rows."""
    out = df.copy()
    out["failure_reason"] = reason
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. Null / Empty check
# ─────────────────────────────────────────────────────────────────────────────

def check_nulls(df: pd.DataFrame, mandatory_cols: list, dataset_name: str):
    """
    Reject records where any mandatory field is null or an empty string.
    Returns (clean_df, rejected_df).
    """
    mask = df[mandatory_cols].isnull().any(axis=1)
    for col in mandatory_cols:
        if df[col].dtype == object:
            mask |= df[col].astype(str).str.strip().eq("")

    rejected = _tag_rejected(df[mask], f"NULL_VALUE: mandatory field(s) [{', '.join(mandatory_cols)}]")
    clean    = df[~mask].copy()

    if mask.sum():
        log.warning("  [%s] check_nulls: %d records rejected", dataset_name, mask.sum())
    else:
        log.info("  [%s] check_nulls: PASS", dataset_name)

    return clean, rejected


# ─────────────────────────────────────────────────────────────────────────────
# 2. Duplicate check
# ─────────────────────────────────────────────────────────────────────────────

def check_duplicates(df: pd.DataFrame, primary_key: list, dataset_name: str):
    """
    Detect duplicate records based on a configurable composite key.
    First occurrence is kept; all subsequent duplicates are rejected.
    """
    dup_mask = df.duplicated(subset=primary_key, keep="first")

    rejected = _tag_rejected(
        df[dup_mask],
        f"DUPLICATE: composite key ({', '.join(primary_key)}) appears more than once"
    )
    clean = df[~dup_mask].copy()

    if dup_mask.sum():
        log.warning("  [%s] check_duplicates: %d duplicate records rejected", dataset_name, dup_mask.sum())
    else:
        log.info("  [%s] check_duplicates: PASS", dataset_name)

    return clean, rejected


# ─────────────────────────────────────────────────────────────────────────────
# 3. Referential integrity check
# ─────────────────────────────────────────────────────────────────────────────

def check_referential_integrity(df: pd.DataFrame, fk_col: str,
                                 ref_df: pd.DataFrame, ref_col: str,
                                 dataset_name: str):
    """
    Validate that every FK value in df[fk_col] exists in ref_df[ref_col].
    Orphan records (no matching parent) are rejected.
    """
    valid_keys = set(ref_df[ref_col].dropna().unique())
    mask       = ~df[fk_col].isin(valid_keys)

    rejected = _tag_rejected(
        df[mask],
        f"REF_INTEGRITY: {fk_col} value not found in reference column [{ref_col}]"
    )
    clean = df[~mask].copy()

    if mask.sum():
        log.warning("  [%s] check_referential_integrity: %d orphan records rejected "
                    "(sample bad keys: %s)",
                    dataset_name, mask.sum(),
                    list(df.loc[mask, fk_col].unique())[:5])
    else:
        log.info("  [%s] check_referential_integrity: PASS", dataset_name)

    return clean, rejected


# ─────────────────────────────────────────────────────────────────────────────
# 4. Value range check
# ─────────────────────────────────────────────────────────────────────────────

def check_value_range(df: pd.DataFrame, col: str, min_val, max_val,
                      dataset_name: str):
    """
    Assert that a numeric field falls within [min_val, max_val].
    Pass None for either bound to skip that side.
    NaN values (coercion failures) are also rejected here.
    """
    mask = df[col].isnull()   # catches to_numeric coercion failures
    if min_val is not None:
        mask |= df[col] < min_val
    if max_val is not None:
        mask |= df[col] > max_val

    bounds   = f"[{min_val}, {max_val}]"
    rejected = _tag_rejected(df[mask], f"VALUE_RANGE: {col} outside expected bounds {bounds}")
    clean    = df[~mask].copy()

    if mask.sum():
        log.warning("  [%s] check_value_range (%s %s): %d records rejected",
                    dataset_name, col, bounds, mask.sum())
    else:
        log.info("  [%s] check_value_range (%s): PASS", dataset_name, col)

    return clean, rejected


# ─────────────────────────────────────────────────────────────────────────────
# 5. Allowed values check (categorical)
# ─────────────────────────────────────────────────────────────────────────────

def check_allowed_values(df: pd.DataFrame, col: str, allowed: list,
                          dataset_name: str):
    """
    Validate that a categorical column contains only allowed values.
    Comparison is case-insensitive; clean output is normalised to canonical casing.
    """
    normalised    = df[col].astype(str).str.strip()
    allowed_lower = [str(v).lower() for v in allowed]
    mask          = ~normalised.str.lower().isin(allowed_lower)

    unique_bad = normalised[mask].unique()
    rejected   = _tag_rejected(
        df[mask],
        f"INVALID_VALUE: {col} unexpected value(s): {list(unique_bad)[:5]}"
    )

    clean = df[~mask].copy()
    # Normalise to canonical casing (e.g. "un-favorable" → "Un-Favorable")
    canonical_map = {v.lower(): v for v in allowed}
    clean[col]    = clean[col].astype(str).str.strip().str.lower().map(canonical_map)

    if mask.sum():
        log.warning("  [%s] check_allowed_values (%s): %d records rejected. "
                    "Unknown values: %s",
                    dataset_name, col, mask.sum(), list(unique_bad)[:10])
    else:
        log.info("  [%s] check_allowed_values (%s): PASS", dataset_name, col)

    return clean, rejected


# ─────────────────────────────────────────────────────────────────────────────
# 6. Ghost transaction check
# ─────────────────────────────────────────────────────────────────────────────

def check_ghost_transactions(df: pd.DataFrame, vol_col: str, val_col: str,
                              dataset_name: str):
    """
    Flag ghost entries: volume <= 0 but bill value > 0.
    These are SFA connectivity / retry artifacts and must be quarantined.
    """
    mask = (df[vol_col] <= 0) & (df[val_col] > 0)

    rejected = _tag_rejected(
        df[mask],
        f"GHOST_TRANSACTION: {vol_col}<=0 but {val_col}>0 (SFA connectivity artifact)"
    )
    clean = df[~mask].copy()

    if mask.sum():
        log.warning("  [%s] check_ghost_transactions: %d ghost records rejected", dataset_name, mask.sum())
    else:
        log.info("  [%s] check_ghost_transactions: PASS", dataset_name)

    return clean, rejected


# ─────────────────────────────────────────────────────────────────────────────
# 7. Volume spike check
# ─────────────────────────────────────────────────────────────────────────────

def check_volume_spikes(df: pd.DataFrame, outlet_col: str, vol_col: str,
                         spike_multiplier: float, dataset_name: str):
    """
    Reject single-period rows where volume exceeds spike_multiplier × outlet mean.
    Designed to catch data-entry errors (e.g. extra zero added).
    """
    outlet_mean = df.groupby(outlet_col)[vol_col].transform("mean")
    mask        = df[vol_col] > (spike_multiplier * outlet_mean)

    rejected = _tag_rejected(
        df[mask],
        f"VOLUME_SPIKE: {vol_col} > {spike_multiplier}× outlet mean (likely data-entry error)"
    )
    clean = df[~mask].copy()

    if mask.sum():
        log.warning("  [%s] check_volume_spikes: %d spike records rejected", dataset_name, mask.sum())
    else:
        log.info("  [%s] check_volume_spikes: PASS", dataset_name)

    return clean, rejected


# ─────────────────────────────────────────────────────────────────────────────
# 8. Future date check
# ─────────────────────────────────────────────────────────────────────────────

def check_future_dates(df: pd.DataFrame, year_col: str, month_col: str,
                        dataset_name: str,
                        ceiling_year: int = 2025, ceiling_month: int = 12):
    """
    Reject records with Year/Month beyond the data-collection ceiling.
    Default ceiling: December 2025 (3-year dataset).
    """
    mask = (df[year_col] > ceiling_year) | (
        (df[year_col] == ceiling_year) & (df[month_col] > ceiling_month)
    )

    rejected = _tag_rejected(
        df[mask],
        f"FUTURE_DATE: {year_col}/{month_col} exceeds collection ceiling "
        f"{ceiling_year}-{ceiling_month:02d}"
    )
    clean = df[~mask].copy()

    if mask.sum():
        log.warning("  [%s] check_future_dates: %d future-dated records rejected", dataset_name, mask.sum())
    else:
        log.info("  [%s] check_future_dates: PASS", dataset_name)

    return clean, rejected


# ─────────────────────────────────────────────────────────────────────────────
# 9. Coordinate sanity check
# ─────────────────────────────────────────────────────────────────────────────

def check_coordinates(df: pd.DataFrame, lat_col: str, lon_col: str,
                       dataset_name: str):
    """
    Reject coordinates that are:
      - Null
      - Exactly (0, 0) — ghost coordinates
      - Outside the Sri Lanka bounding box
    """
    null_mask  = df[lat_col].isnull() | df[lon_col].isnull()
    ghost_mask = (~null_mask) & (df[lat_col] == 0) & (df[lon_col] == 0)
    bbox_mask  = (
        (~null_mask) & (~ghost_mask) & (
            (df[lat_col] < SL_LAT_MIN) | (df[lat_col] > SL_LAT_MAX) |
            (df[lon_col] < SL_LON_MIN) | (df[lon_col] > SL_LON_MAX)
        )
    )
    mask = null_mask | ghost_mask | bbox_mask

    def _reason(row):
        if pd.isnull(row[lat_col]) or pd.isnull(row[lon_col]):
            return "NULL_COORD: latitude or longitude is null"
        if row[lat_col] == 0 and row[lon_col] == 0:
            return "GHOST_COORD: (0, 0) placeholder coordinate"
        return (f"OUT_OF_BOUNDS: lat={row[lat_col]:.4f}, lon={row[lon_col]:.4f} "
                f"outside Sri Lanka bbox")

    rejected_df = df[mask].copy()
    if not rejected_df.empty:
        rejected_df["failure_reason"] = rejected_df.apply(_reason, axis=1)

    clean = df[~mask].copy()

    if mask.sum():
        log.warning("  [%s] check_coordinates: %d invalid coordinate records rejected",
                    dataset_name, mask.sum())
    else:
        log.info("  [%s] check_coordinates: PASS", dataset_name)

    return clean, rejected_df


# ─────────────────────────────────────────────────────────────────────────────
# 10. DQ Summary
# ─────────────────────────────────────────────────────────────────────────────

def dq_summary(dataset_name: str, original_count: int,
               clean_count: int, rejected_count: int) -> dict:
    """Return a structured summary dict for the JSON DQ report."""
    retention = (clean_count / original_count * 100) if original_count else 0
    log.info(
        "  [%s] SUMMARY — original=%d | clean=%d | rejected=%d | retention=%.1f%%",
        dataset_name, original_count, clean_count, rejected_count, retention,
    )
    return {
        "dataset":         dataset_name,
        "original_count":  original_count,
        "clean_count":     clean_count,
        "rejected_count":  rejected_count,
        "retention_pct":   round(retention, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11. Utility: discover unique values in a column (use before check_allowed_values)
# ─────────────────────────────────────────────────────────────────────────────

def discover_unique_values(df: pd.DataFrame, col: str, dataset_name: str) -> list:
    """
    Log and return unique values for a categorical column.
    Run this during Bronze/Silver audit to learn what's actually in the data
    before committing to an allowed-values list.
    """
    uniques = sorted(df[col].dropna().astype(str).str.strip().unique().tolist())
    log.info("  [%s] unique values in '%s' (%d): %s",
             dataset_name, col, len(uniques), uniques)
    return uniques