"""
=============================================================================
Data Quality Functions — SkyNet DataStorm v7
=============================================================================
All reusable, parameterised DQ checks live here.
Every function returns (clean_df, rejected_df) and NEVER silently drops rows.
Every rejected row carries a 'failure_reason' column.
=============================================================================
"""

import re
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Sri Lanka bounding box (approx.) ─────────────────────────────────────────
SRI_LANKA_BOUNDS = {
    "lat_min": 5.85,
    "lat_max": 9.90,
    "lon_min": 79.65,
    "lon_max": 81.90,
}


# =============================================================================
# Helper: tag rows with a failure reason and append to a quarantine bucket
# =============================================================================

def _tag_and_quarantine(
    bad_rows: pd.DataFrame,
    reason: str,
    quarantine_bucket: list,
) -> None:
    """Stamp a failure_reason and push rows into the quarantine bucket."""
    if bad_rows.empty:
        return
    tagged = bad_rows.copy()
    if "failure_reason" not in tagged.columns:
        tagged["failure_reason"] = ""
    # Append to existing reason (a row may fail multiple checks)
    tagged["failure_reason"] = tagged["failure_reason"].apply(
        lambda x: f"{x}; {reason}" if x else reason
    )
    quarantine_bucket.append(tagged)
    log.warning("  [DQ] %d rows flagged — %s", len(bad_rows), reason)


# =============================================================================
# 1. Null / Missing Value Check
# =============================================================================

def check_nulls(
    df: pd.DataFrame,
    mandatory_cols: list,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag rows where any mandatory column is null.

    Returns
    -------
    clean_df     : rows where ALL mandatory columns are non-null
    rejected_df  : rows with at least one null in mandatory_cols
    """
    log.info("[%s] check_nulls — mandatory columns: %s", dataset_name, mandatory_cols)
    existing_cols = [c for c in mandatory_cols if c in df.columns]
    missing_mask = df[existing_cols].isnull().any(axis=1)

    quarantine: list = []
    _tag_and_quarantine(
        df[missing_mask],
        reason="NULL value in mandatory column",
        quarantine_bucket=quarantine,
    )
    clean_df = df[~missing_mask].copy()
    rejected_df = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=df.columns)
    log.info("  → clean=%d  rejected=%d", len(clean_df), len(rejected_df))
    return clean_df, rejected_df


# =============================================================================
# 2. Duplicate Check
# =============================================================================

def check_duplicates(
    df: pd.DataFrame,
    primary_key: list,
    dataset_name: str,
    keep: str = "first",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag duplicate rows based on a composite primary key.

    Parameters
    ----------
    keep : 'first' keeps the first occurrence; duplicates are quarantined.
    """
    log.info("[%s] check_duplicates — pk: %s", dataset_name, primary_key)
    existing_pk = [c for c in primary_key if c in df.columns]
    dup_mask = df.duplicated(subset=existing_pk, keep=keep)

    quarantine: list = []
    _tag_and_quarantine(
        df[dup_mask],
        reason=f"Duplicate primary key: {existing_pk}",
        quarantine_bucket=quarantine,
    )
    clean_df = df[~dup_mask].copy()
    rejected_df = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=df.columns)
    log.info("  → clean=%d  rejected=%d", len(clean_df), len(rejected_df))
    return clean_df, rejected_df


# =============================================================================
# 3. Referential Integrity Check
# =============================================================================

def check_referential_integrity(
    df: pd.DataFrame,
    fk_col: str,
    ref_df: pd.DataFrame,
    ref_col: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag rows where fk_col value does not exist in ref_df[ref_col].
    Classic use-case: transactions referencing unknown Outlet_IDs.
    """
    log.info(
        "[%s] check_referential_integrity — %s → %s",
        dataset_name, fk_col, ref_col,
    )
    valid_keys = set(ref_df[ref_col].dropna().unique())
    orphan_mask = ~df[fk_col].isin(valid_keys)

    quarantine: list = []
    _tag_and_quarantine(
        df[orphan_mask],
        reason=f"Referential integrity failure: {fk_col} not in reference set",
        quarantine_bucket=quarantine,
    )
    clean_df = df[~orphan_mask].copy()
    rejected_df = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=df.columns)
    log.info("  → clean=%d  rejected=%d", len(clean_df), len(rejected_df))
    return clean_df, rejected_df


# =============================================================================
# 4. Value Range Check
# =============================================================================

def check_value_range(
    df: pd.DataFrame,
    col: str,
    min_val: Optional[float],
    max_val: Optional[float],
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag rows where col is outside [min_val, max_val].
    Pass None to skip a boundary check.
    """
    log.info(
        "[%s] check_value_range — %s in [%s, %s]",
        dataset_name, col, min_val, max_val,
    )
    if col not in df.columns:
        log.warning("  Column '%s' not found — skipping range check.", col)
        return df.copy(), pd.DataFrame(columns=df.columns)

    mask = pd.Series(False, index=df.index)
    if min_val is not None:
        mask |= df[col] < min_val
    if max_val is not None:
        mask |= df[col] > max_val

    quarantine: list = []
    _tag_and_quarantine(
        df[mask],
        reason=f"Value out of range for {col}: expected [{min_val}, {max_val}]",
        quarantine_bucket=quarantine,
    )
    clean_df = df[~mask].copy()
    rejected_df = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=df.columns)
    log.info("  → clean=%d  rejected=%d", len(clean_df), len(rejected_df))
    return clean_df, rejected_df


# =============================================================================
# 5. Format / Pattern Check
# =============================================================================

def check_format(
    df: pd.DataFrame,
    col: str,
    pattern: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag rows where col does not match the given regex pattern.
    """
    log.info("[%s] check_format — %s ~ /%s/", dataset_name, col, pattern)
    if col not in df.columns:
        log.warning("  Column '%s' not found — skipping format check.", col)
        return df.copy(), pd.DataFrame(columns=df.columns)

    compiled = re.compile(pattern)
    bad_mask = ~df[col].astype(str).str.match(compiled)

    quarantine: list = []
    _tag_and_quarantine(
        df[bad_mask],
        reason=f"Format mismatch in {col}: does not match pattern '{pattern}'",
        quarantine_bucket=quarantine,
    )
    clean_df = df[~bad_mask].copy()
    rejected_df = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=df.columns)
    log.info("  → clean=%d  rejected=%d", len(clean_df), len(rejected_df))
    return clean_df, rejected_df


# =============================================================================
# 6. Ghost Transaction Check (zero qty / volume but non-zero bill value)
# =============================================================================

def check_ghost_transactions(
    df: pd.DataFrame,
    volume_col: str,
    value_col: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ghost transaction: Volume_Liters == 0 but Total_Bill_Value > 0.
    Also catches negative volumes (credit note miscodings).
    """
    log.info("[%s] check_ghost_transactions", dataset_name)
    ghost_mask = (df[volume_col] == 0) & (df[value_col] > 0)
    negative_mask = df[volume_col] < 0

    quarantine: list = []
    _tag_and_quarantine(
        df[ghost_mask],
        reason="Ghost transaction: zero volume with non-zero bill value",
        quarantine_bucket=quarantine,
    )
    _tag_and_quarantine(
        df[negative_mask & ~ghost_mask],
        reason="Negative volume: likely credit note miscoding",
        quarantine_bucket=quarantine,
    )
    combined_bad = ghost_mask | negative_mask
    clean_df = df[~combined_bad].copy()
    rejected_df = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=df.columns)
    log.info("  → clean=%d  rejected=%d", len(clean_df), len(rejected_df))
    return clean_df, rejected_df


# =============================================================================
# 7. Implausible Volume Spike Check (10× monthly average per outlet)
# =============================================================================

def check_volume_spikes(
    df: pd.DataFrame,
    group_col: str,
    volume_col: str,
    spike_factor: float,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag rows where volume > spike_factor × mean volume for that outlet.
    spike_factor=10 catches data-entry errors (10× average in a single month).
    """
    log.info(
        "[%s] check_volume_spikes — group=%s  factor=%.1f×",
        dataset_name, group_col, spike_factor,
    )
    group_mean = df.groupby(group_col)[volume_col].transform("mean")
    spike_mask = df[volume_col] > (spike_factor * group_mean)

    quarantine: list = []
    _tag_and_quarantine(
        df[spike_mask],
        reason=f"Implausible volume spike (>{spike_factor}× outlet mean)",
        quarantine_bucket=quarantine,
    )
    clean_df = df[~spike_mask].copy()
    rejected_df = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=df.columns)
    log.info("  → clean=%d  rejected=%d", len(clean_df), len(rejected_df))
    return clean_df, rejected_df


# =============================================================================
# 8. Future-Dated Transaction Check
# =============================================================================

def check_future_dates(
    df: pd.DataFrame,
    year_col: str,
    month_col: str,
    dataset_name: str,
    reference_date: Optional[datetime] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag rows where (Year, Month) is in the future relative to reference_date.
    Defaults to today if reference_date is None.
    """
    log.info("[%s] check_future_dates", dataset_name)
    ref = reference_date or datetime.now()
    ref_year, ref_month = ref.year, ref.month

    future_mask = (df[year_col] > ref_year) | (
        (df[year_col] == ref_year) & (df[month_col] > ref_month)
    )
    quarantine: list = []
    _tag_and_quarantine(
        df[future_mask],
        reason=f"Future-dated transaction (after {ref_year}-{ref_month:02d})",
        quarantine_bucket=quarantine,
    )
    clean_df = df[~future_mask].copy()
    rejected_df = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=df.columns)
    log.info("  → clean=%d  rejected=%d", len(clean_df), len(rejected_df))
    return clean_df, rejected_df


# =============================================================================
# 9. Coordinate Validity Check (must be inside Sri Lanka bounding box)
# =============================================================================

def check_coordinates(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag outlets whose GPS coordinates fall outside Sri Lanka's bounding box
    or in the ocean (rough heuristic: both lat AND lon numeric and in bounds).
    """
    log.info("[%s] check_coordinates", dataset_name)
    b = SRI_LANKA_BOUNDS

    lat_invalid = (
        df[lat_col].isnull()
        | (df[lat_col] < b["lat_min"])
        | (df[lat_col] > b["lat_max"])
    )
    lon_invalid = (
        df[lon_col].isnull()
        | (df[lon_col] < b["lon_min"])
        | (df[lon_col] > b["lon_max"])
    )
    bad_mask = lat_invalid | lon_invalid

    quarantine: list = []
    _tag_and_quarantine(
        df[bad_mask],
        reason="Coordinates outside Sri Lanka bounding box (or null)",
        quarantine_bucket=quarantine,
    )
    clean_df = df[~bad_mask].copy()
    rejected_df = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=df.columns)
    log.info("  → clean=%d  rejected=%d", len(clean_df), len(rejected_df))
    return clean_df, rejected_df


# =============================================================================
# 10. Duplicate Order ID on Different Dates (system retry artifact)
# =============================================================================

def check_order_id_date_conflict(
    df: pd.DataFrame,
    order_id_col: str,
    year_col: str,
    month_col: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect order IDs that appear on more than one (Year, Month) combination —
    classic sign of system retry artefacts.
    """
    log.info("[%s] check_order_id_date_conflict", dataset_name)
    if order_id_col not in df.columns:
        log.warning("  Column '%s' not found — skipping.", order_id_col)
        return df.copy(), pd.DataFrame(columns=df.columns)

    period_per_order = (
        df.groupby(order_id_col)[[year_col, month_col]]
        .nunique()
        .max(axis=1)
    )
    conflicted_ids = period_per_order[period_per_order > 1].index
    conflict_mask = df[order_id_col].isin(conflicted_ids)

    quarantine: list = []
    _tag_and_quarantine(
        df[conflict_mask],
        reason="Order ID appears across multiple (Year, Month) periods — retry artefact",
        quarantine_bucket=quarantine,
    )
    clean_df = df[~conflict_mask].copy()
    rejected_df = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=df.columns)
    log.info("  → clean=%d  rejected=%d", len(clean_df), len(rejected_df))
    return clean_df, rejected_df


# =============================================================================
# 11. Allowed Values Check (categorical columns)
# =============================================================================

def check_allowed_values(
    df: pd.DataFrame,
    col: str,
    allowed: list,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag rows where col contains values not in the allowed set.
    """
    log.info("[%s] check_allowed_values — %s ∈ %s", dataset_name, col, allowed)
    if col not in df.columns:
        log.warning("  Column '%s' not found — skipping.", col)
        return df.copy(), pd.DataFrame(columns=df.columns)

    bad_mask = ~df[col].isin(allowed)
    quarantine: list = []
    _tag_and_quarantine(
        df[bad_mask],
        reason=f"Invalid value in {col}: not in {allowed}",
        quarantine_bucket=quarantine,
    )
    clean_df = df[~bad_mask].copy()
    rejected_df = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=df.columns)
    log.info("  → clean=%d  rejected=%d", len(clean_df), len(rejected_df))
    return clean_df, rejected_df


# =============================================================================
# 12. DQ Summary Report
# =============================================================================

def dq_summary(
    dataset_name: str,
    original_count: int,
    clean_count: int,
    rejected_count: int,
) -> dict:
    """Return a structured DQ summary dict for logging / reporting."""
    pct_rejected = (rejected_count / original_count * 100) if original_count else 0
    summary = {
        "dataset": dataset_name,
        "original_rows": original_count,
        "clean_rows": clean_count,
        "rejected_rows": rejected_count,
        "rejection_rate_pct": round(pct_rejected, 2),
        "run_timestamp": datetime.now().isoformat(),
    }
    log.info(
        "[%s] DQ Summary → original=%d  clean=%d  rejected=%d  (%.1f%% rejected)",
        dataset_name, original_count, clean_count, rejected_count, pct_rejected,
    )
    return summary
