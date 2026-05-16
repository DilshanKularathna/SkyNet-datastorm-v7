"""
=============================================================================
Pipeline Step 01 — Bronze Ingestion
=============================================================================
Rule: ZERO transformations. Load raw files exactly as-is.
Logs ingestion timestamp, row/col counts, null summary, and unique values
for categorical columns — used by Silver cleaning to discover allowed values.
=============================================================================
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[1]
SOURCE_DIR  = ROOT / "data" / "bronze"
BRONZE_DIR  = ROOT / "data" / "bronze"
LOG_FILE    = ROOT / "data" / "bronze" / "_ingestion_log.json"

BRONZE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "transactions_history_final.csv",
    "outlet_master.csv",
    "outlet_coordinates.csv",
    "distributor_seasonality_details.csv",
    "holiday_list.csv",
]

# Columns we want to profile unique values for (helps Silver define allowed lists)
CATEGORICAL_PROFILE_COLS = {
    "outlet_master.csv":                 ["Outlet_Size", "Outlet_Type"],
    "distributor_seasonality_details.csv": ["Seasonality_Index"],
    "holiday_list.csv":                  ["Holiday_Type"],
    "transactions_history_final.csv":    ["Distributor_ID"],
}


def ingest_file(filename: str) -> dict:
    """
    Load a single CSV from bronze, validate readability, log metadata.
    NO transforms — dtype inference only.
    """
    src_path = SOURCE_DIR / filename
    if not src_path.exists():
        log.error("  [MISSING] %s not found in %s", filename, SOURCE_DIR)
        return {
            "file": filename, "status": "MISSING",
            "rows": None, "columns": None,
            "ingested_at": datetime.now().isoformat(),
            "size_bytes": None,
        }

    log.info("Ingesting → %s", filename)
    df = pd.read_csv(src_path, low_memory=False)

    row_count  = len(df)
    col_count  = len(df.columns)
    size_bytes = src_path.stat().st_size

    log.info("  ✓ %s | rows=%d  cols=%d  size=%.1f KB",
             filename, row_count, col_count, size_bytes / 1024)

    # ── Null summary ──────────────────────────────────────────────────────────
    null_summary = df.isnull().sum().to_dict()
    null_cols    = {c: v for c, v in null_summary.items() if v > 0}
    if null_cols:
        log.warning("  Null counts: %s", null_cols)

    # ── Unique value profiling for categorical columns ─────────────────────────
    unique_value_profile = {}
    if filename in CATEGORICAL_PROFILE_COLS:
        for col in CATEGORICAL_PROFILE_COLS[filename]:
            if col in df.columns:
                uniques = sorted(df[col].dropna().astype(str).str.strip().unique().tolist())
                unique_value_profile[col] = uniques
                log.info("  Unique %-28s : %s", f"'{col}'", uniques)

    record = {
        "file":                  filename,
        "status":                "OK",
        "rows":                  row_count,
        "columns":               col_count,
        "column_names":          list(df.columns),
        "ingested_at":           datetime.now().isoformat(),
        "size_bytes":            size_bytes,
        "dtypes":                {c: str(t) for c, t in df.dtypes.items()},
        "null_summary":          null_summary,
        "null_cols_only":        null_cols,
        "unique_value_profile":  unique_value_profile,
        "sample_head":           df.head(3).to_dict(orient="records"),
    }
    return record


def run_bronze_ingestion() -> list:
    log.info("=" * 70)
    log.info("BRONZE INGESTION START — %s", datetime.now().isoformat())
    log.info("=" * 70)

    ingestion_log = []
    for filename in DATASETS:
        record = ingest_file(filename)
        ingestion_log.append(record)

    with open(LOG_FILE, "w", encoding="utf-8") as fh:
        json.dump(ingestion_log, fh, indent=2, default=str)

    ok_count  = sum(1 for r in ingestion_log if r["status"] == "OK")
    err_count = len(ingestion_log) - ok_count
    log.info("-" * 70)
    log.info("BRONZE INGESTION COMPLETE — %d OK  |  %d MISSING/ERROR", ok_count, err_count)
    log.info("Ingestion log → %s", LOG_FILE)
    log.info("=" * 70)
    return ingestion_log


if __name__ == "__main__":
    run_bronze_ingestion()