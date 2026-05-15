"""
=============================================================================
Pipeline Step 01 — Bronze Ingestion
=============================================================================
Rule: ZERO transformations. Load raw files exactly as-is.
Log ingestion timestamp + row counts for audit trail.
=============================================================================
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# Source: wherever the raw files arrive (can be an uploads/raw folder or bronze itself)
SOURCE_DIR  = ROOT / "data" / "bronze"
BRONZE_DIR  = ROOT / "data" / "bronze"
LOG_FILE    = ROOT / "data" / "bronze" / "_ingestion_log.json"

BRONZE_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset manifest ───────────────────────────────────────────────────────────
DATASETS = [
    "transactions_history_final.csv",
    "outlet_master.csv",
    "outlet_coordinates.csv",
    "distributor_seasonality_details.csv",
    "holiday_list.csv",
]


# =============================================================================
# Core ingestion function
# =============================================================================

def ingest_file(filename: str) -> dict:
    """
    Load a single CSV from bronze, validate it is readable, and log metadata.
    NO transforms — dtype inference only.
    """
    src_path = SOURCE_DIR / filename
    if not src_path.exists():
        log.error("  [MISSING] %s not found in %s", filename, SOURCE_DIR)
        return {
            "file": filename,
            "status": "MISSING",
            "rows": None,
            "columns": None,
            "ingested_at": datetime.now().isoformat(),
            "size_bytes": None,
        }

    log.info("Ingesting → %s", filename)
    df = pd.read_csv(src_path, low_memory=False)

    row_count = len(df)
    col_count = len(df.columns)
    size_bytes = src_path.stat().st_size

    log.info(
        "  ✓ %s | rows=%d  cols=%d  size=%.1f KB",
        filename, row_count, col_count, size_bytes / 1024,
    )

    return {
        "file": filename,
        "status": "OK",
        "rows": row_count,
        "columns": col_count,
        "column_names": list(df.columns),
        "ingested_at": datetime.now().isoformat(),
        "size_bytes": size_bytes,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_summary": df.isnull().sum().to_dict(),
        "sample_head": df.head(3).to_dict(orient="records"),
    }


# =============================================================================
# Main
# =============================================================================

def run_bronze_ingestion() -> list[dict]:
    log.info("=" * 70)
    log.info("BRONZE INGESTION START — %s", datetime.now().isoformat())
    log.info("=" * 70)

    ingestion_log = []
    for filename in DATASETS:
        record = ingest_file(filename)
        ingestion_log.append(record)

    # ── Persist ingestion log ──────────────────────────────────────────────────
    with open(LOG_FILE, "w", encoding="utf-8") as fh:
        json.dump(ingestion_log, fh, indent=2, default=str)

    log.info("-" * 70)
    ok_count  = sum(1 for r in ingestion_log if r["status"] == "OK")
    err_count = len(ingestion_log) - ok_count
    log.info(
        "BRONZE INGESTION COMPLETE — %d OK  |  %d MISSING/ERROR",
        ok_count, err_count,
    )
    log.info("Ingestion log saved → %s", LOG_FILE)
    log.info("=" * 70)

    return ingestion_log


if __name__ == "__main__":
    run_bronze_ingestion()
