"""
Great Expectations validation suite for raw credit data.
Runs schema, null, range, and uniqueness checks before cleaning.
"""

import os
import json
import pandas as pd
from typing import Dict, Any
from src.utils.logger import get_logger

log = get_logger("validation")


# ── Expectation definitions ────────────────────────────────────────────────────
# Each entry: (column, check_type, params)
# These mirror what Great Expectations calls "expectations"

EXPECTATIONS = [
    # --- Schema checks ---
    {"col": "SK_ID_CURR",       "check": "not_null",      "params": {}},
    {"col": "TARGET",           "check": "not_null",      "params": {}},
    {"col": "AMT_INCOME_TOTAL", "check": "not_null",      "params": {}},

    # --- Range checks ---
    {"col": "TARGET",           "check": "in_set",        "params": {"values": [0, 1]}},
    {"col": "AMT_INCOME_TOTAL", "check": "min_value",     "params": {"min": 0}},
    {"col": "AMT_CREDIT",       "check": "min_value",     "params": {"min": 0}},
    {"col": "AMT_ANNUITY",      "check": "min_value",     "params": {"min": 0}},
    {"col": "CNT_CHILDREN",     "check": "between",       "params": {"min": 0, "max": 20}},

    # --- Uniqueness ---
    {"col": "SK_ID_CURR",       "check": "unique",        "params": {}},

    # --- Null rate thresholds (allow some nulls for external scores) ---
    {"col": "EXT_SOURCE_1",     "check": "null_rate_lt",  "params": {"threshold": 0.60}},
    {"col": "EXT_SOURCE_2",     "check": "null_rate_lt",  "params": {"threshold": 0.15}},
    {"col": "EXT_SOURCE_3",     "check": "null_rate_lt",  "params": {"threshold": 0.30}},

    # --- Categorical values ---
    {"col": "CODE_GENDER",      "check": "in_set",        "params": {"values": ["M", "F", "XNA"]}},
    {"col": "FLAG_OWN_CAR",     "check": "in_set",        "params": {"values": ["Y", "N"]}},
    {"col": "FLAG_OWN_REALTY",  "check": "in_set",        "params": {"values": ["Y", "N"]}},
]


def run_validation_suite(raw_dir: str) -> Dict[str, Any]:
    """
    Load application_train.csv and run all expectations.
    Returns a results dict with success flag and details.
    """
    csv_path = os.path.join(raw_dir, "application_train.csv")
    log.info(f"Loading data for validation: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw data not found at {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} cols")

    passed, failed = [], []

    for exp in EXPECTATIONS:
        col = exp["col"]
        check = exp["check"]
        params = exp["params"]

        if col not in df.columns:
            failed.append({"col": col, "check": check, "reason": "column missing"})
            continue

        ok, reason = _run_check(df, col, check, params)

        if ok:
            passed.append({"col": col, "check": check})
        else:
            failed.append({"col": col, "check": check, "reason": reason})
            log.warning(f"  FAIL  [{check}] on '{col}': {reason}")

    success = len(failed) == 0
    log.info(
        f"Validation {'PASSED' if success else 'FAILED'}: "
        f"{len(passed)} passed, {len(failed)} failed"
    )

    # Write results to file for audit trail
    results = {
        "success": success,
        "total_rows": len(df),
        "passed_checks": len(passed),
        "failed_checks": failed,
        "passed_details": passed,
    }

    results_path = os.path.join(raw_dir, "validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {results_path}")

    return results


# ── Check implementations ──────────────────────────────────────────────────────

def _run_check(df: pd.DataFrame, col: str, check: str, params: dict):
    """Returns (passed: bool, reason: str)."""
    series = df[col]

    if check == "not_null":
        null_count = series.isnull().sum()
        ok = null_count == 0
        return ok, None if ok else f"{null_count} null values"

    elif check == "unique":
        dup_count = series.duplicated().sum()
        ok = dup_count == 0
        return ok, None if ok else f"{dup_count} duplicate values"

    elif check == "in_set":
        values = set(params["values"])
        bad = series.dropna()[~series.dropna().isin(values)]
        ok = len(bad) == 0
        return ok, None if ok else f"{len(bad)} rows with unexpected values: {bad.unique()[:5]}"

    elif check == "min_value":
        min_val = series.min()
        ok = min_val >= params["min"]
        return ok, None if ok else f"min={min_val} below threshold={params['min']}"

    elif check == "between":
        lo, hi = params["min"], params["max"]
        bad = series[(series < lo) | (series > hi)]
        ok = len(bad) == 0
        return ok, None if ok else f"{len(bad)} values outside [{lo}, {hi}]"

    elif check == "null_rate_lt":
        null_rate = series.isnull().mean()
        ok = null_rate < params["threshold"]
        return ok, None if ok else f"null rate {null_rate:.2%} ≥ threshold {params['threshold']:.2%}"

    else:
        return False, f"unknown check type '{check}'"
