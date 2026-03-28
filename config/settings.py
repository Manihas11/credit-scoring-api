"""
Pipeline configuration.
Override any value via environment variable of the same name.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.getenv("BASE_DIR",       "/opt/airflow")
RAW_DIR     = os.getenv("RAW_DIR",        f"{BASE_DIR}/data/raw")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", f"{BASE_DIR}/data/processed")
VALIDATED_DIR = os.getenv("VALIDATED_DIR", f"{BASE_DIR}/data/validated")

# ── Kaggle ────────────────────────────────────────────────────────────────────
KAGGLE_COMPETITION = "home-credit-default-risk"

# ── Validation thresholds ─────────────────────────────────────────────────────
MAX_NULL_RATE_CRITICAL = 0.05   # columns like TARGET, SK_ID_CURR
MAX_NULL_RATE_EXTERNAL = 0.60   # external credit scores (expected to be sparse)
MIN_ROWS               = 100_000

# ── Model (used in Month 2) ───────────────────────────────────────────────────
TARGET_COL    = "TARGET"
ID_COL        = "SK_ID_CURR"
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
CV_FOLDS      = 5

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
