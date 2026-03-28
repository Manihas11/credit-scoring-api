"""
Tests for CreditDataCleaner.
Run: pytest tests/ -v
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.ingestion.cleaner import CreditDataCleaner


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe mimicking raw Home Credit data."""
    return pd.DataFrame({
        "SK_ID_CURR":          [100001, 100002, 100003, 100004, 100005],
        "TARGET":              [0, 1, 0, 1, 0],
        "AMT_INCOME_TOTAL":    [180000, 270000, None, 450000, 135000],
        "AMT_CREDIT":          [406597, 1293502, 135000, 312682, 450000],
        "AMT_ANNUITY":         [24700, 35698, 6750, 14175, 20250],
        "AMT_GOODS_PRICE":     [351000, 1129500, 135000, 297000, 450000],
        "DAYS_BIRTH":          [-9461, -16765, -19046, -19005, -17932],
        "DAYS_EMPLOYED":       [-637, -1188, 365243, -3039, -199],  # 365243 = sentinel
        "DAYS_REGISTRATION":   [-3648, -1186, -291, -3899, -290],
        "DAYS_ID_PUBLISH":     [-2120, -291, -4176, -4739, -3510],
        "CNT_CHILDREN":        [0, 0, 2, 1, 0],
        "CNT_FAM_MEMBERS":     [1, 2, 4, 3, 2],
        "EXT_SOURCE_1":        [0.083, None, 0.502, 0.623, 0.456],
        "EXT_SOURCE_2":        [0.262, 0.622, 0.555, 0.311, 0.789],
        "EXT_SOURCE_3":        [None, 0.511, None, 0.722, 0.600],
        "CODE_GENDER":         ["M", "F", "F", "M", "F"],
        "FLAG_OWN_CAR":        ["N", "N", "Y", "N", "Y"],
        "FLAG_OWN_REALTY":     ["Y", "N", "Y", "Y", "N"],
        "NAME_EDUCATION_TYPE": [
            "Secondary / secondary special",
            "Higher education",
            "Secondary / secondary special",
            "Incomplete higher",
            "Higher education",
        ],
        "NAME_INCOME_TYPE":    [
            "Working", "Commercial associate", "Working",
            "State servant", "Working",
        ],
        "NAME_CONTRACT_TYPE":  [
            "Cash loans", "Cash loans", "Revolving loans",
            "Cash loans", "Revolving loans",
        ],
    })


@pytest.fixture
def cleaner(tmp_path, sample_df):
    raw_dir = str(tmp_path / "raw")
    output_dir = str(tmp_path / "processed")
    os.makedirs(raw_dir)
    sample_df.to_csv(os.path.join(raw_dir, "application_train.csv"), index=False)
    return CreditDataCleaner(raw_dir=raw_dir, output_dir=output_dir)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_sentinel_replaced(cleaner, tmp_path):
    output_path = cleaner.run()
    df = pd.read_parquet(output_path)
    # 365243 should no longer exist anywhere
    assert 365243 not in df["DAYS_EMPLOYED"].values


def test_no_nulls_after_clean(cleaner):
    output_path = cleaner.run()
    df = pd.read_parquet(output_path)
    assert df.isnull().sum().sum() == 0, "Nulls remain after cleaning"


def test_basic_features_created(cleaner):
    output_path = cleaner.run()
    df = pd.read_parquet(output_path)
    for feature in ["AGE_YEARS", "EMPLOYED_YEARS", "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME"]:
        assert feature in df.columns, f"Expected feature '{feature}' not found"


def test_age_reasonable(cleaner):
    output_path = cleaner.run()
    df = pd.read_parquet(output_path)
    assert df["AGE_YEARS"].between(18, 80).all(), "AGE_YEARS has unrealistic values"


def test_target_intact(cleaner, sample_df):
    output_path = cleaner.run()
    df = pd.read_parquet(output_path)
    assert set(df["TARGET"].unique()).issubset({0, 1})
    assert len(df) == len(sample_df), "Row count changed during cleaning"


def test_parquet_output_exists(cleaner, tmp_path):
    output_path = cleaner.run()
    assert os.path.exists(output_path)
    assert output_path.endswith(".parquet")


def test_no_infinite_values(cleaner):
    output_path = cleaner.run()
    df = pd.read_parquet(output_path)
    num_df = df.select_dtypes(include=np.number)
    assert not np.isinf(num_df.values).any(), "Infinite values found in cleaned data"
