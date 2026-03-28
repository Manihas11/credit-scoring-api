"""
Month 2 Tests — Feature engineering + model output
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from src.features.engineer import CreditFeatureEngineer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_clean_df(tmp_path):
    """Synthetic clean parquet matching Month 1 output schema."""
    n = 200
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "SK_ID_CURR":           range(100001, 100001 + n),
        "TARGET":               rng.integers(0, 2, n),
        "AMT_INCOME_TOTAL":     rng.uniform(45000, 472500, n),
        "AMT_CREDIT":           rng.uniform(76410, 1854000, n),
        "AMT_ANNUITY":          rng.uniform(6183, 70006, n),
        "AMT_GOODS_PRICE":      rng.uniform(50000, 1500000, n),
        "AGE_YEARS":            rng.uniform(22, 68, n).round(1),
        "EMPLOYED_YEARS":       rng.uniform(0, 30, n).round(1),
        "DAYS_BIRTH":           -rng.integers(8000, 25000, n),
        "DAYS_EMPLOYED":        -rng.integers(100, 10000, n),
        "DAYS_REGISTRATION":    -rng.integers(100, 15000, n),
        "DAYS_ID_PUBLISH":      -rng.integers(100, 10000, n),
        "CNT_CHILDREN":         rng.integers(0, 4, n),
        "CNT_FAM_MEMBERS":      rng.integers(1, 6, n),
        "EXT_SOURCE_1":         rng.uniform(0.1, 0.9, n).round(4),
        "EXT_SOURCE_2":         rng.uniform(0.1, 0.9, n).round(4),
        "EXT_SOURCE_3":         rng.uniform(0.1, 0.9, n).round(4),
        "CREDIT_TO_INCOME":     rng.uniform(0.5, 13, n).round(4),
        "ANNUITY_TO_INCOME":    rng.uniform(0.04, 0.48, n).round(4),
        "CHILDREN_RATIO":       rng.uniform(0, 1, n).round(4),
        "CODE_GENDER":          rng.integers(0, 2, n),
        "FLAG_OWN_CAR":         rng.integers(0, 2, n),
        "FLAG_OWN_REALTY":      rng.integers(0, 2, n),
    })

    path = str(tmp_path / "application_clean.parquet")
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def engineered_df(sample_clean_df):
    fe = CreditFeatureEngineer(sample_clean_df)
    return fe.run()


# ── Feature engineering tests ─────────────────────────────────────────────────

def test_feature_engineer_runs(sample_clean_df):
    fe = CreditFeatureEngineer(sample_clean_df)
    df = fe.run()
    assert df is not None
    assert len(df) == 200


def test_debt_features_created(engineered_df):
    for col in ["DEBT_TO_INCOME", "MONTHLY_PAYMENT_RATIO", "LOAN_TO_GOODS", "CREDIT_OVERRUN"]:
        assert col in engineered_df.columns, f"Missing: {col}"


def test_stability_features_created(engineered_df):
    for col in ["REGISTRATION_STABILITY", "ID_STABILITY", "CAREER_FRACTION"]:
        assert col in engineered_df.columns, f"Missing: {col}"


def test_external_score_features_created(engineered_df):
    for col in ["EXT_SCORE_MEAN", "EXT_SCORE_MIN", "EXT_SCORE_MAX", "EXT_SCORE_RANGE"]:
        assert col in engineered_df.columns, f"Missing: {col}"


def test_behavioral_windows_created(engineered_df):
    for window in [30, 60, 90]:
        for prefix in ["TXN_COUNT", "TXN_AMOUNT", "PAYMENT_REGULARITY"]:
            col = f"{prefix}_{window}D"
            assert col in engineered_df.columns, f"Missing: {col}"


def test_no_nulls_in_features(engineered_df):
    null_cols = engineered_df.columns[engineered_df.isnull().any()].tolist()
    assert len(null_cols) == 0, f"Null values in: {null_cols}"


def test_no_infinite_values(engineered_df):
    num = engineered_df.select_dtypes(include=np.number)
    assert not np.isinf(num.values).any(), "Infinite values in features"


def test_career_fraction_bounded(engineered_df):
    assert engineered_df["CAREER_FRACTION"].between(0, 1).all()


def test_target_preserved(engineered_df):
    assert "TARGET" in engineered_df.columns
    assert set(engineered_df["TARGET"].unique()).issubset({0, 1})


def test_raw_days_cols_dropped(engineered_df):
    """Raw DAYS_* columns should be removed after feature engineering."""
    for col in ["DAYS_BIRTH", "DAYS_EMPLOYED"]:
        assert col not in engineered_df.columns, f"Raw column not dropped: {col}"


def test_feature_count_reasonable(engineered_df):
    """Should have substantially more features than the clean data (35 → 50+)."""
    assert len(engineered_df.columns) >= 40, (
        f"Expected 45+ features, got {len(engineered_df.columns)}"
    )
