"""
Feature Engineering 
Builds 30/60/90-day behavioral window features
on top of the clean parquet.
"""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger("features")


class CreditFeatureEngineer:
    """
    Builds model-ready features from application_clean.parquet.

    Usage:
        fe = CreditFeatureEngineer("data/processed/application_clean.parquet")
        df = fe.run()
        df.to_parquet("data/processed/features.parquet", index=False)
    """

    def __init__(self, clean_path: str):
        self.clean_path = clean_path

    def run(self) -> pd.DataFrame:
        log.info(f"Loading clean data from {self.clean_path}")
        df = pd.read_parquet(self.clean_path)
        log.info(f"Loaded shape: {df.shape}")

        df = self._debt_burden_features(df)
        df = self._stability_features(df)
        df = self._age_employment_features(df)
        df = self._external_score_features(df)
        df = self._family_features(df)
        df = self._simulate_behavioral_windows(df)
        df = self._drop_raw_cols(df)

        log.info(f"Final feature shape: {df.shape}")
        log.info(f"Features: {list(df.columns)}")
        return df

    # ── Feature groups ────────────────────────────────────────────────────────

    def _debt_burden_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """How much debt relative to income."""
        df["DEBT_TO_INCOME"] = (
            df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        ).round(4)

        df["MONTHLY_PAYMENT_RATIO"] = (
            df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] / 12).replace(0, np.nan)
        ).round(4)

        df["LOAN_TO_GOODS"] = (
            df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
        ).round(4)

        # How much extra they're borrowing beyond goods price
        df["CREDIT_OVERRUN"] = (df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]).round(2)

        log.info("  Debt burden features added")
        return df

    def _stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy for financial stability."""
        # How recently they registered / changed ID (longer ago = more stable)
        df["REGISTRATION_STABILITY"] = (-df["DAYS_REGISTRATION"] / 365).round(1)
        df["ID_STABILITY"] = (-df["DAYS_ID_PUBLISH"] / 365).round(1)

        # Ratio of employed years to age (career fraction)
        df["CAREER_FRACTION"] = (
            df["EMPLOYED_YEARS"] / df["AGE_YEARS"].replace(0, np.nan)
        ).clip(0, 1).round(4)

        log.info("  Stability features added")
        return df

    def _age_employment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Age and employment risk bands."""
        # Young borrowers (<25) and very old (>65) are higher risk
        df["AGE_RISK_BAND"] = pd.cut(
            df["AGE_YEARS"],
            bins=[0, 25, 35, 50, 65, 100],
            labels=[4, 2, 1, 3, 5], ordered=False  # lower=less risk
        ).astype(float)

        # Short employment tenure = higher risk
        df["EMPLOYMENT_RISK"] = (df["EMPLOYED_YEARS"] < 1).astype(int)

        log.info("  Age/employment features added")
        return df

    def _external_score_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate the three external credit bureau scores."""
        ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
                    if c in df.columns]

        if ext_cols:
            df["EXT_SCORE_MEAN"]   = df[ext_cols].mean(axis=1).round(4)
            df["EXT_SCORE_MIN"]    = df[ext_cols].min(axis=1).round(4)
            df["EXT_SCORE_MAX"]    = df[ext_cols].max(axis=1).round(4)
            df["EXT_SCORE_RANGE"]  = (df["EXT_SCORE_MAX"] - df["EXT_SCORE_MIN"]).round(4)
            df["EXT_SCORE_PROD"]   = df[ext_cols].prod(axis=1).round(4)

        log.info("  External score features added")
        return df

    def _family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Family structure risk signals."""
        if "CNT_CHILDREN" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["INCOME_PER_PERSON"] = (
                df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1).replace(0, np.nan)
            ).round(2)

        log.info("  Family features added")
        return df

    def _simulate_behavioral_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate 30/60/90-day behavioral window features.

        In production these come from real UPI/transaction data.
        Here we derive plausible proxies from the static application data
        so the feature schema is correct for when real data arrives.
        """
        rng = np.random.default_rng(seed=42)
        n = len(df)

        # Transaction frequency proxy (higher income = more transactions)
        income_norm = (df["AMT_INCOME_TOTAL"] / df["AMT_INCOME_TOTAL"].max()).values

        for window in [30, 60, 90]:
            scale = window / 30  # 60-day window has ~2x transactions

            df[f"TXN_COUNT_{window}D"] = (
                rng.poisson(lam=income_norm * 20 * scale, size=n)
            ).clip(0, 200)

            df[f"TXN_AMOUNT_{window}D"] = (
                df["AMT_INCOME_TOTAL"] / 12 * (window / 30)
                * rng.uniform(0.6, 1.4, size=n)
            ).round(2)

            # Payment regularity: 0=irregular, 1=perfectly regular
            df[f"PAYMENT_REGULARITY_{window}D"] = (
                rng.beta(a=5, b=2, size=n)  # skewed towards regular
                * df["CAREER_FRACTION"].values
            ).clip(0, 1).round(4)

        # Trend: is spending increasing? (risk signal if yes + high debt)
        df["SPEND_TREND_30_90"] = (
            df["TXN_AMOUNT_30D"] / df["TXN_AMOUNT_90D"].replace(0, np.nan)
        ).fillna(1.0).clip(0, 3).round(4)

        log.info("  Behavioral window features added (30/60/90-day)")
        return df

    def _drop_raw_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop raw columns that have been superseded by engineered features."""
        drop = [
            "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
            "AMT_GOODS_PRICE",  # captured in LOAN_TO_GOODS
        ]
        drop_existing = [c for c in drop if c in df.columns]
        df = df.drop(columns=drop_existing)
        log.info(f"  Dropped {len(drop_existing)} raw columns")
        return df
