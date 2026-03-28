"""
Credit Data Cleaner
Handles null imputation, dtype fixing, and outlier capping
for the Home Credit Default Risk dataset.
"""

import os
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger("cleaner")


class CreditDataCleaner:
    """
    Cleans raw application_train.csv into a model-ready parquet file.

    Usage:
        cleaner = CreditDataCleaner(raw_dir="data/raw", output_dir="data/processed")
        output_path = cleaner.run()
    """

    # Columns to keep for Month 1 MVP (expand in Month 2)
    KEEP_COLS = [
        "SK_ID_CURR",          # applicant ID
        "TARGET",              # 1 = defaulted, 0 = repaid
        "AMT_INCOME_TOTAL",    # annual income
        "AMT_CREDIT",          # loan amount requested
        "AMT_ANNUITY",         # loan annuity
        "AMT_GOODS_PRICE",     # goods price of loan
        "DAYS_BIRTH",          # age in days (negative)
        "DAYS_EMPLOYED",       # employment duration in days (negative)
        "DAYS_REGISTRATION",   # days since last registration change
        "DAYS_ID_PUBLISH",     # days since ID was changed
        "CNT_CHILDREN",        # number of children
        "CNT_FAM_MEMBERS",     # family size
        "EXT_SOURCE_1",        # external credit score 1
        "EXT_SOURCE_2",        # external credit score 2
        "EXT_SOURCE_3",        # external credit score 3
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_EDUCATION_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_CONTRACT_TYPE",
    ]

    # Numeric columns that use 365243 as a sentinel for "unemployed/N/A"
    SENTINEL_COLS = ["DAYS_EMPLOYED"]
    SENTINEL_VALUE = 365243

    def __init__(self, raw_dir: str, output_dir: str):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> str:
        log.info("Loading raw data...")
        df = self._load()

        log.info(f"Raw shape: {df.shape}")

        df = self._select_columns(df)
        df = self._fix_sentinels(df)
        df = self._engineer_basic_features(df)
        df = self._impute_nulls(df)
        df = self._encode_categoricals(df)
        df = self._cap_outliers(df)

        output_path = os.path.join(self.output_dir, "application_clean.parquet")
        df.to_parquet(output_path, index=False)
        log.info(f"Clean data saved → {output_path}  shape={df.shape}")

        self._log_summary(df)
        return output_path

    # ── Steps ─────────────────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        path = os.path.join(self.raw_dir, "application_train.csv")
        return pd.read_csv(path, low_memory=False)

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.KEEP_COLS if c in df.columns]
        dropped = set(self.KEEP_COLS) - set(available)
        if dropped:
            log.warning(f"Columns not found in raw data (skipped): {dropped}")
        return df[available].copy()

    def _fix_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace 365243 sentinel in DAYS_EMPLOYED with NaN."""
        for col in self.SENTINEL_COLS:
            if col in df.columns:
                mask = df[col] == self.SENTINEL_VALUE
                df.loc[mask, col] = np.nan
                log.info(f"  {col}: replaced {mask.sum()} sentinels with NaN")
        return df

    def _engineer_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive simple ratio features before imputation."""
        if "DAYS_BIRTH" in df.columns:
            df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)

        if "DAYS_EMPLOYED" in df.columns:
            df["EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED"] / 365).round(1)

        if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["CREDIT_TO_INCOME"] = (
                df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
            ).round(4)

        if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["ANNUITY_TO_INCOME"] = (
                df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
            ).round(4)

        if "CNT_CHILDREN" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
            df["CHILDREN_RATIO"] = (
                df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
            ).round(4)

        log.info("  Basic features engineered: AGE_YEARS, CREDIT_TO_INCOME, etc.")
        return df

    def _impute_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Median imputation for numerics, mode for categoricals."""
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include="object").columns

        for col in num_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                median = df[col].median()
                df[col] = df[col].fillna(median)
                log.info(f"  {col}: filled {null_count} nulls with median={median:.4f}")

        for col in cat_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                mode = df[col].mode()[0]
                df[col] = df[col].fillna(mode)
                log.info(f"  {col}: filled {null_count} nulls with mode='{mode}'")

        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode binary categoricals; one-hot encode multi-class."""
        binary_map = {"Y": 1, "N": 0, "M": 1, "F": 0}

        for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CODE_GENDER"]:
            if col in df.columns:
                df[col] = df[col].map(binary_map)

        multi_cat_cols = ["NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "NAME_CONTRACT_TYPE"]
        ohe_cols = [c for c in multi_cat_cols if c in df.columns]
        if ohe_cols:
            df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
            log.info(f"  One-hot encoded: {ohe_cols}")

        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap extreme values at 1st and 99th percentile."""
        cap_cols = [
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
            "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME",
        ]
        for col in cap_cols:
            if col not in df.columns:
                continue
            p01 = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            before = df[col].describe()["max"]
            df[col] = df[col].clip(lower=p01, upper=p99)
            log.info(f"  {col}: capped [{p01:.2f}, {p99:.2f}] (was max={before:.2f})")
        return df

    def _log_summary(self, df: pd.DataFrame):
        null_pct = (df.isnull().sum() / len(df) * 100).max()
        target_rate = df["TARGET"].mean() * 100 if "TARGET" in df.columns else None
        log.info("── Clean data summary ──────────────────────────")
        log.info(f"  Shape         : {df.shape}")
        log.info(f"  Max null %    : {null_pct:.2f}%")
        if target_rate:
            log.info(f"  Default rate  : {target_rate:.1f}%")
        log.info(f"  Dtypes        : {df.dtypes.value_counts().to_dict()}")
        log.info("────────────────────────────────────────────────")
