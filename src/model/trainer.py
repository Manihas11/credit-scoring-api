"""
Credit Scoring Model — Month 2
XGBoost + SHAP explainability + MLflow tracking

Usage:
    python -m src.model.trainer
"""

import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger

log = get_logger("trainer")

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH  = "data/processed/features.parquet"
MODELS_DIR     = "models"
TARGET_COL     = "TARGET"
ID_COL         = "SK_ID_CURR"
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
CV_FOLDS       = 5

XGBOOST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "scale_pos_weight": 11,   # ~8.1% default rate → weight minority class
    "eval_metric":      "auc",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    "tree_method":      "hist",
}


class CreditModelTrainer:

    def __init__(self, features_path: str = FEATURES_PATH):
        self.features_path = features_path
        os.makedirs(MODELS_DIR, exist_ok=True)
        mlflow.set_experiment("credit_scoring")

    def run(self):
        log.info("── Starting training run ────────────────────────")

        df = self._load_features()
        X, y = self._prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        log.info(f"Train: {X_train.shape}  Test: {X_test.shape}")
        log.info(f"Default rate — train: {y_train.mean():.2%}  test: {y_test.mean():.2%}")

        with mlflow.start_run(run_name="xgboost_baseline"):
            # ── Cross-validation ──────────────────────────────────────────────
            cv_auc = self._cross_validate(X_train, y_train)
            log.info(f"CV AUC: {cv_auc:.4f}")
            mlflow.log_metric("cv_auc", cv_auc)

            # ── Final model ───────────────────────────────────────────────────
            model = xgb.XGBClassifier(**XGBOOST_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=50,
            )

            # ── Evaluation ────────────────────────────────────────────────────
            metrics = self._evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.log_params(XGBOOST_PARAMS)

            # ── SHAP ──────────────────────────────────────────────────────────
            shap_values, feature_importance = self._compute_shap(model, X_test)

            # ── Save artifacts ────────────────────────────────────────────────
            self._save_artifacts(model, X_train, feature_importance, metrics)
            mlflow.xgboost.log_model(model, "model")

            log.info("── Training complete ────────────────────────────")
            log.info(f"  Test AUC-ROC : {metrics['test_auc_roc']:.4f}")
            log.info(f"  Test AP      : {metrics['test_avg_precision']:.4f}")
            log.info(f"  CV AUC       : {cv_auc:.4f}")
            log.info(f"  Model saved  → {MODELS_DIR}/")

        return model, metrics

    # ── Steps ─────────────────────────────────────────────────────────────────

    def _load_features(self) -> pd.DataFrame:
        log.info(f"Loading features from {self.features_path}")
        df = pd.read_parquet(self.features_path)
        log.info(f"  Shape: {df.shape}")
        return df

    def _prepare_xy(self, df: pd.DataFrame):
        drop_cols = [TARGET_COL, ID_COL]
        feature_cols = [c for c in df.columns if c not in drop_cols]

        # Convert bool columns to int (XGBoost wants numeric)
        for col in df[feature_cols].select_dtypes(include="bool").columns:
            df[col] = df[col].astype(int)

        # Convert any remaining categoricals
        for col in df[feature_cols].select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        X = df[feature_cols].astype(float)
        y = df[TARGET_COL].astype(int)

        log.info(f"  Features: {len(feature_cols)}  Samples: {len(X)}")
        return X, y

    def _cross_validate(self, X, y) -> float:
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        aucs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            m = xgb.XGBClassifier(**{**XGBOOST_PARAMS, "n_estimators": 200, "verbose": 0})
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            preds = m.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, preds)
            aucs.append(auc)
            log.info(f"  Fold {fold}: AUC = {auc:.4f}")

        return float(np.mean(aucs))

    def _evaluate(self, model, X_test, y_test) -> dict:
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "test_auc_roc":       round(roc_auc_score(y_test, proba), 4),
            "test_avg_precision": round(average_precision_score(y_test, proba), 4),
        }

        log.info("\n" + classification_report(y_test, preds, target_names=["Repaid", "Default"]))
        return metrics

    def _compute_shap(self, model, X_test):
        log.info("Computing SHAP values (this takes ~1 min)...")
        explainer = shap.TreeExplainer(model)

        # Use a sample for speed
        sample = X_test.sample(min(2000, len(X_test)), random_state=RANDOM_STATE)
        shap_values = explainer.shap_values(sample)

        # Feature importance from SHAP
        importance = pd.DataFrame({
            "feature":    X_test.columns,
            "shap_importance": np.abs(shap_values).mean(axis=0),
        }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

        log.info("\nTop 15 features by SHAP importance:")
        log.info(importance.head(15).to_string(index=False))

        importance.to_csv(f"{MODELS_DIR}/shap_importance.csv", index=False)
        mlflow.log_artifact(f"{MODELS_DIR}/shap_importance.csv")

        return shap_values, importance

    def _save_artifacts(self, model, X_train, feature_importance, metrics):
        # Save model
        model.save_model(f"{MODELS_DIR}/credit_model.json")

        # Save feature list (needed by API later)
        feature_list = list(X_train.columns)
        with open(f"{MODELS_DIR}/feature_list.json", "w") as f:
            json.dump(feature_list, f, indent=2)

        # Save metrics
        with open(f"{MODELS_DIR}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        log.info(f"  Saved: credit_model.json, feature_list.json, metrics.json, shap_importance.csv")


if __name__ == "__main__":
    trainer = CreditModelTrainer()
    trainer.run()
