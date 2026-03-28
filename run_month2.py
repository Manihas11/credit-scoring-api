import os
import pandas as pd
from src.features.engineer import CreditFeatureEngineer
from src.model.trainer import CreditModelTrainer
from src.utils.logger import get_logger

log = get_logger("month2")

CLEAN_PATH    = "data/processed/application_clean.parquet"
FEATURES_PATH = "data/processed/features.parquet"


def main():
    # ── Step 1: Feature engineering ───────────────────────────────────────────
    log.info("═" * 50)
    log.info("STEP 1 — Feature Engineering")
    log.info("═" * 50)

    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(
            f"Clean data not found at {CLEAN_PATH}\n"
            "Run Month 1 first: python -c \"from src.ingestion.cleaner import "
            "CreditDataCleaner; CreditDataCleaner('data/raw','data/processed').run()\""
        )

    fe = CreditFeatureEngineer(CLEAN_PATH)
    df = fe.run()
    df.to_parquet(FEATURES_PATH, index=False)
    log.info(f"Features saved → {FEATURES_PATH}  shape={df.shape}")

    # ── Step 2: Model training ────────────────────────────────────────────────
    log.info("═" * 50)
    log.info("STEP 2 — Model Training")
    log.info("═" * 50)

    trainer = CreditModelTrainer(features_path=FEATURES_PATH)
    model, metrics = trainer.run()

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("═" * 50)
    log.info("MONTH 2 COMPLETE")
    log.info("═" * 50)
    log.info(f"  Test AUC-ROC  : {metrics['test_auc_roc']}")
    log.info(f"  Test Avg Prec : {metrics['test_avg_precision']}")
    log.info(f"  Features path : {FEATURES_PATH}")
    log.info(f"  Model path    : models/credit_model.json")
    log.info(f"  SHAP rankings : models/shap_importance.csv")
    log.info(f"  MLflow UI     : run 'mlflow ui' then open http://localhost:5000")
    log.info("═" * 50)


if __name__ == "__main__":
    main()
