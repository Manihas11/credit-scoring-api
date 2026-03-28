"""
CreditPredictor — wraps the trained model for API inference.
Loaded once at startup, reused for every request.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from src.utils.logger import get_logger

log = get_logger("predictor")

SCORE_MIN = 300
SCORE_MAX = 900

BANDS = [
    (750, 900, "Excellent", "Very low risk — strong approval candidate"),
    (700, 749, "Good",      "Low risk — likely to repay comfortably"),
    (650, 699, "Fair",      "Moderate risk — review recommended"),
    (600, 649, "Poor",      "High risk — additional verification needed"),
    (300, 599, "Very Poor", "Very high risk — likely to default"),
]

FEATURE_LABELS = {
    "EXT_SCORE_MEAN":          "average credit bureau score",
    "EXT_SCORE_PROD":          "combined credit bureau score",
    "EXT_SCORE_MIN":           "lowest credit bureau score",
    "DEBT_TO_INCOME":          "debt-to-income ratio",
    "MONTHLY_PAYMENT_RATIO":   "monthly repayment burden",
    "CREDIT_TO_INCOME":        "loan amount vs income",
    "CAREER_FRACTION":         "employment stability",
    "EMPLOYED_YEARS":          "years of employment",
    "AGE_YEARS":               "applicant age profile",
    "ANNUITY_TO_INCOME":       "annuity-to-income ratio",
    "PAYMENT_REGULARITY_30D":  "recent payment regularity",
    "PAYMENT_REGULARITY_90D":  "long-term payment regularity",
    "TXN_COUNT_30D":           "recent transaction activity",
    "INCOME_PER_PERSON":       "income per family member",
    "LOAN_TO_GOODS":           "loan-to-goods ratio",
    "AMT_CREDIT":              "loan amount",
    "AMT_ANNUITY":             "loan annuity amount",
    "CODE_GENDER":             "gender profile",
    "FLAG_OWN_CAR":            "vehicle ownership",
    "FLAG_OWN_REALTY":         "property ownership",
    "AGE_RISK_BAND":           "age risk profile",
    "EMPLOYMENT_RISK":         "employment tenure risk",
    "SPEND_TREND_30_90":       "spending trend",
    "ID_STABILITY":            "identity stability",
    "CREDIT_OVERRUN":          "credit overrun amount",
}


class CreditPredictor:

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model = self._load_model()
        self.feature_list = self._load_feature_list()
        self.explainer = shap.TreeExplainer(self.model)
        log.info(f"CreditPredictor ready — {len(self.feature_list)} features")

    def predict(self, features: dict) -> dict:
        X = self._build_vector(features)
        probability = float(self.model.predict_proba(X)[0, 1])
        score = self._to_score(probability)
        band, description = self._get_band(score)
        top_factors = self._get_factors(X)

        return {
            "score":       score,
            "band":        band,
            "description": description,
            "probability": round(probability, 4),
            "top_factors": top_factors,
        }

    def _load_model(self):
        model = xgb.XGBClassifier()
        model.load_model(f"{self.models_dir}/credit_model.json")
        log.info(f"Model loaded from {self.models_dir}/credit_model.json")
        return model

    def _load_feature_list(self):
        with open(f"{self.models_dir}/feature_list.json") as f:
            return json.load(f)

    def _build_vector(self, features: dict) -> pd.DataFrame:
        row = {col: features.get(col, 0.0) for col in self.feature_list}
        return pd.DataFrame([row], columns=self.feature_list)

    def _to_score(self, probability: float) -> int:
        prob = np.clip(probability, 1e-6, 1 - 1e-6)
        log_odds = np.log(prob / (1 - prob))
        score = SCORE_MAX + (log_odds / 6) * (SCORE_MAX - SCORE_MIN) / 2
        return int(np.clip(score, SCORE_MIN, SCORE_MAX))

    def _get_band(self, score: int):
        for lo, hi, band, desc in BANDS:
            if lo <= score <= hi:
                return band, desc
        return "Unknown", ""

    def _get_factors(self, X: pd.DataFrame, top_n: int = 5) -> list:
        shap_vals = self.explainer.shap_values(X)[0]
        pairs = sorted(
            zip(self.feature_list, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        result = []
        for col, val in pairs[:top_n]:
            label = FEATURE_LABELS.get(col, col.replace("_", " ").lower())
            result.append({
                "factor":    label,
                "direction": "increasing risk" if val > 0 else "decreasing risk",
                "impact":    round(abs(val), 4),
            })
        return result
