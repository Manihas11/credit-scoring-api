"""
Score Converter
Turns raw XGBoost probability into a 300–900 credit score
with a human-readable band and top risk factors (via SHAP).

This is the core of the investor demo — what makes it feel
like a real product rather than a Jupyter notebook.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from src.utils.logger import get_logger

log = get_logger("scorer")

SCORE_MIN = 300
SCORE_MAX = 900

BANDS = [
    (750, 900, "Excellent", "Very low risk — strong approval candidate"),
    (700, 749, "Good",      "Low risk — likely to repay comfortably"),
    (650, 699, "Fair",      "Moderate risk — review recommended"),
    (600, 649, "Poor",      "High risk — additional verification needed"),
    (300, 599, "Very Poor", "Very high risk — likely to default"),
]

# Human-readable names for the top features
FEATURE_LABELS = {
    "EXT_SCORE_MEAN":          "average credit bureau score",
    "EXT_SCORE_MIN":           "lowest credit bureau score",
    "EXT_SCORE_PROD":          "combined credit bureau score",
    "DEBT_TO_INCOME":          "high debt-to-income ratio",
    "MONTHLY_PAYMENT_RATIO":   "monthly repayment burden",
    "CREDIT_TO_INCOME":        "loan amount vs income",
    "CAREER_FRACTION":         "employment stability",
    "EMPLOYED_YEARS":          "years of employment",
    "AGE_YEARS":               "applicant age",
    "ANNUITY_TO_INCOME":       "annuity-to-income ratio",
    "PAYMENT_REGULARITY_30D":  "recent payment regularity",
    "PAYMENT_REGULARITY_90D":  "payment regularity trend",
    "TXN_COUNT_30D":           "recent transaction activity",
    "INCOME_PER_PERSON":       "income per family member",
    "LOAN_TO_GOODS":           "loan-to-goods ratio",
    "AGE_RISK_BAND":           "age risk profile",
    "EMPLOYMENT_RISK":         "short employment tenure",
    "SPEND_TREND_30_90":       "spending trend",
}


class CreditScorer:
    """
    Loads a trained model and scores a single applicant.

    Usage:
        scorer = CreditScorer(models_dir="models")
        result = scorer.score(applicant_features_dict)
        # returns: { score, band, description, top_factors, probability }
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model = self._load_model()
        self.feature_list = self._load_feature_list()
        self.explainer = shap.TreeExplainer(self.model)
        log.info(f"CreditScorer ready — {len(self.feature_list)} features")

    def score(self, features: dict) -> dict:
        """
        Score a single applicant.

        Args:
            features: dict of feature_name → value

        Returns:
            {
                score:        742,
                band:         "Good",
                description:  "Low risk — likely to repay comfortably",
                probability:  0.087,
                top_factors:  [
                    { factor, direction, impact },
                    ...
                ]
            }
        """
        X = self._build_feature_vector(features)
        probability = float(self.model.predict_proba(X)[0, 1])
        score = self._probability_to_score(probability)
        band, description = self._get_band(score)
        top_factors = self._get_top_factors(X)

        result = {
            "score":       score,
            "band":        band,
            "description": description,
            "probability": round(probability, 4),
            "top_factors": top_factors,
        }

        log.info(f"Score: {score} ({band})  P(default)={probability:.4f}")
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self) -> xgb.XGBClassifier:
        model = xgb.XGBClassifier()
        model.load_model(f"{self.models_dir}/credit_model.json")
        return model

    def _load_feature_list(self) -> list:
        with open(f"{self.models_dir}/feature_list.json") as f:
            return json.load(f)

    def _build_feature_vector(self, features: dict) -> pd.DataFrame:
        """Fill missing features with 0 (safe default for inference)."""
        row = {col: features.get(col, 0.0) for col in self.feature_list}
        return pd.DataFrame([row], columns=self.feature_list)

    def _probability_to_score(self, probability: float) -> int:
        """
        Convert default probability to 300–900 score.
        Higher score = lower risk (like CIBIL/FICO convention).
        """
        # Log-odds scaling for a more linear feel
        prob = np.clip(probability, 1e-6, 1 - 1e-6)
        log_odds = np.log(prob / (1 - prob))

        # Map log-odds range [-6, +6] to score range [900, 300]
        score = SCORE_MAX + (log_odds / 6) * (SCORE_MAX - SCORE_MIN) / 2
        return int(np.clip(score, SCORE_MIN, SCORE_MAX))

    def _get_band(self, score: int) -> tuple:
        for lo, hi, band, desc in BANDS:
            if lo <= score <= hi:
                return band, desc
        return "Unknown", ""

    def _get_top_factors(self, X: pd.DataFrame, top_n: int = 5) -> list:
        """Return top N factors driving the score, with direction."""
        shap_vals = self.explainer.shap_values(X)[0]

        factors = []
        for i, col in enumerate(self.feature_list):
            factors.append({
                "feature":   col,
                "shap":      float(shap_vals[i]),
                "value":     float(X[col].iloc[0]),
            })

        # Sort by absolute SHAP impact
        factors.sort(key=lambda x: abs(x["shap"]), reverse=True)

        result = []
        for f in factors[:top_n]:
            label = FEATURE_LABELS.get(f["feature"], f["feature"].replace("_", " ").lower())
            direction = "increasing risk" if f["shap"] > 0 else "decreasing risk"
            result.append({
                "factor":    label,
                "direction": direction,
                "impact":    round(abs(f["shap"]), 4),
            })

        return result
