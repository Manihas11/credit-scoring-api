"""
API Schemas — Request and Response models
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ScoreRequest(BaseModel):
    """
    Applicant data for credit scoring.
    All fields are optional — missing values are handled gracefully.
    """

    # ── Income & loan ─────────────────────────────────────────────────────────
    annual_income: float = Field(
        default=180000, ge=0,
        description="Annual income in INR",
        example=360000,
    )
    loan_amount: float = Field(
        default=500000, ge=0,
        description="Requested loan amount in INR",
        example=500000,
    )
    loan_annuity: float = Field(
        default=25000, ge=0,
        description="Monthly loan annuity / EMI in INR",
        example=24000,
    )
    goods_price: float = Field(
        default=450000, ge=0,
        description="Price of goods the loan is for",
        example=450000,
    )

    # ── Demographics ──────────────────────────────────────────────────────────
    age_years: float = Field(
        default=34, ge=18, le=80,
        description="Applicant age in years",
        example=34,
    )
    employed_years: float = Field(
        default=6, ge=0,
        description="Years at current employer",
        example=6,
    )
    gender: str = Field(
        default="M",
        description="Gender: M or F",
        example="M",
    )
    owns_car: bool = Field(default=False, description="Owns a car")
    owns_realty: bool = Field(default=True, description="Owns real estate")

    # ── Family ────────────────────────────────────────────────────────────────
    num_children: int = Field(default=0, ge=0, le=20, description="Number of children")
    family_size: int = Field(default=2, ge=1, le=20, description="Total family members")

    # ── External credit bureau scores (0–1) ──────────────────────────────────
    ext_score_1: Optional[float] = Field(default=0.5, ge=0, le=1, description="Bureau score 1")
    ext_score_2: Optional[float] = Field(default=0.6, ge=0, le=1, description="Bureau score 2")
    ext_score_3: Optional[float] = Field(default=0.55, ge=0, le=1, description="Bureau score 3")

    # ── Behavioral (UPI/transaction proxies) ─────────────────────────────────
    txn_count_30d: int = Field(default=25, ge=0, description="Transactions in last 30 days")
    txn_amount_30d: float = Field(default=30000, ge=0, description="Total spend last 30 days (INR)")
    payment_regularity_30d: float = Field(default=0.8, ge=0, le=1, description="Payment regularity score (0–1)")
    payment_regularity_90d: float = Field(default=0.75, ge=0, le=1, description="Payment regularity score 90-day (0–1)")

    class Config:
        json_schema_extra = {
            "example": {
                "annual_income": 360000,
                "loan_amount": 500000,
                "loan_annuity": 24000,
                "goods_price": 450000,
                "age_years": 34,
                "employed_years": 6,
                "gender": "M",
                "owns_car": False,
                "owns_realty": True,
                "num_children": 0,
                "family_size": 2,
                "ext_score_1": 0.62,
                "ext_score_2": 0.71,
                "ext_score_3": 0.58,
                "txn_count_30d": 28,
                "txn_amount_30d": 32000,
                "payment_regularity_30d": 0.85,
                "payment_regularity_90d": 0.80,
            }
        }

    def to_features(self) -> dict:
        """Convert API request to the feature dict the model expects."""
        import numpy as np

        income = self.annual_income or 1
        ext_scores = [s for s in [self.ext_score_1, self.ext_score_2, self.ext_score_3] if s]

        employed_years = self.employed_years
        age_years = self.age_years
        career_fraction = min(employed_years / age_years, 1.0) if age_years > 0 else 0

        txn_amount_90d = self.txn_amount_30d * 3 * 0.95   # proxy
        spend_trend = (self.txn_amount_30d / (txn_amount_90d / 3)) if txn_amount_90d > 0 else 1.0

        return {
            # Raw
            "AMT_INCOME_TOTAL":     self.annual_income,
            "AMT_CREDIT":           self.loan_amount,
            "AMT_ANNUITY":          self.loan_annuity,
            "AGE_YEARS":            self.age_years,
            "EMPLOYED_YEARS":       self.employed_years,
            "CNT_CHILDREN":         self.num_children,
            "CNT_FAM_MEMBERS":      self.family_size,
            "EXT_SOURCE_1":         self.ext_score_1 or 0.5,
            "EXT_SOURCE_2":         self.ext_score_2 or 0.5,
            "EXT_SOURCE_3":         self.ext_score_3 or 0.5,
            "CODE_GENDER":          1 if self.gender == "M" else 0,
            "FLAG_OWN_CAR":         int(self.owns_car),
            "FLAG_OWN_REALTY":      int(self.owns_realty),

            # Engineered
            "CREDIT_TO_INCOME":     round(self.loan_amount / income, 4),
            "ANNUITY_TO_INCOME":    round(self.loan_annuity / income, 4),
            "CHILDREN_RATIO":       round(self.num_children / self.family_size, 4) if self.family_size else 0,
            "DEBT_TO_INCOME":       round(self.loan_amount / income, 4),
            "MONTHLY_PAYMENT_RATIO": round(self.loan_annuity / (income / 12), 4),
            "LOAN_TO_GOODS":        round(self.loan_amount / self.goods_price, 4) if self.goods_price else 1,
            "CREDIT_OVERRUN":       round(self.loan_amount - self.goods_price, 2),
            "REGISTRATION_STABILITY": age_years * 0.6,
            "ID_STABILITY":         age_years * 0.5,
            "CAREER_FRACTION":      round(career_fraction, 4),
            "AGE_RISK_BAND":        self._age_risk_band(self.age_years),
            "EMPLOYMENT_RISK":      int(self.employed_years < 1),
            "INCOME_PER_PERSON":    round(income / self.family_size, 2) if self.family_size else income,

            # External score aggregates
            "EXT_SCORE_MEAN":  round(np.mean(ext_scores), 4),
            "EXT_SCORE_MIN":   round(np.min(ext_scores), 4),
            "EXT_SCORE_MAX":   round(np.max(ext_scores), 4),
            "EXT_SCORE_RANGE": round(np.max(ext_scores) - np.min(ext_scores), 4),
            "EXT_SCORE_PROD":  round(np.prod(ext_scores), 4),

            # Behavioral
            "TXN_COUNT_30D":           self.txn_count_30d,
            "TXN_AMOUNT_30D":          self.txn_amount_30d,
            "PAYMENT_REGULARITY_30D":  self.payment_regularity_30d,
            "TXN_COUNT_60D":           self.txn_count_30d * 2,
            "TXN_AMOUNT_60D":          self.txn_amount_30d * 2,
            "PAYMENT_REGULARITY_60D":  (self.payment_regularity_30d + self.payment_regularity_90d) / 2,
            "TXN_COUNT_90D":           self.txn_count_30d * 3,
            "TXN_AMOUNT_90D":          txn_amount_90d,
            "PAYMENT_REGULARITY_90D":  self.payment_regularity_90d,
            "SPEND_TREND_30_90":       round(spend_trend, 4),
        }

    @staticmethod
    def _age_risk_band(age: float) -> int:
        if age < 25:   return 4
        if age < 35:   return 2
        if age < 50:   return 1
        if age < 65:   return 3
        return 5

    @classmethod
    def sample(cls) -> dict:
        return cls().model_dump()


class FactorItem(BaseModel):
    factor: str
    direction: str
    impact: float


class ScoreResponse(BaseModel):
    score: int = Field(description="Credit score 300–900")
    band: str = Field(description="Risk band: Excellent / Good / Fair / Poor / Very Poor")
    description: str = Field(description="Plain-English risk summary")
    probability: float = Field(description="Raw default probability (0–1)")
    top_factors: List[FactorItem] = Field(description="Top 5 factors driving the score")
    latency_ms: float = Field(description="Prediction latency in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "score": 742,
                "band": "Good",
                "description": "Low risk — likely to repay comfortably",
                "probability": 0.0872,
                "top_factors": [
                    {"factor": "average credit bureau score", "direction": "decreasing risk", "impact": 0.38},
                    {"factor": "high debt-to-income ratio",   "direction": "increasing risk", "impact": 0.12},
                    {"factor": "years of employment",          "direction": "decreasing risk", "impact": 0.08},
                ],
                "latency_ms": 42.3,
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
