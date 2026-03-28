"""
API Tests — Month 3
Tests the FastAPI endpoint without needing a running server.
Run: pytest tests/test_api.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Mock the predictor so tests don't need the real model ────────────────────
MOCK_RESULT = {
    "score":       742,
    "band":        "Good",
    "description": "Low risk — likely to repay comfortably",
    "probability": 0.0872,
    "top_factors": [
        {"factor": "average credit bureau score", "direction": "decreasing risk", "impact": 0.38},
        {"factor": "debt-to-income ratio",        "direction": "increasing risk",  "impact": 0.12},
        {"factor": "years of employment",          "direction": "decreasing risk", "impact": 0.08},
        {"factor": "loan-to-goods ratio",          "direction": "increasing risk",  "impact": 0.07},
        {"factor": "applicant age profile",        "direction": "decreasing risk", "impact": 0.05},
    ],
}


@pytest.fixture
def client():
    with patch("api.main.CreditPredictor") as MockPredictor:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = MOCK_RESULT
        MockPredictor.return_value = mock_instance

        import api.main as main_module
        main_module.predictor = mock_instance

        from fastapi.testclient import TestClient
        from api.main import app
        yield TestClient(app)


@pytest.fixture
def sample_payload():
    return {
        "annual_income":          360000,
        "loan_amount":            500000,
        "loan_annuity":           24000,
        "goods_price":            450000,
        "age_years":              34,
        "employed_years":         6,
        "gender":                 "M",
        "owns_car":               False,
        "owns_realty":            True,
        "num_children":           0,
        "family_size":            2,
        "ext_score_1":            0.62,
        "ext_score_2":            0.71,
        "ext_score_3":            0.58,
        "txn_count_30d":          28,
        "txn_amount_30d":         32000,
        "payment_regularity_30d": 0.85,
        "payment_regularity_90d": 0.80,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_score_endpoint_returns_200(client, sample_payload):
    r = client.post("/score", json=sample_payload)
    assert r.status_code == 200


def test_score_response_has_required_fields(client, sample_payload):
    r = client.post("/score", json=sample_payload)
    body = r.json()
    for field in ["score", "band", "description", "probability", "top_factors", "latency_ms"]:
        assert field in body, f"Missing field: {field}"


def test_score_in_valid_range(client, sample_payload):
    r = client.post("/score", json=sample_payload)
    score = r.json()["score"]
    assert 300 <= score <= 900


def test_probability_in_valid_range(client, sample_payload):
    r = client.post("/score", json=sample_payload)
    prob = r.json()["probability"]
    assert 0.0 <= prob <= 1.0


def test_top_factors_returned(client, sample_payload):
    r = client.post("/score", json=sample_payload)
    factors = r.json()["top_factors"]
    assert len(factors) >= 1
    assert "factor" in factors[0]
    assert "direction" in factors[0]
    assert "impact" in factors[0]


def test_direction_values_valid(client, sample_payload):
    r = client.post("/score", json=sample_payload)
    for f in r.json()["top_factors"]:
        assert f["direction"] in ("increasing risk", "decreasing risk")


def test_root_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Credit Scoring" in r.text


def test_missing_optional_fields_still_works(client):
    """API should work with minimal payload — all fields have defaults."""
    r = client.post("/score", json={"annual_income": 300000, "loan_amount": 400000})
    assert r.status_code == 200


def test_schemas_feature_mapping():
    """ScoreRequest.to_features() should return all expected keys."""
    from api.schemas import ScoreRequest
    req = ScoreRequest()
    features = req.to_features()
    required_keys = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AGE_YEARS",
        "EXT_SCORE_MEAN", "DEBT_TO_INCOME", "CAREER_FRACTION",
        "TXN_COUNT_30D", "PAYMENT_REGULARITY_90D",
    ]
    for key in required_keys:
        assert key in features, f"Missing feature key: {key}"
