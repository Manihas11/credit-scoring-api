"""
Credit Scoring API — Month 3
FastAPI endpoint: POST /score → credit score + reasons

Run locally:
    uvicorn api.main:app --reload

Deploy to Render/Railway:
    Set start command to: uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import time

from api.schemas import ScoreRequest, ScoreResponse, HealthResponse
from api.predictor import CreditPredictor
from src.utils.logger import get_logger

log = get_logger("api")

# ── Lifespan: load model once at startup ──────────────────────────────────────
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    log.info("Loading credit scoring model...")
    predictor = CreditPredictor(models_dir="models")
    log.info("Model loaded — API ready")
    yield
    log.info("Shutting down")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Scoring API",
    description="""
## Alternative Credit Scoring for Bharat

Scores credit-invisible applicants using non-traditional data:
UPI transaction patterns, employment stability, and behavioral signals.

**Score range:** 300 (very high risk) → 900 (excellent)

Built by a Data Engineer as an open-source MVP.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Landing page — investor demo entry point."""
    return """
    <html>
    <head>
        <title>Credit Scoring API</title>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 700px;
                   margin: 60px auto; padding: 0 24px; color: #1a1a1a; }
            h1   { font-size: 28px; margin-bottom: 4px; }
            .sub { color: #666; margin-bottom: 40px; }
            .card { background: #f8f8f8; border-radius: 12px;
                    padding: 24px; margin: 16px 0; }
            .score { font-size: 64px; font-weight: 700; color: #2d7d46; }
            .band  { font-size: 20px; color: #2d7d46; margin-top: -8px; }
            .btn   { display: inline-block; background: #1a1a1a; color: white;
                     padding: 12px 24px; border-radius: 8px; text-decoration: none;
                     font-size: 15px; margin-top: 24px; }
            .factor { padding: 8px 0; border-bottom: 1px solid #eee; font-size: 14px; }
            .up { color: #c0392b; } .down { color: #2d7d46; }
            code { background: #eee; padding: 2px 6px; border-radius: 4px; font-size: 13px; }
        </style>
    </head>
    <body>
        <h1>Credit Scoring API</h1>
        <p class="sub">Alternative credit scoring for India's credit-invisible population</p>

        <div class="card">
            <div class="score">742</div>
            <div class="band">Good — Low risk</div>
            <div style="margin-top:16px; font-size:14px; color:#444;">
                Sample output for a Working professional, age 34,
                income ₹3.6L/year, requesting ₹5L loan
            </div>
            <div style="margin-top: 16px;">
                <div class="factor"><span class="down">↓</span> Strong average credit bureau score</div>
                <div class="factor"><span class="down">↓</span> 6 years of stable employment</div>
                <div class="factor"><span class="up">↑</span> Loan-to-goods ratio slightly elevated</div>
                <div class="factor"><span class="down">↓</span> Regular payment pattern (90-day)</div>
            </div>
        </div>

        <a href="/docs" class="btn">Try the API →</a>
        &nbsp;&nbsp;
        <a href="/health" class="btn" style="background:#444">Health check</a>

        <p style="margin-top:40px; font-size:13px; color:#999;">
            Model: XGBoost · AUC-ROC: 0.7545 · Trained on 307,511 applicants ·
            <a href="/docs">Swagger docs</a>
        </p>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — used by Render/Railway to verify the service is up."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        version="1.0.0",
    )


@app.post("/score", response_model=ScoreResponse)
def score_applicant(request: ScoreRequest):
    """
    Score a credit applicant.

    Returns a 300–900 credit score, risk band, default probability,
    and the top factors driving the score (via SHAP).
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.time()

    try:
        features = request.to_features()
        result = predictor.predict(features)
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.time() - start) * 1000, 1)
    log.info(
        f"Scored applicant → {result['score']} ({result['band']}) "
        f"P={result['probability']:.4f} [{latency_ms}ms]"
    )

    return ScoreResponse(
        score=result["score"],
        band=result["band"],
        description=result["description"],
        probability=result["probability"],
        top_factors=result["top_factors"],
        latency_ms=latency_ms,
    )


@app.get("/sample-request", include_in_schema=False)
def sample_request():
    """Returns a ready-to-use sample request body for testing."""
    return ScoreRequest.sample()
