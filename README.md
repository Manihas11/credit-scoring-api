# Credit Scoring — Month 1: Data Ingestion Pipeline

Alternative credit scoring for India's 190M credit-invisible population.
Built on UPI transactions, utility payments, and mobile behavior data.

---

## Project structure

```
credit_scoring/
├── dags/
│   └── credit_ingestion_dag.py   ← Airflow DAG (download → validate → clean → profile)
├── src/
│   ├── ingestion/
│   │   └── cleaner.py            ← Null handling, feature derivation, outlier capping
│   ├── validation/
│   │   └── ge_runner.py          ← Schema, range, uniqueness checks
│   └── utils/
│       └── logger.py             ← Shared logger
├── tests/
│   └── test_cleaner.py           ← 7 pytest unit tests
├── config/
│   └── settings.py               ← All config in one place
├── data/
│   ├── raw/                      ← Downloaded CSVs land here
│   └── processed/                ← Clean parquet files + profile report
├── notebooks/                    ← EDA notebooks (add as you explore)
├── docker-compose.yml            ← Spin up Airflow locally
└── requirements.txt
```

---

## Quickstart (local, no Docker)

```bash
# 1. Create virtualenv
python -m venv venv && source venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Set Kaggle credentials (get from kaggle.com → Account → API)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# 4. Run the cleaner directly (skip Airflow for now)
python -c "
from src.ingestion.cleaner import CreditDataCleaner
c = CreditDataCleaner(raw_dir='data/raw', output_dir='data/processed')
c.run()
"

# 5. Run tests
pytest tests/ -v
```

---

## Quickstart (with Airflow via Docker)

```bash
# 1. Create .env file with Kaggle credentials
echo "KAGGLE_USERNAME=your_username" > .env
echo "KAGGLE_KEY=your_api_key" >> .env

# 2. Start Airflow
docker compose up -d

# 3. Open UI → http://localhost:8080
#    Login: airflow / airflow
#    Trigger DAG: credit_data_ingestion

# 4. Check output
ls data/processed/
# application_clean.parquet
# profile_report.html
```

---

## What the pipeline does

```
Download (Kaggle API)
  └─► Validate (14 checks: nulls, ranges, uniques, categories)
        └─► Clean (sentinels, imputation, encoding, outlier cap)
              └─► Feature engineering (age, ratios, windows)
                    └─► Parquet output + HTML profile report
```

### Key output columns added by the cleaner

| Column | Description |
|---|---|
| `AGE_YEARS` | Applicant age derived from `DAYS_BIRTH` |
| `EMPLOYED_YEARS` | Employment duration (365243 sentinel → NaN → median) |
| `CREDIT_TO_INCOME` | Loan amount ÷ income (debt load proxy) |
| `ANNUITY_TO_INCOME` | Monthly payment ÷ income (repayment stress) |
| `CHILDREN_RATIO` | Children ÷ family size |

---

## Month 2 preview (feature store + model)

- Feast feature store (local mode)
- 30/60/90-day rolling behavioral windows
- XGBoost with Optuna hyperparameter tuning
- SHAP for per-applicant explanations
- MLflow for experiment tracking

---

## Month 3 preview (API)

- FastAPI `POST /score` endpoint
- Returns `{ score: 742, band: "Good", top_factors: [...] }`
- Deployed free on Render or Railway
- Streamlit dashboard for investor demo
