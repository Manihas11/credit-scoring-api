"""
Credit Scoring - Data Ingestion DAG
Pipeline: Download → Validate → Clean → Store

Schedule: Daily at 2 AM IST
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    "owner": "credit_scoring",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# ── DAG definition ────────────────────────────────────────────────────────────
dag = DAG(
    dag_id="credit_data_ingestion",
    default_args=default_args,
    description="Ingest, validate and clean credit scoring data",
    schedule_interval="30 20 * * *",  # 2 AM IST = 8:30 PM UTC
    start_date=days_ago(1),
    catchup=False,
    tags=["credit-scoring", "ingestion", "month-1"],
)

# ── Task imports (inline to keep DAG self-contained) ─────────────────────────
def download_kaggle_data(**context):
    """Download Home Credit Default Risk dataset from Kaggle."""
    import os
    import subprocess
    from src.utils.logger import get_logger

    log = get_logger("download")
    raw_dir = "/opt/airflow/data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    # Check if already downloaded (avoid re-downloading on retries)
    target = os.path.join(raw_dir, "application_train.csv")
    if os.path.exists(target):
        log.info("Data already present, skipping download.")
        return target

    log.info("Downloading Home Credit dataset from Kaggle...")
    result = subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", "home-credit-default-risk",
            "-p", raw_dir,
        ],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed:\n{result.stderr}")

    log.info("Unzipping dataset...")
    subprocess.run(["unzip", "-o", f"{raw_dir}/home-credit-default-risk.zip", "-d", raw_dir])

    log.info(f"Download complete → {raw_dir}")
    context["task_instance"].xcom_push(key="raw_dir", value=raw_dir)
    return raw_dir


def validate_raw_data(**context):
    """Run Great Expectations suite on raw CSV files."""
    from src.validation.ge_runner import run_validation_suite

    raw_dir = context["task_instance"].xcom_pull(
        task_ids="download_data", key="raw_dir"
    ) or "/opt/airflow/data/raw"

    results = run_validation_suite(raw_dir)

    if not results["success"]:
        raise ValueError(
            f"Data validation FAILED.\n"
            f"Failed checks: {results['failed_checks']}"
        )

    context["task_instance"].xcom_push(key="validation_results", value=results)
    print(f"Validation passed: {results['passed_checks']} checks OK")


def clean_and_transform(**context):
    """Clean raw data: handle nulls, fix dtypes, engineer base features."""
    from src.ingestion.cleaner import CreditDataCleaner

    raw_dir = "/opt/airflow/data/raw"
    processed_dir = "/opt/airflow/data/processed"

    cleaner = CreditDataCleaner(raw_dir=raw_dir, output_dir=processed_dir)
    output_path = cleaner.run()

    context["task_instance"].xcom_push(key="processed_path", value=output_path)
    print(f"Clean data written → {output_path}")


def generate_data_profile(**context):
    """Generate a ydata-profiling HTML report for the processed data."""
    import pandas as pd
    from ydata_profiling import ProfileReport

    processed_path = context["task_instance"].xcom_pull(
        task_ids="clean_data", key="processed_path"
    ) or "/opt/airflow/data/processed/application_clean.parquet"

    df = pd.read_parquet(processed_path)
    profile = ProfileReport(
        df,
        title="Credit Scoring — Data Profile",
        minimal=True,  # fast mode; set False for full report
    )
    report_path = "/opt/airflow/data/processed/profile_report.html"
    profile.to_file(report_path)
    print(f"Profile report saved → {report_path}")


# ── Tasks ─────────────────────────────────────────────────────────────────────
t1_download = PythonOperator(
    task_id="download_data",
    python_callable=download_kaggle_data,
    dag=dag,
)

t2_validate = PythonOperator(
    task_id="validate_raw_data",
    python_callable=validate_raw_data,
    dag=dag,
)

t3_clean = PythonOperator(
    task_id="clean_data",
    python_callable=clean_and_transform,
    dag=dag,
)

t4_profile = PythonOperator(
    task_id="profile_data",
    python_callable=generate_data_profile,
    dag=dag,
)

t5_done = BashOperator(
    task_id="notify_complete",
    bash_command='echo "Pipeline complete at $(date). Check /data/processed/"',
    dag=dag,
)

# ── Dependency chain ──────────────────────────────────────────────────────────
t1_download >> t2_validate >> t3_clean >> t4_profile >> t5_done
