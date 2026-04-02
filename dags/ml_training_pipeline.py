from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import BranchPythonOperator
from datetime import datetime, timedelta
import json
import os

# Визначаємо поріг якості моделі
QUALITY_THRESHOLD = 0.85


def check_model_quality():
    """Перевірка метрик із файлу metrics.json"""
    metrics_path = "/opt/airflow/metrics.json"
    if not os.path.exists(metrics_path):
        return "stop_pipeline"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Перевіряємо метрику r2
    if metrics.get("r2", 0) >= QUALITY_THRESHOLD:
        return "register_model"
    return "stop_pipeline"


with DAG(
    "ml_salary_prediction_pipeline",
    default_args={
        "start_date": datetime(2026, 4, 2),
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    schedule="@weekly",
    catchup=False,
) as dag:
    # 1. Перевірка доступності даних
    check_data = BashOperator(
        task_id="check_raw_data",
        bash_command="ls /opt/airflow/data/raw/global_graduate_employability_index.csv",
    )

    # 2. Підготовка даних
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="python /opt/airflow/src/prepare.py /opt/airflow/data/raw/global_graduate_employability_index.csv /opt/airflow/data/prepared",
    )

    # 3. Тренування та оптимізація
    train_model = BashOperator(
        task_id="train_model",
        bash_command='python /opt/airflow/src/optimize.py hpo.n_trials=5 mlflow.tracking_uri="file:/opt/airflow/mlruns"',
    )

    # 4. Логіка розгалуження (Quality Gate)
    evaluate_model = BranchPythonOperator(
        task_id="evaluate_model", python_callable=check_model_quality
    )

    # 5. Реєстрація моделі
    register_model = BashOperator(
        task_id="register_model",
        bash_command='echo "Модель пройшла поріг якості та зареєстрована."',
    )

    # 6. Зупинка пайплайну
    stop_pipeline = BashOperator(
        task_id="stop_pipeline", bash_command='echo "Якість моделі недостатня."'
    )

    check_data >> prepare_data >> train_model >> evaluate_model
    evaluate_model >> [register_model, stop_pipeline]
