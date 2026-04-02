from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.dates import days_ago
import json
import os

QUALITY_THRESHOLD = 0.85


def check_model_quality():
    metrics_path = '/opt/airflow/metrics.json'
    if not os.path.exists(metrics_path):
        return 'stop_pipeline'

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    if metrics.get('r2', 0) >= QUALITY_THRESHOLD:
        return 'register_model'
    return 'stop_pipeline'


with DAG(
        'ml_salary_prediction_pipeline',
        default_args={'start_date': days_ago(1)},
        schedule_interval='@weekly',
        catchup=False
) as dag:
    check_data = BashOperator(
        task_id='check_raw_data',
        bash_command='ls /opt/airflow/data/raw/global_graduate_employability_index.csv'
    )

    prepare_data = BashOperator(
        task_id='prepare_data',
        bash_command='python /opt/airflow/src/prepare.py /opt/airflow/data/raw/global_graduate_employability_index.csv /opt/airflow/data/prepared'
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command='python /opt/airflow/src/optimize.py hpo.n_trials=5 mlflow.tracking_uri="file:/opt/airflow/mlruns"'
    )

    evaluate_model = BranchPythonOperator(
        task_id='evaluate_model',
        python_callable=check_model_quality
    )

    register_model = BashOperator(
        task_id='register_model',
        bash_command='echo "Модель пройшла поріг якості та зареєстрована в MLflow Registry"'
    )

    stop_pipeline = BashOperator(
        task_id='stop_pipeline',
        bash_command='echo "Якість моделі недостатня. Реєстрацію скасовано."'
    )

    check_data >> prepare_data >> train_model >> evaluate_model
    evaluate_model >> [register_model, stop_pipeline]
