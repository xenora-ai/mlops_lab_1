from airflow.models import DagBag
import os


def test_dag_import():
    """Перевірка, що файли DAG не містять синтаксичних помилок"""
    dag_path = os.path.join(os.path.dirname(__file__), '../dags')
    dag_bag = DagBag(dag_folder=dag_path, include_examples=False)

    # Перевіряємо, чи немає помилок імпорту
    assert len(dag_bag.import_errors) == 0, f"DAG Import errors: {dag_bag.import_errors}"