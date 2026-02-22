from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="credit_forecast_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # вручную для ВКР (можно потом cron)
    catchup=False,
    tags=["vkr", "mlops", "lendingclub"],
) as dag:

    # Все шаги запускают CLI внутри изолированного venv,
    # чтобы не ломать зависимости самого Airflow.
    base = "bash /opt/airflow/scripts/run_mlops_step.sh"

    build_series = BashOperator(
        task_id="build_series",
        bash_command=f"{base} build-series",
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"{base} build-features",
    )

    train = BashOperator(
        task_id="train",
        bash_command=f"{base} train",
    )

    score = BashOperator(
        task_id="score",
        bash_command=f"{base} score",
    )

    persist = BashOperator(
        task_id="persist",
        bash_command=f"{base} persist",
    )

    build_series >> build_features >> train >> score >> persist
