from __future__ import annotations

import os
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/airflow/project")
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from common import load_config
from data_prep import preprocess_and_split
from train import train_compare_and_select
from evaluate import evaluate_and_report


with DAG(
    dag_id="lendingclub_simple_pipeline_pg",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["simple", "postgres", "mlflow", "airflow"],
) as dag:

    def _prep():
        cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
        preprocess_and_split(cfg)

    def _train():
        cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
        train_compare_and_select(cfg)

    def _eval():
        cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
        evaluate_and_report(cfg)

    prep = PythonOperator(task_id="prep_data", python_callable=_prep)
    train = PythonOperator(task_id="train_model", python_callable=_train)
    ev = PythonOperator(task_id="evaluate", python_callable=_eval)

    prep >> train >> ev
