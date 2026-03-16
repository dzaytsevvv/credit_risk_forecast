# Lending Club (Simple MLOps): Airflow + MLflow + PostgreSQL

## Что делает проект
Airflow запускает простой ML-пайплайн:
1) prep_data: формирование таргета из loan_status, удаление leakage, time split
2) train_model: LightGBM + логирование параметров/метрик/модели в MLflow
3) evaluate: метрики на test (ROC-AUC, PR-AUC, KS, F1) в MLflow

## Подготовка данных
Скачай CSV с Kaggle Lending Club и положи:
`data/raw/lending_club.csv`

(или поменяй путь в configs/config.yaml)

По умолчанию `prep_data` читает CSV потоково и берет до `data.max_rows` строк
(параметры `data.chunksize` и `data.max_rows` в `configs/config.yaml`), чтобы не падать по OOM в Airflow контейнере.

## Запуск
```bash
cp .env.example .env
docker compose up -d
```

По умолчанию сервисы доступны на:
- Airflow: `http://localhost:8080` (логин/пароль: `admin/admin`)
- MLflow: `http://localhost:5001`

Если порты заняты, измени `POSTGRES_PORT`, `MLFLOW_PORT`, `AIRFLOW_PORT` в `.env`.
