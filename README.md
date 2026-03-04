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

## Запуск
```bash
cp .env.example .env
docker compose up -d