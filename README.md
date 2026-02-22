# VKR MLOps: LendingClub arrears forecasting (Airflow + MLflow + Docker)

## 1) Подготовка
1) Положи датасет в `data/raw/`
2) Скопируй `.env.example` -> `.env` и проверь `LC_RAW_PATH`

## 2) Запуск
```bash
docker compose up -d --build
```

## 3) UI сервисов
- Airflow: `http://localhost:8080`
- MLflow: `http://localhost:${MLFLOW_HOST_PORT:-5001}`

## 4) Примечание по DAG
- На первом запуске любого task Airflow создаст venv и установит зависимости из `requirements.txt`, поэтому первый task стартует дольше обычного.
