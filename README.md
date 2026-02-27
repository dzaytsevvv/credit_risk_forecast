# VKR MLOps: LendingClub arrears forecasting (Airflow + MLflow + Docker)

## 1) Подготовка
1) Положи датасет в `data/raw/`
2) Скопируй `.env.example` -> `.env` и проверь `LC_RAW_PATH`
3) Если датасета пока нет, можно сгенерировать демо-набор:

```bash
python scripts/generate_demo_data.py
```

## 2) Запуск
```bash
docker compose up -d --build
```

## 3) Локальный запуск без Docker (быстрая проверка)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# опционально: демо-данные
python scripts/generate_demo_data.py

export PYTHONPATH=src
export LC_RAW_PATH=data/raw/accepted_loans.csv
export OUTPUT_DIR=data/processed
export MLFLOW_TRACKING_URI=file:./mlruns

python -m mlops build-series
python -m mlops build-features
python -m mlops train
python -m mlops score
```

## 4) UI сервисов
- Airflow: `http://localhost:8080`
- MLflow: `http://localhost:${MLFLOW_HOST_PORT:-5001}`

## 5) Примечание по DAG
- На первом запуске любого task Airflow создаст venv и установит зависимости из `requirements.txt`, поэтому первый task стартует дольше обычного.
