from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    lc_raw_path: str
    output_dir: str
    db_dsn: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str

def get_settings() -> Settings:
    return Settings(
        lc_raw_path=os.getenv("LC_RAW_PATH", "/opt/project/data/raw/accepted_loans.csv"),
        output_dir=os.getenv("OUTPUT_DIR", "/opt/project/data/processed"),
        db_dsn=os.getenv("DB_DSN", "postgresql://lc:lcpass@localhost:5432/lc_mlops"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "LendingClub_ArrearsForecast"),
    )
