from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

from utils import mae, rmse, mape, r2

@dataclass
class TrainResult:
    model_name: str
    model: object
    metrics: dict

def time_split(df: pd.DataFrame, train_end="2016-12-01", test_start="2017-01-01"):
    train = df[df["month"] <= pd.to_datetime(train_end)].copy()
    test = df[df["month"] >= pd.to_datetime(test_start)].copy()
    return train, test

def train_all_models(features_df: pd.DataFrame) -> tuple[list[TrainResult], pd.DataFrame, pd.DataFrame, list[str]]:
    feature_cols = [c for c in features_df.columns if c not in ["month", "D_t", "S_t"]]
    train_df, test_df = time_split(features_df)

    X_train = train_df[feature_cols].values
    y_train = train_df["D_t"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["D_t"].values

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=2, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=42,
            max_iter=500,
            learning_rate=0.03,
            max_depth=6,
            min_samples_leaf=5,
        ),
    }

    results: list[TrainResult] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = y_pred.clip(min=0.0)
        metrics = {
            "mae": mae(y_test, y_pred),
            "rmse": rmse(y_test, y_pred),
            "mape": mape(y_test, y_pred),
            "r2": r2(y_test, y_pred),
        }
        results.append(TrainResult(model_name=name, model=model, metrics=metrics))

    return results, train_df, test_df, feature_cols
