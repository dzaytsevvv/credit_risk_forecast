from __future__ import annotations
import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from config import get_settings
from utils import ensure_dir, save_json, db_conn, mae, rmse, mape, r2
from data_ingestion import read_lendingclub, build_monthly_series
from feature_engineering import make_features
from modeling import train_all_models
from inference import predict


TARGET_COLS = ["month", "D_t", "S_t"]

def path_series(out_dir: str) -> str:
    return os.path.join(out_dir, "monthly_series.csv")

def path_features(out_dir: str) -> str:
    return os.path.join(out_dir, "features.csv")

def path_preds(out_dir: str) -> str:
    return os.path.join(out_dir, "predictions.csv")

def cmd_build_series(args):
    s = get_settings()
    ensure_dir(s.output_dir)
    df = read_lendingclub(s.lc_raw_path)
    series = build_monthly_series(df)
    out = path_series(s.output_dir)
    series.to_csv(out, index=False)
    print(out)

def cmd_build_features(args):
    s = get_settings()
    series = pd.read_csv(path_series(s.output_dir), parse_dates=["month"])
    feats = make_features(series, target_col="D_t")
    out = path_features(s.output_dir)
    feats.to_csv(out, index=False)
    print(out)

def cmd_train(args):
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment_name)

    feats = pd.read_csv(path_features(s.output_dir), parse_dates=["month"])

    results, train_df, test_df, feature_cols = train_all_models(feats)

    report = {}
    for r in results:
        with mlflow.start_run(run_name=f"train_{r.model_name}"):
            mlflow.log_param("model_name", r.model_name)
            mlflow.log_param("train_end", "2016-12-01")
            mlflow.log_param("test_start", "2017-01-01")
            mlflow.log_metrics(r.metrics)

            mlflow.sklearn.log_model(r.model, artifact_path="model")
            report[r.model_name] = r.metrics

    save_json(os.path.join(s.output_dir, "train_metrics.json"), report)
    print("OK train. Metrics saved:", os.path.join(s.output_dir, "train_metrics.json"))

def cmd_score(args):
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment_name)

    feats = pd.read_csv(path_features(s.output_dir), parse_dates=["month"])

    # Находим лучший run по RMSE
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(s.mlflow_experiment_name)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No MLflow runs found. Run train first.")

    best = runs[0]
    run_id = best.info.run_id
    model_name = best.data.params.get("model_name", "unknown")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # test часть такая же, как в train_all_models
    test_df = feats[feats["month"] >= pd.to_datetime("2017-01-01")].copy()
    feature_cols = [c for c in feats.columns if c not in TARGET_COLS]

    pred_df = predict(model, test_df, feature_cols)
    pred_df["model_name"] = model_name
    pred_df["run_id"] = run_id

    out = path_preds(s.output_dir)
    pred_df.to_csv(out, index=False)
    print("OK score. Predictions saved:", out)

def cmd_persist(args):
    s = get_settings()
    pred_df = pd.read_csv(path_preds(s.output_dir), parse_dates=["month"])

    # метрики на тесте
    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }

    model_name = str(pred_df["model_name"].iloc[0])
    run_id = str(pred_df["run_id"].iloc[0])

    with db_conn(s.db_dsn) as conn:
        conn.autocommit = True
        cur = conn.cursor()

        # predictions
        for _, r in pred_df.iterrows():
            cur.execute(
                "INSERT INTO predictions(model_name, run_id, month, y_true, y_pred) VALUES (%s,%s,%s,%s,%s)",
                (model_name, run_id, r["month"].date(), float(r["y_true"]), float(r["y_pred"]))
            )

        # model_metrics
        cur.execute(
            "INSERT INTO model_metrics(model_name, run_id, split_name, mae, rmse, mape, r2) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (model_name, run_id, "test", metrics["mae"], metrics["rmse"], metrics["mape"], metrics["r2"])
        )

        cur.close()

    save_json(os.path.join(s.output_dir, "test_metrics.json"), metrics)
    print("OK persist. Metrics:", metrics)

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-series").set_defaults(func=cmd_build_series)
    sub.add_parser("build-features").set_defaults(func=cmd_build_features)
    sub.add_parser("train").set_defaults(func=cmd_train)
    sub.add_parser("score").set_defaults(func=cmd_score)
    sub.add_parser("persist").set_defaults(func=cmd_persist)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
