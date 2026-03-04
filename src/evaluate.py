from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import joblib
import mlflow
import numpy as np
import pandas as pd

from common import ensure_dir, resolve_path
from metrics import best_f1_threshold, binary_metrics, classification_report_at_threshold, ks_stat


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError("Model does not support predict_proba or decision_function")


def _markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return ""
            return f"{v:.6f}"
        return str(v)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join([header, sep] + body)


def _deciles_table(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> pd.DataFrame:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    df = (
        pd.DataFrame({"y": y_true.astype(int), "p": y_prob.astype(float)})
        .sort_values("p", ascending=False)
        .reset_index(drop=True)
    )
    n = int(len(df))
    if n == 0:
        raise ValueError("Empty data for deciles")

    # 1..n_bins, bucket=1 is highest risk (largest probability)
    df["bucket"] = (np.arange(n) * n_bins // n) + 1

    agg = (
        df.groupby("bucket", as_index=False)
        .agg(
            n=("y", "size"),
            avg_p=("p", "mean"),
            default_rate=("y", "mean"),
            defaults=("y", "sum"),
        )
        .sort_values("bucket", ascending=True)
        .reset_index(drop=True)
    )

    agg["cum_defaults"] = agg["defaults"].cumsum()
    agg["cum_n"] = agg["n"].cumsum()

    total_defaults = float(agg["defaults"].sum())
    total_n = float(agg["n"].sum())
    base_rate = float(total_defaults / max(total_n, 1.0))

    agg["cum_default_capture"] = agg["cum_defaults"] / max(total_defaults, 1.0)
    agg["cum_population"] = agg["cum_n"] / max(total_n, 1.0)
    agg["lift"] = agg["default_rate"] / max(base_rate, 1e-12)
    return agg


def evaluate_and_report(cfg: Dict[str, Any]) -> None:
    """
    Metrics-only evaluation:
    - uses artifacts produced by `train_compare_and_select`
    - evaluates the selected best model on valid/test
    - writes tables (csv/md/json) and logs metrics/artifacts to MLflow
    """

    exp_name = cfg["experiment"]["name"]
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(exp_name)

    ensure_dir("data/reports")

    best_json_path = resolve_path("data/reports/best_model.json")
    if not os.path.exists(best_json_path):
        raise FileNotFoundError(f"Missing artifact: {best_json_path}. Run train step first.")

    with open(best_json_path, "r", encoding="utf-8") as f:
        best_meta = json.load(f)
    best_model = str(best_meta["best_model"])
    best_by = str(best_meta.get("best_by", "valid_pr_auc"))
    best_score = float(best_meta.get("best_score", float("nan")))

    cmp_path = resolve_path("data/reports/model_comparison.csv")
    if not os.path.exists(cmp_path):
        raise FileNotFoundError(f"Missing artifact: {cmp_path}. Run train step first.")
    cmp_df = pd.read_csv(cmp_path)

    train_df = pd.read_parquet(resolve_path("data/splits/train.parquet"))
    valid_df = pd.read_parquet(resolve_path("data/splits/valid.parquet"))
    test_df = pd.read_parquet(resolve_path("data/splits/test.parquet"))

    X_valid, y_valid = valid_df.drop(columns=["y"]), valid_df["y"].values.astype(int)
    X_test, y_test = test_df.drop(columns=["y"]), test_df["y"].values.astype(int)

    model_path = resolve_path(f"data/artifacts/model_{best_model}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model artifact not found: {model_path}")
    model = joblib.load(model_path)

    t0 = time.time()
    p_valid = _predict_proba(model, X_valid)
    p_test = _predict_proba(model, X_test)
    predict_seconds = float(time.time() - t0)

    # Metrics (threshold-free)
    valid_metrics = binary_metrics(y_valid, p_valid)
    test_metrics = binary_metrics(y_test, p_test)

    # Thresholds are chosen on valid only
    _, thr_f1 = best_f1_threshold(y_valid, p_valid)
    _, thr_ks = ks_stat(y_valid, p_valid)

    thr_rows = []
    for name, thr in [
        ("t=0.5", 0.5),
        ("t*=argmax F1 (valid)", float(thr_f1)),
        ("t*=argmax KS (valid)", float(thr_ks)),
    ]:
        rep = classification_report_at_threshold(y_test, p_test, float(thr))
        pred_pos_rate = float((p_test >= float(thr)).mean())
        thr_rows.append(
            {
                "policy": name,
                "threshold": float(thr),
                "predicted_default_rate": pred_pos_rate,
                "approval_rate": 1.0 - pred_pos_rate,
                "precision": float(rep["precision"]),
                "recall": float(rep["recall"]),
                "f1": float(rep["f1"]),
            }
        )

    thr_df = pd.DataFrame(thr_rows)
    thr_csv = resolve_path("data/reports/threshold_policies_best_model.csv")
    thr_df.to_csv(thr_csv, index=False)

    dec_df = _deciles_table(y_test, p_test, n_bins=10)
    dec_csv = resolve_path("data/reports/deciles_best_model.csv")
    dec_df.to_csv(dec_csv, index=False)

    generated_at = _now_utc_iso()

    # Metrics-only markdown report (no narrative)
    report_path = resolve_path("data/reports/final_report.md")
    cmp_cols = [
        "model",
        "valid_roc_auc",
        "valid_pr_auc",
        "valid_ks",
        "valid_f1_at_0_5",
        "test_roc_auc",
        "test_pr_auc",
        "test_ks",
        "test_f1_at_0_5",
        "fit_seconds",
    ]

    core_keys = ["roc_auc", "pr_auc", "ks", "f1_at_0_5"]
    best_cols = ["split"] + core_keys
    best_rows = [
        {"split": "valid", **{k: valid_metrics.get(k) for k in core_keys}},
        {"split": "test", **{k: test_metrics.get(k) for k in core_keys}},
    ]

    dataset_rows = [
        {"split": "train", "n": int(len(train_df)), "base_rate": float(train_df["y"].mean())},
        {"split": "valid", "n": int(len(valid_df)), "base_rate": float(valid_df["y"].mean())},
        {"split": "test", "n": int(len(test_df)), "base_rate": float(test_df["y"].mean())},
    ]

    dec_preview = dec_df.copy()
    for c in ["avg_p", "default_rate", "cum_default_capture", "cum_population", "lift"]:
        if c in dec_preview.columns:
            dec_preview[c] = dec_preview[c].astype(float).round(6)
    dec_block = dec_preview.to_string(index=False)

    thr_preview = thr_df.copy()
    for c in ["threshold", "predicted_default_rate", "approval_rate", "precision", "recall", "f1"]:
        if c in thr_preview.columns:
            thr_preview[c] = thr_preview[c].astype(float).round(6)
    thr_md = _markdown_table(thr_preview.to_dict(orient="records"), list(thr_preview.columns))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Metrics report (Airflow → MLflow)\n\n")
        f.write(f"- generated_at_utc: `{generated_at}`\n")
        f.write(f"- selection_metric: `{best_by}`\n")
        f.write(f"- best_model: `{best_model}` (score={best_score:.6f})\n")
        f.write(f"- split: train_end=`{cfg['split']['train_end']}`, valid_end=`{cfg['split']['valid_end']}`\n\n")

        f.write("## Dataset\n\n")
        f.write(_markdown_table(dataset_rows, ["split", "n", "base_rate"]))
        f.write("\n\n")

        f.write("## Model comparison (valid/test)\n\n")
        f.write(_markdown_table(cmp_df.to_dict(orient="records"), cmp_cols))
        f.write("\n\n")

        f.write("## Best model metrics\n\n")
        f.write(_markdown_table(best_rows, best_cols))
        f.write("\n\n")

        f.write("## Threshold policies (test; thresholds picked on valid)\n\n")
        f.write(thr_md)
        f.write("\n\n")

        f.write("## Deciles on test (bucket=1 is highest risk)\n\n")
        f.write("```text\n")
        f.write(dec_block)
        f.write("\n```\n")

    # Machine-readable summary
    summary_json = resolve_path("data/reports/best_model_metrics.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": generated_at,
                "best_model": best_model,
                "selected_by": best_by,
                "best_score": best_score,
                "split": {
                    "train_end": cfg["split"]["train_end"],
                    "valid_end": cfg["split"]["valid_end"],
                },
                "dataset": {r["split"]: {"n": r["n"], "base_rate": r["base_rate"]} for r in dataset_rows},
                "best_model_metrics": {
                    "valid": {k: float(valid_metrics[k]) for k in core_keys},
                    "test": {k: float(test_metrics[k]) for k in core_keys},
                },
                "thresholds_valid": {"f1_best_threshold": float(thr_f1), "ks_threshold": float(thr_ks)},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[metrics] best_model={best_model} selected_by={best_by} best_score={best_score:.6f}")
    print(
        "[metrics] test: "
        f"roc_auc={float(test_metrics['roc_auc']):.6f} "
        f"pr_auc={float(test_metrics['pr_auc']):.6f} "
        f"ks={float(test_metrics['ks']):.6f}"
    )

    with mlflow.start_run(run_name="metrics_report"):
        mlflow.log_param("best_model", best_model)
        mlflow.log_param("selected_by", best_by)
        mlflow.log_metric("best_score", best_score)
        mlflow.log_metric("predict_seconds", predict_seconds)
        mlflow.log_param("n_train", int(len(train_df)))
        mlflow.log_param("n_valid", int(len(valid_df)))
        mlflow.log_param("n_test", int(len(test_df)))

        mlflow.log_metrics({f"best_valid_{k}": float(valid_metrics[k]) for k in core_keys})
        mlflow.log_metrics({f"best_test_{k}": float(test_metrics[k]) for k in core_keys})

        mlflow.log_artifact(report_path)
        mlflow.log_artifact(thr_csv)
        mlflow.log_artifact(dec_csv)
        mlflow.log_artifact(summary_json)

        # Convenience: attach selection artifacts too
        mlflow.log_artifact(best_json_path)
        mlflow.log_artifact(cmp_path)
        cmp_md = resolve_path("data/reports/model_comparison.md")
        if os.path.exists(cmp_md):
            mlflow.log_artifact(cmp_md)
