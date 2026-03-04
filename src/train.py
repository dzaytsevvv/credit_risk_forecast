from __future__ import annotations

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Sequence

import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier

from lightgbm import LGBMClassifier

from common import ensure_dir, resolve_path
from metrics import binary_metrics


CORE_METRICS = ("roc_auc", "pr_auc", "ks", "f1_at_0_5")


def build_preprocessor(df: pd.DataFrame, *, scale_numeric: bool):
    X = df.drop(columns=["y"])
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler(with_mean=False)))

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]), cat_cols),
        ],
        remainder="drop",
    )
    return pre


def _markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    def fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return ""
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join([header, sep] + body)


def _predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError("Model does not support predict_proba or decision_function")


def _fit_predict_proba(pipe: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray, X_eval: pd.DataFrame) -> np.ndarray:
    pipe.fit(X_train, y_train)
    return _predict_proba(pipe, X_eval)


def train_compare_and_select(cfg: Dict[str, Any]) -> None:
    exp_name = cfg["experiment"]["name"]
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(exp_name)

    train_df = pd.read_parquet(resolve_path("data/splits/train.parquet"))
    valid_df = pd.read_parquet(resolve_path("data/splits/valid.parquet"))
    test_df = pd.read_parquet(resolve_path("data/splits/test.parquet"))

    X_train, y_train = train_df.drop(columns=["y"]), train_df["y"].values
    X_valid, y_valid = valid_df.drop(columns=["y"]), valid_df["y"].values
    X_test, y_test = test_df.drop(columns=["y"]), test_df["y"].values

    # дисбаланс классов
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = float(neg / max(pos, 1))

    candidates = []

    # 1) Logistic Regression (interpretable baseline)
    logreg_params = {
        "solver": "saga",
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 200,
        "n_jobs": 2,
        "class_weight": "balanced",
        "random_state": int(cfg["model"]["random_state"]),
    }
    candidates.append(
        (
            "logreg",
            Pipeline([("pre", build_preprocessor(train_df, scale_numeric=True)), ("clf", LogisticRegression(**logreg_params))]),
            logreg_params,
        )
    )

    # 2) SGD Logistic Regression (fast baseline for large data)
    sgd_params = {
        "loss": "log_loss",
        "alpha": 1e-5,
        "penalty": "l2",
        "max_iter": 50,
        "n_jobs": 2,
        "class_weight": "balanced",
        "random_state": int(cfg["model"]["random_state"]),
    }
    candidates.append(
        (
            "sgd_logreg",
            Pipeline([("pre", build_preprocessor(train_df, scale_numeric=True)), ("clf", SGDClassifier(**sgd_params))]),
            sgd_params,
        )
    )

    # 3) LightGBM (strong tabular baseline)
    lgbm_params = dict(cfg["model"]["lgbm_params"])
    lgbm_params["random_state"] = int(cfg["model"]["random_state"])
    lgbm_params["n_jobs"] = 2
    lgbm_params["scale_pos_weight"] = scale_pos_weight
    candidates.append(
        (
            "lgbm",
            Pipeline([("pre", build_preprocessor(train_df, scale_numeric=False)), ("clf", LGBMClassifier(**lgbm_params))]),
            lgbm_params,
        )
    )

    ensure_dir("data/artifacts")
    ensure_dir("data/reports")

    rows = []
    for model_name, pipe, params in candidates:
        model_path = resolve_path(f"data/artifacts/model_{model_name}.joblib")

        with mlflow.start_run(run_name=f"model_{model_name}"):
            mlflow.log_param("model_name", model_name)
            for k, v in params.items():
                mlflow.log_param(k, v)

            t0 = time.time()
            p_valid = _fit_predict_proba(pipe, X_train, y_train, X_valid)
            fit_seconds = float(time.time() - t0)

            valid_metrics = binary_metrics(y_valid, p_valid)
            mlflow.log_metric("fit_seconds", fit_seconds)
            core_valid = {k: float(valid_metrics[k]) for k in CORE_METRICS}
            mlflow.log_metrics({f"valid_{k}": v for k, v in core_valid.items()})

            p_test = _predict_proba(pipe, X_test)
            test_metrics = binary_metrics(y_test, p_test)
            core_test = {k: float(test_metrics[k]) for k in CORE_METRICS}
            mlflow.log_metrics({f"test_{k}": v for k, v in core_test.items()})

            # feature importance / coefficients (lightweight, top features only)
            try:
                feature_names = pipe.named_steps["pre"].get_feature_names_out()
            except Exception:
                feature_names = None

            clf = pipe.named_steps.get("clf")
            fi_path = resolve_path(f"data/reports/feature_importance_{model_name}.csv")
            try:
                if feature_names is not None and hasattr(clf, "coef_"):
                    coefs = np.asarray(clf.coef_).reshape(-1)
                    fi = (
                        pd.DataFrame({"feature": feature_names, "coef": coefs})
                        .assign(abs_coef=lambda d: d["coef"].abs())
                        .sort_values("abs_coef", ascending=False)
                        .head(50)
                    )
                    fi.to_csv(fi_path, index=False)
                    mlflow.log_artifact(fi_path)
                elif feature_names is not None and hasattr(clf, "feature_importances_"):
                    imps = np.asarray(clf.feature_importances_).reshape(-1)
                    fi = (
                        pd.DataFrame({"feature": feature_names, "importance": imps})
                        .sort_values("importance", ascending=False)
                        .head(50)
                    )
                    fi.to_csv(fi_path, index=False)
                    mlflow.log_artifact(fi_path)
            except Exception:
                # do not fail the pipeline if importance extraction fails
                pass

            joblib.dump(pipe, model_path)
            try:
                mlflow.log_metric("model_size_bytes", float(os.path.getsize(model_path)))
            except OSError:
                pass
            mlflow.log_artifact(model_path)

            row = {
                "model": model_name,
                "fit_seconds": fit_seconds,
                "artifact_path": model_path,
                "scale_pos_weight": scale_pos_weight if model_name == "lgbm" else None,
            }
            row.update({f"valid_{k}": core_valid[k] for k in CORE_METRICS})
            row.update({f"test_{k}": core_test[k] for k in CORE_METRICS})
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(by=["valid_pr_auc", "valid_roc_auc"], ascending=False).reset_index(drop=True)

    best_model = str(df.loc[0, "model"])
    best_by = "valid_pr_auc"
    best_score = float(df.loc[0, best_by])

    summary_csv = resolve_path("data/reports/model_comparison.csv")
    df.to_csv(summary_csv, index=False)

    summary_md = resolve_path("data/reports/model_comparison.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Model comparison (valid/test)\n\n")
        f.write(f"- Selection metric: **{best_by}**\n")
        f.write(f"- Best model: **{best_model}** (score={best_score:.6f})\n\n")
        cols = [
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
        f.write(_markdown_table(df.to_dict(orient="records"), cols))
        f.write("\n")

    best_json = resolve_path("data/reports/best_model.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(
            {"best_model": best_model, "best_by": best_by, "best_score": best_score},
            f,
            ensure_ascii=False,
            indent=2,
        )

    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_param("best_model", best_model)
        mlflow.log_param("selection_metric", best_by)
        mlflow.log_metric("best_score", best_score)
        mlflow.log_artifact(summary_csv)
        mlflow.log_artifact(summary_md)
        mlflow.log_artifact(best_json)
