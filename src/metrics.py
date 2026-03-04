from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_stat(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ks_vals = tpr - fpr
    idx = int(np.nanargmax(ks_vals))
    return float(ks_vals[idx]), float(thr[idx])


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0:
        return 0.0, 0.5

    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    idx = int(np.nanargmax(f1))
    return float(f1[idx]), float(thr[idx])


def classification_report_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    roc = float(roc_auc_score(y_true, y_prob))
    pr = float(average_precision_score(y_true, y_prob))
    ks, ks_thr = ks_stat(y_true, y_prob)
    best_f1, best_thr = best_f1_threshold(y_true, y_prob)

    out: Dict[str, Any] = {
        "roc_auc": roc,
        "pr_auc": pr,
        "ks": ks,
        "ks_threshold": ks_thr,
        "f1_best": best_f1,
        "f1_best_threshold": best_thr,
    }

    out.update({f"at_0_5_{k}": v for k, v in classification_report_at_threshold(y_true, y_prob, 0.5).items() if k != "threshold"})
    out["f1_at_0_5"] = out.pop("at_0_5_f1")
    out["precision_at_0_5"] = out.pop("at_0_5_precision")
    out["recall_at_0_5"] = out.pop("at_0_5_recall")
    out["tn_at_0_5"] = out.pop("at_0_5_tn")
    out["fp_at_0_5"] = out.pop("at_0_5_fp")
    out["fn_at_0_5"] = out.pop("at_0_5_fn")
    out["tp_at_0_5"] = out.pop("at_0_5_tp")
    return out
