from __future__ import annotations

import os
import re
import pandas as pd
from typing import Any, Dict

from common import ensure_dir, resolve_path

DEFAULT_STATUSES = {"Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)"}
GOOD_STATUSES = {"Fully Paid"}

LEAKAGE_PATTERNS = [
    r"^last_pymnt_",
    r"^next_pymnt_",
    r"^total_rec_",
    r"^recoveries$",
    r"^collection_recovery_fee$",
    r"^out_prncp",
    r"^total_pymnt",
]


def parse_issue_d(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if dt.isna().mean() > 0.2:
        dt = pd.to_datetime(s, format="%b-%Y", errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()


def preprocess_and_split(cfg: Dict[str, Any]) -> None:
    raw_path = resolve_path(cfg["data"]["raw_csv_path"])
    issue_col = cfg["data"]["issue_date_col"]
    status_col = cfg["data"]["loan_status_col"]

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw CSV not found: {raw_path}\n"
            f"Скачай CSV с Kaggle и положи в data/raw/lending_club.csv (или измени путь в config.yaml)."
        )

    df = pd.read_csv(raw_path, low_memory=False)

    if issue_col not in df.columns:
        raise ValueError(f"В датасете нет колонки {issue_col}")
    if status_col not in df.columns:
        raise ValueError(f"В датасете нет колонки {status_col}")

    df[issue_col] = parse_issue_d(df[issue_col])

    # target
    df = df[df[status_col].isin(DEFAULT_STATUSES | GOOD_STATUSES)].copy()
    df["y"] = df[status_col].isin(DEFAULT_STATUSES).astype(int)
    df.drop(columns=[status_col], inplace=True, errors="ignore")

    # leakage drop
    drop_cols = []
    for c in df.columns:
        for pat in LEAKAGE_PATTERNS:
            if re.search(pat, c):
                drop_cols.append(c)
                break
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # drop too-empty columns
    na_frac = df.isna().mean()
    too_empty = na_frac[na_frac > 0.95].index.tolist()
    if too_empty:
        df.drop(columns=too_empty, inplace=True)

    ensure_dir("data/processed")
    df.to_parquet(resolve_path("data/processed/processed.parquet"), index=False)

    # time split
    df = df.dropna(subset=[issue_col]).copy()

    def make_split(train_end_ts: pd.Timestamp, valid_end_ts: pd.Timestamp):
        train_df = df[df[issue_col] <= train_end_ts].copy()
        valid_df = df[(df[issue_col] > train_end_ts) & (df[issue_col] <= valid_end_ts)].copy()
        test_df = df[df[issue_col] > valid_end_ts].copy()
        return train_df, valid_df, test_df

    train_end = pd.to_datetime(cfg["split"]["train_end"])
    valid_end = pd.to_datetime(cfg["split"]["valid_end"])
    train, valid, test = make_split(train_end, valid_end)

    if min(len(train), len(valid), len(test)) == 0:
        unique_dates = sorted(df[issue_col].dropna().unique())
        if len(unique_dates) < 3:
            raise ValueError(
                f"Недостаточно уникальных значений {issue_col} для time split: {len(unique_dates)}. "
                f"Проверь колонку {issue_col} и границы split."
            )

        # Auto-fix invalid boundaries (keeps train/valid/test non-empty)
        train_idx = max(0, min(int(len(unique_dates) * 0.70) - 1, len(unique_dates) - 3))
        valid_idx = max(train_idx + 1, min(int(len(unique_dates) * 0.85) - 1, len(unique_dates) - 2))
        train_end = pd.to_datetime(unique_dates[train_idx])
        valid_end = pd.to_datetime(unique_dates[valid_idx])
        train, valid, test = make_split(train_end, valid_end)

        if min(len(train), len(valid), len(test)) == 0:
            raise ValueError(
                f"Плохие границы сплита: train={len(train)}, valid={len(valid)}, test={len(test)}. "
                "Проверь split/train_end и split/valid_end."
            )

        print(
            "Split boundaries auto-adjusted "
            f"(train_end={train_end.date()}, valid_end={valid_end.date()}) "
            f"to keep splits non-empty: train={len(train)}, valid={len(valid)}, test={len(test)}"
        )

    ensure_dir("data/splits")
    # issue_d is used for time split; remove it from features to avoid datetime dtype issues in sklearn pipelines
    for part in (train, valid, test):
        part.drop(columns=[issue_col], inplace=True, errors="ignore")
    train.to_parquet(resolve_path("data/splits/train.parquet"), index=False)
    valid.to_parquet(resolve_path("data/splits/valid.parquet"), index=False)
    test.to_parquet(resolve_path("data/splits/test.parquet"), index=False)
