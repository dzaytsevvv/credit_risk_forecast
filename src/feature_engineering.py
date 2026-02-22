from __future__ import annotations
import math
import pandas as pd

def make_features(series: pd.DataFrame, target_col: str = "D_t") -> pd.DataFrame:
    df = series.copy()

    df["month_num"] = df["month"].dt.month
    df["year"] = df["month"].dt.year
    df["month_sin"] = df["month_num"].apply(lambda m: math.sin(2 * math.pi * m / 12.0))
    df["month_cos"] = df["month_num"].apply(lambda m: math.cos(2 * math.pi * m / 12.0))

    for l in [1, 2, 3, 6, 12]:
        df[f"{target_col}_lag{l}"] = df[target_col].shift(l)

    for w in [3, 6, 12]:
        df[f"{target_col}_ma{w}"] = df[target_col].shift(1).rolling(w).mean()
        df[f"{target_col}_std{w}"] = df[target_col].shift(1).rolling(w).std()

    feature_cols = [c for c in df.columns if c not in ["month", "P_t", "D_t", "S_t"]]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df
