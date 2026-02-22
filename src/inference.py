from __future__ import annotations
import pandas as pd

def predict(model, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].values
    y_pred = model.predict(X)
    out = pd.DataFrame({
        "month": df["month"].values,
        "y_true": df["D_t"].values,
        "y_pred": y_pred,
    })
    return out
