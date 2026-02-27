from __future__ import annotations
import pandas as pd

LC_USECOLS = ["issue_d", "loan_amnt", "loan_status"]

def read_lendingclub(path: str) -> pd.DataFrame:
    # pandas сам понимает .csv.gz
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in LC_USECOLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")
    return df[LC_USECOLS].copy()

BAD_STATUSES = {
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Charged Off",
    "Default",
}

def build_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Портфельный временной ряд по месяцу выдачи:
        P_t = sum(loan_amnt)
        D_t = sum(loan_amnt) where loan_status in BAD_STATUSES
        S_t = D_t / P_t
    """
    x = df.copy()
    x["issue_d"] = pd.to_datetime(x["issue_d"], errors="coerce")
    x = x.dropna(subset=["issue_d", "loan_amnt", "loan_status"])
    x["month"] = x["issue_d"].dt.to_period("M").dt.to_timestamp()
    x["is_bad"] = x["loan_status"].astype(str).isin(BAD_STATUSES)

    portfolio = x.groupby("month", as_index=False)["loan_amnt"].sum().rename(columns={"loan_amnt": "P_t"})
    bad = x[x["is_bad"]].groupby("month", as_index=False)["loan_amnt"].sum().rename(columns={"loan_amnt": "D_t"})

    out = portfolio.merge(bad, on="month", how="left").fillna({"D_t": 0.0})
    out["S_t"] = (out["D_t"] / out["P_t"]).fillna(0.0)
    out = out.sort_values("month").reset_index(drop=True)
    return out
