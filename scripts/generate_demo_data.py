from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def main() -> None:
    rng = np.random.default_rng(42)

    months = pd.date_range("2010-01-01", "2021-12-01", freq="MS")
    rows: list[dict] = []

    for m in months:
        n_loans = int(rng.integers(600, 1300))
        seasonal = 0.6 + 0.4 * np.sin(2 * np.pi * (m.month / 12.0))
        bad_rate = np.clip(0.06 + 0.04 * seasonal + rng.normal(0, 0.01), 0.01, 0.25)

        for _ in range(n_loans):
            loan_amnt = float(np.round(np.exp(rng.normal(8.8, 0.5)) / 10) * 10)
            is_bad = rng.random() < bad_rate
            loan_status = "Charged Off" if is_bad else "Fully Paid"
            rows.append(
                {
                    "issue_d": m.strftime("%Y-%m-%d"),
                    "loan_amnt": loan_amnt,
                    "loan_status": loan_status,
                }
            )

    df = pd.DataFrame(rows)
    out_path = Path("data/raw/accepted_loans.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Demo dataset created: {out_path} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
