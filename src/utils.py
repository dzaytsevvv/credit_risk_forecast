import os
import json
import numpy as np
import psycopg2
from contextlib import contextmanager

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def mae(y, yhat): return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))
def rmse(y, yhat): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))
def mape(y, yhat, eps=1e-9):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat) / np.maximum(np.abs(y), eps))) * 100.0)
def r2(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(0.0 if ss_tot == 0 else (1 - ss_res / ss_tot))

@contextmanager
def db_conn(dsn: str):
    conn = psycopg2.connect(dsn)
    try:
        yield conn
    finally:
        conn.close()
